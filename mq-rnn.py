import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from time import time

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Input, Lambda, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import GlorotUniform, Orthogonal
from tensorflow.keras.callbacks import EarlyStopping

from constants import DTYPE, HORIZON_LENGTH, QUANTILES, MAX_SERIES_LENGTH, LOSS_SIZE, FULL_SALES_DATA_PATH, ALL_FEATURES_DATA_PATH, CALENDAR_DATA_PATH

from mlp import MLP
from pinball import PinballLoss
from data_generator import TimeSeriesSequence

"""
TODO PINBALL LOSS
- look at histogram of sales for all item-level stuff
    - if similar, leave pinball as is
    - eventually add scaling
    - eventually add masking for last H observations
TODO
- normalize each sample (individually)?
-  hierarchy
- speed
"""

## CONSTANTS
TESTING = True # UPDATE!!!!
PREDS_SAVE_FILE = "test-forecasts.csv" ### UPDATE!!!!
MODEL_SAVE_FILE = 'models/model1' ### UPDATE!!!

TEMPORAL_CONTEXT_SIZE = 2 * len(QUANTILES)
TIME_AGNOSTIC_CONTEXT_SIZE = 10
RNN_UNITS = 128
GLOBAL_OUTPUT_SIZE = TIME_AGNOSTIC_CONTEXT_SIZE + HORIZON_LENGTH * TEMPORAL_CONTEXT_SIZE
GLOBAL_DECODER_SHAPES = (64, GLOBAL_OUTPUT_SIZE)
LOCAL_DECODER_SHAPES = (64, len(QUANTILES))

EPOCHS = 1 if TESTING else 20
BATCH_SIZE = 32
VAL_SPLIT = 0.2
FIT_VERBOSITY = 2
PREDICT_VERBOSITY = 1 if TESTING else 0
NUM_OBSERVATIONS = 64 # for testing

## LOAD DATA
FEATURES = ['department', 'category', 'store', 'state'] # 'item', 'department'

sales_df = pd.read_csv(FULL_SALES_DATA_PATH)
aggregate_features_df = pd.read_csv(ALL_FEATURES_DATA_PATH).loc[:, FEATURES]
if TESTING: 
    sales_df = sales_df.iloc[0:NUM_OBSERVATIONS]
    aggregate_features_df = aggregate_features_df.iloc[0:NUM_OBSERVATIONS]

encoder = OneHotEncoder(categories='auto', drop='first', sparse=False)
one_hot_features = encoder.fit_transform(aggregate_features_df)

train_sales_columns = sales_df.columns[6:]
item_store_ids = [(row.item_id, row.store_id) for _, row in sales_df.iterrows()]
x_train = sales_df.loc[:, train_sales_columns].to_numpy()

# DATA DEPENDENT CONSTANTS
# assert(MAX_SERIES_LENGTH == len(sales_df.columns) - 6)
NUM_STATIC_FEATURES = one_hot_features.shape[1]
NUM_DAY_LABELS = 6 + 3 # includes snap days
calendar_map = {'National': 2, 'Religious': 3, 'Cultural': 4, 'Sporting': 5} # NaN is 0, Xmas is 1
TIME_OBSERVATION_DIMENSION = NUM_STATIC_FEATURES + NUM_DAY_LABELS + 1 # TODO add 1 for price 
# MQ-RNN has static features in each time step

calendar = pd.read_csv(CALENDAR_DATA_PATH)
binary_day_labels = np.zeros((len(calendar), NUM_DAY_LABELS-3))
for k in range(len(calendar)):
    day = calendar.iloc[k]
    if day.event_name_1 is np.nan:
        binary_day_labels[k][0] = 1
    else:
        if day.event_name_1 == 'Christmas':
            binary_day_labels[k][1] = 1
        else:
            binary_day_labels[k][calendar_map[day.event_type_1]] = 1
        if day.event_name_2 == 'Christmas':
            binary_day_labels[k][1] = 1
        elif day.event_name_2 is not np.nan:
            binary_day_labels[k][calendar_map[day.event_type_2]] = 1

binary_day_labels = np.concatenate([binary_day_labels, np.array(calendar.loc[:, ['snap_CA', 'snap_TX', 'snap_WI']])], axis=-1)

data_sequence = TimeSeriesSequence(x_train, item_store_ids, one_hot_features, binary_day_labels, BATCH_SIZE, use_weights=True)



## MODEL
K.clear_session()
tf.random.set_seed(0)

_input_series = Input(shape=(MAX_SERIES_LENGTH, TIME_OBSERVATION_DIMENSION), dtype=DTYPE, name='time_series')
_input_pred_features = Input(shape=(MAX_SERIES_LENGTH, HORIZON_LENGTH, TIME_OBSERVATION_DIMENSION-1), dtype=DTYPE, name='predict_features')
# _input_val_forecasts = Input(shape=(HORIZON_LENGTH, len(QUANTILES)), name='input_val_forecasts')
# _input_eval_time_features = Input(shape=(HORIZON_LENGTH, TIME_OBSERVATION_DIMENSION-1), dtype=DTYPE, name='forecast_info') TODO INCLUDE

rnn_model = LSTM(RNN_UNITS, return_sequences=True, kernel_initializer=GlorotUniform(seed=1), recurrent_initializer=Orthogonal(seed=2))
full_rnn_outputs = rnn_model(_input_series) # output shape: (batch_size, timesteps, units)
# TODO? pass state to MLP decoders? make stateful=True for the rnn_model

MLP.set_next_seed(3)
global_decoder = MLP(layer_shapes=GLOBAL_DECODER_SHAPES, name='global_decoder')
local_decoder = MLP(layer_shapes=LOCAL_DECODER_SHAPES, name='local_decoder') 
# TODO? Dropout(0.15)(final)

all_global_decodings = global_decoder(full_rnn_outputs)

context_split_sizes = [TIME_AGNOSTIC_CONTEXT_SIZE] + HORIZON_LENGTH * [TEMPORAL_CONTEXT_SIZE]
all_context = tf.split(all_global_decodings, num_or_size_splits=context_split_sizes, axis=-1, name='context_split')
time_agnostic_context = all_context[0]

forecast_inputs = []
for k in range(HORIZON_LENGTH):
    features = _input_pred_features[:, :, k, :]
    concat_name = 'horizon_{}'.format(k+1)
    forecast_input = Concatenate(name=concat_name)([time_agnostic_context, all_context[k+1], features])
    forecast_inputs.append(forecast_input)

# forecast_input = tf.stack(forecast_inputs, name='forecast_input_stack')
# all_local_decodings = local_decoder(forecast_inputs)
all_local_decodings = tf.stack([local_decoder(forecast_input) for forecast_input in forecast_inputs])
all_local_decodings = tf.transpose(all_local_decodings, perm=[1, 2, 0, 3])

# TODO eventually subtract only 1, remove 2nd size, and mask those in between end and end - HORIZON_LENGTH
# update loss and model.fit() input when that happens too
if HORIZON_LENGTH > 1: # TODO replace with just [MAX_SERIES_LENGTH - 1, 1]
    decoding_split_sizes = [LOSS_SIZE, HORIZON_LENGTH-1, 1]
else:
    decoding_split_sizes = [LOSS_SIZE, 1]

train_forecasts, _, quantile_forecasts = tf.split(all_local_decodings, num_or_size_splits=decoding_split_sizes, axis=-3, name='split_outputs')
train_forecasts = Lambda(lambda x: x, name='train_forecasts')(train_forecasts)
quantile_forecasts = Lambda(lambda x: x, name='quantile_forecasts')(quantile_forecasts)

model = Model(inputs=[_input_series, _input_pred_features], outputs=[quantile_forecasts, train_forecasts])
model.compile(loss={'train_forecasts': PinballLoss()}, 
              optimizer='adam',)
            #   metrics={'train_forecasts': PinballMetric()})
# model.summary(positions=[50, 110, 120, 170])
# with open('model-summary.txt', 'w') as file:
    # model.summary(print_fn=lambda x: file.write(x + '\n'))

## TRAIN
history = model.fit(data_sequence, epochs=EPOCHS, verbose=FIT_VERBOSITY,
                    callbacks=[EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)])  #TODO validation set
                    # free_memory()])

print('done')

# model.save(MODEL_SAVE_FILE)
# loaded_model = tf.keras.models.load_model(MODEL_SAVE_FILE, compile=False)

test_forecasts, _ = model.predict(data_sequence, verbose=PREDICT_VERBOSITY)
print('all done')

np.savetxt(PREDS_SAVE_FILE, np.reshape(test_forecasts, (-1, HORIZON_LENGTH * len(QUANTILES))), delimiter=',')

loaded_test_forecasts = np.loadtxt(PREDS_SAVE_FILE, delimiter=',')
loaded_test_forecasts = np.reshape(loaded_test_forecasts, (-1, HORIZON_LENGTH, len(QUANTILES)))

print(np.all(loaded_test_forecasts == np.reshape(test_forecasts, (-1, HORIZON_LENGTH, len(QUANTILES)))))

print('ok')


## OTHER STUFF
# model.predict([response,context,df_features])
#             df1out, df2out = Lambda(lambda x: tf.split(x,num_or_size_splits=2,axis=0))(finalsig)   
#             self.model.compile(loss=['binary_crossentropy','binary_crossentropy'],
#                             optimizer='adam',  metrics=[F1Macro(), 'accuracy'], loss_weights = [1, 1/p])

# def print_MB(array):
#     print('{} MB'.format(array.size * array.itemsize / 1024 / 1024))