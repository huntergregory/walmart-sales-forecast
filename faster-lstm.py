import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from time import time

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Input, Lambda, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import GlorotUniform, Orthogonal

from constants import DTYPE, HORIZON_LENGTH, QUANTILES, MAX_SERIES_LENGTH, LOSS_SIZE

from mlp import MLP
from pinball import pinball, PinballMetric

"""
TODO PINBALL LOSS
- look at histogram of sales for all item-level stuff
    - if similar, leave pinball as is
    - eventually add scaling
    - eventually add masking for last H observations
TODO
- normalize each sample (individually)?
- other time series and hierarchy
- speed
"""

## MODEL CONSTANTS
TEMPORAL_CONTEXT_SIZE = 2 * len(QUANTILES)
TIME_AGNOSTIC_CONTEXT_SIZE = 10
RNN_UNITS = 128

EPOCHS = 3
BATCH_SIZE = 32
VERBOSITY = 3

## LOAD DATA
sales_df = pd.read_csv('../forecasting/data/sales_train_validation.csv')
train_sales_columns = sales_df.columns[6:]
NUM_OBSERVATIONS = 512 # FIXME stop using (used for testing)
x_train = sales_df.loc[0:(NUM_OBSERVATIONS-1), train_sales_columns].to_numpy() # TODO make whole df

features = sales_df.loc[:,['dept_id', 'store_id']]
encoder = OneHotEncoder(categories='auto', drop='first', sparse=False)
one_hot_features = encoder.fit_transform(features)
one_hot_features = one_hot_features[0:NUM_OBSERVATIONS, :] # TODO remove

## DATA DEPENDENT CONSTANTS
NUM_STATIC_FEATURES = one_hot_features.shape[1]
assert(MAX_SERIES_LENGTH == len(sales_df.columns) - 6)
TIME_OBSERVATION_DIMENSION = 1 # eventually add holiday --> one hot encode holidays and get dim length

## MODEL
K.clear_session()
tf.random.set_seed(0)

_input_series = Input(shape=(MAX_SERIES_LENGTH, TIME_OBSERVATION_DIMENSION), dtype=DTYPE, name='time_series')
_input_static_features = Input(shape=(MAX_SERIES_LENGTH, NUM_STATIC_FEATURES), dtype=DTYPE, name='static_features') 
# MQ-RNN has static features in each time step
# _input_eval_time_features = Input(shape=(HORIZON_LENGTH, TIME_OBSERVATION_DIMENSION-1), dtype=DTYPE, name='forecast_info') TODO INCLUDE

elongated_static_features = _input_static_features # K.repeat(_input_static_features, MAX_SERIES_LENGTH)
rnn_input_series = Concatenate(name='add_static_features')([_input_series, elongated_static_features])

rnn_model = LSTM(RNN_UNITS, return_sequences=True, kernel_initializer=GlorotUniform(seed=1), recurrent_initializer=Orthogonal(seed=2))
full_rnn_outputs = rnn_model(rnn_input_series) # output shape: (batch_size, timesteps, units)
# TODO? pass state to MLP decoders? make stateful=True for the rnn_model

global_output_size = TIME_AGNOSTIC_CONTEXT_SIZE + HORIZON_LENGTH * TEMPORAL_CONTEXT_SIZE
MLP.set_next_seed(3)
global_decoder = MLP(layer_shapes=(64, global_output_size), name='global_decoder')
local_decoder = MLP(layer_shapes=(64, len(QUANTILES)), name='local_decoder') 
# TODO? Dropout(0.15)(final)

all_global_decodings = global_decoder(full_rnn_outputs)

context_split_sizes = [TIME_AGNOSTIC_CONTEXT_SIZE] + HORIZON_LENGTH * [TEMPORAL_CONTEXT_SIZE]
all_context = tf.split(all_global_decodings, num_or_size_splits=context_split_sizes, axis=-1, name='context_split')
time_agnostic_context = all_context[0]

forecast_inputs = []
for k in range(HORIZON_LENGTH):
    features = _input_static_features
    concat_name = 'horizon_{}'.format(k+1)
    forecast_input = Concatenate(name=concat_name)([time_agnostic_context, all_context[k+1], elongated_static_features])
    # TODO include _input_eval_time_features[k, :] (except better slice)
    forecast_inputs.append(forecast_input)

# forecast_input = tf.stack(forecast_inputs, name='forecast_input_stack')
# all_local_decodings = local_decoder(forecast_inputs)
all_local_decodings = tf.stack([local_decoder(forecast_input) for forecast_input in forecast_inputs])
all_local_decodings = tf.transpose(all_local_decodings, perm=[1, 2, 0, 3])

# TODO eventually subtract only 1, remove 2nd size, and mask those in between end and end - HORIZON_LENGTH
# update loss and model.fit() input when that happens too
if HORIZON_LENGTH > 1:
    decoding_split_sizes = [LOSS_SIZE, HORIZON_LENGTH-1, 1]
else:
    decoding_split_sizes = [LOSS_SIZE, 1]

train_forecasts, _, quantile_forecasts = tf.split(all_local_decodings, num_or_size_splits=decoding_split_sizes, axis=-3, name='split_outputs')
train_forecasts = Lambda(lambda x: x, name='train_forecasts')(train_forecasts)
quantile_forecasts = Lambda(lambda x: x, name='quantile_forecasts')(quantile_forecasts)

model = Model(inputs=[_input_series, _input_static_features], outputs=[quantile_forecasts, train_forecasts])
model.compile(loss={'train_forecasts': pinball}, 
              optimizer='adam', 
              metrics={'train_forecasts': PinballMetric()})
model.summary(positions=[50, 110, 120, 170])
# with open('model-summary.txt', 'w') as file:
    # model.summary(print_fn=lambda x: file.write(x + '\n'))

## TRAIN
before1 = time()
y_train = []
for i in range(x_train.shape[0]):
    series = x_train[i]
    augmented_series = []
    y_train.append(augmented_series)
    for j in range(1, LOSS_SIZE + 1):
        augmented_series.append([len(QUANTILES) * [series[j+k]] for k in range(HORIZON_LENGTH)])
y_train = np.array(y_train)
after1 = time()
after1 - before1

one_hot_features = K.repeat(tf.convert_to_tensor(one_hot_features, dtype=DTYPE), MAX_SERIES_LENGTH)
print("x train shape: {}\ny train shape: {}\none hot features shape: {}".format(x_train.shape, y_train.shape, one_hot_features.shape))

history = model.fit([x_train, one_hot_features],
                    {'train_forecasts': y_train}, 
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_split=0.2,
                    verbose=VERBOSITY)
                    # callbacks=[free_memory()]
                                #EarlyStopping('val_f1_macro', patience=7, mode='max', restore_best_weights=False),

print('done')

## OTHER STUFF
# model.predict([response,context,df_features])
#             df1out, df2out = Lambda(lambda x: tf.split(x,num_or_size_splits=2,axis=0))(finalsig)   
#             self.model.compile(loss=['binary_crossentropy','binary_crossentropy'],
#                             optimizer='adam',  metrics=[F1Macro(), 'accuracy'], loss_weights = [1, 1/p])

# def print_MB(array):
#     print("{} MB".format(array.size * array.itemsize / 1024 / 1024))