import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Input, Lambda, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import GlorotUniform, Orthogonal
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from mlp import MLP
from pinball import PinballLoss, PinballMetric
from data_generator import TimeSeriesSequence
from custom_model import CustomModel

from constants import DTYPE, HORIZON_LENGTH, QUANTILES, MAX_SERIES_LENGTH, LOSS_SIZE, AGGREGATE_EVALUATION_SALES_PATH, ALL_FEATURES_DATA_PATH, CALENDAR_DATA_PATH, INCLUDE_VALIDATION_METRIC

"""
TODO
Eventually make this file into a function with hyperparameters and filenames passed as parameters...
Right now just have to duplicate the file and update runtime constants and model constants
"""

## RUNTIME CONSTANTS
TESTING = True # UPDATE!!
PREDICT_FROM_SAVED_WEIGHTS = False
MODEL_FOLDER = './models/model1/' # UPDATE!!
MODEL_WEIGHTS_FILE = MODEL_FOLDER + 'weights.02' # UPDATE!! if PREDICT_FROM_SAVED_WEIGHTS is True

PREDICTION_FILE = MODEL_FOLDER + 'test-forecasts.csv'
if not PREDICT_FROM_SAVED_WEIGHTS:
    Path(MODEL_FOLDER).mkdir(parents=True, exist_ok=False)
    MODEL_WEIGHTS_FILE_FORMAT = MODEL_FOLDER + 'weights.{epoch:02d}'

## MODEL CONSTANTS
EPOCHS = 25 if not TESTING else 2
BATCH_SIZE = 32
NUM_OBSERVATIONS = 40 # for testing

TEMPORAL_CONTEXT_SIZE = 2 * len(QUANTILES)
TIME_AGNOSTIC_CONTEXT_SIZE = 10
RNN_UNITS = 64
GLOBAL_OUTPUT_SIZE = TIME_AGNOSTIC_CONTEXT_SIZE + HORIZON_LENGTH * TEMPORAL_CONTEXT_SIZE
GLOBAL_DECODER_SHAPES = (100, GLOBAL_OUTPUT_SIZE)
LOCAL_DECODER_SHAPES = (32, len(QUANTILES))

OPTIMIZER = 'adam'

FIT_VERBOSITY = 2
PREDICT_VERBOSITY = 1 if TESTING else 0

## LOAD DATA
FEATURES = ['department', 'category', 'store', 'state'] # 'item'
'./models/model-final-light/test-forecasts.csv'

sales_df = pd.read_csv(AGGREGATE_EVALUATION_SALES_PATH)
aggregate_features_df = pd.read_csv(ALL_FEATURES_DATA_PATH).loc[:, FEATURES]
if TESTING: 
    sales_df = sales_df.iloc[0:NUM_OBSERVATIONS]
    aggregate_features_df = aggregate_features_df.iloc[0:NUM_OBSERVATIONS]

encoder = OneHotEncoder(categories='auto', drop='first', sparse=False)
one_hot_features = encoder.fit_transform(aggregate_features_df)

train_sales_columns = sales_df.columns[2:]
x_train = sales_df.loc[:, train_sales_columns].to_numpy()

# DATA DEPENDENT CONSTANTS
NUM_STATIC_FEATURES = one_hot_features.shape[1]
NUM_DAY_LABELS = 6 + 3 # includes snap days
calendar_map = {'National': 2, 'Religious': 3, 'Cultural': 4, 'Sporting': 5} # NaN is 0, Xmas is 1
TIME_OBSERVATION_DIMENSION = NUM_STATIC_FEATURES + NUM_DAY_LABELS + 1 # TODO eventually add 1 for price

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

data_sequence = TimeSeriesSequence(x_train, one_hot_features, binary_day_labels, BATCH_SIZE)

## MODEL
K.clear_session()
tf.random.set_seed(0)

_input_series = Input(shape=(MAX_SERIES_LENGTH, TIME_OBSERVATION_DIMENSION), dtype=DTYPE, name='time_series')
_input_pred_features = Input(shape=(MAX_SERIES_LENGTH, HORIZON_LENGTH, TIME_OBSERVATION_DIMENSION-1), dtype=DTYPE, name='predict_features')
# TODO sample normalization?

rnn_model = LSTM(RNN_UNITS, return_sequences=True, kernel_initializer=GlorotUniform(seed=1), recurrent_initializer=Orthogonal(seed=2))
full_rnn_outputs = rnn_model(_input_series) # output shape: (batch_size, timesteps, units)

MLP.set_next_seed(3)
global_decoder = MLP(layer_shapes=GLOBAL_DECODER_SHAPES, name='global_decoder')
local_decoder = MLP(layer_shapes=LOCAL_DECODER_SHAPES, name='local_decoder')

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

all_local_decodings = tf.stack([local_decoder(forecast_input) for forecast_input in forecast_inputs])
all_local_decodings = tf.transpose(all_local_decodings, perm=[1, 2, 0, 3])

if HORIZON_LENGTH > 1:
    decoding_split_sizes = [LOSS_SIZE, HORIZON_LENGTH-1, 1]
else:
    decoding_split_sizes = [LOSS_SIZE, 1]

train_forecasts, _, quantile_forecasts = tf.split(all_local_decodings, num_or_size_splits=decoding_split_sizes, axis=-3, name='split_outputs')
train_forecasts = Lambda(lambda x: x, name='train_forecasts')(train_forecasts)
quantile_forecasts = Lambda(lambda x: x, name='quantile_forecasts')(quantile_forecasts)

model = CustomModel(inputs=[_input_series, _input_pred_features], outputs=[quantile_forecasts, train_forecasts])
metrics = {'quantile_forecasts': PinballMetric()} if INCLUDE_VALIDATION_METRIC else None
model.compile(loss={'train_forecasts': PinballLoss()}, optimizer=OPTIMIZER, metrics=metrics) # keras Model doesn't use sample weighting for metrics

model.summary(positions=[50, 110, 120, 170])
# with open('model-summary.txt', 'w') as file:
#     model.summary(print_fn=lambda x: file.write(x + '\n'))

## TRAIN
def print_dictionary(d):
    for key, lst in d.items():
        print('{}: {}'.format(key, lst))

if not PREDICT_FROM_SAVED_WEIGHTS:
    history = model.fit(data_sequence, epochs=EPOCHS, verbose=FIT_VERBOSITY,
                        callbacks=[
                            EarlyStopping(monitor='loss', patience=3, restore_best_weights=True),
                            ModelCheckpoint(MODEL_WEIGHTS_FILE_FORMAT, monitor='loss', save_best_only=True, save_weights_only=True, verbose=1),
                        ])
    print('finished training with history: ')
    print_dictionary(history.history)
else:
    model.load_weights(MODEL_WEIGHTS_FILE)
    print('finished loading weights')

## PREDICT
quantile_forecasts = model.predict(data_sequence, verbose=PREDICT_VERBOSITY)
print('finished predicting')
np.savetxt(PREDICTION_FILE, np.reshape(quantile_forecasts, (-1, HORIZON_LENGTH * len(QUANTILES))), delimiter=',')

## How to Load Prediction File
# loaded_quantile_forecasts = np.loadtxt(PREDICTION_FILE, delimiter=',')
# loaded_quantile_forecasts = np.reshape(loaded_quantile_forecasts, (-1, HORIZON_LENGTH, len(QUANTILES)))
