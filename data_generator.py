import numpy as np
import pandas as pd
import pickle
import math
from random import shuffle, seed
from tensorflow.keras.utils import Sequence


from constants import LOSS_SIZE, QUANTILES, HORIZON_LENGTH, MAX_SERIES_LENGTH, SELL_PRICES_DATA_PATH, FIRST_DAY_FILE, SELL_PRICE_INDICES_FILE, FIRST_WEEK, WEIGHTS_PATH
# from tqdm import tqdm

with open(FIRST_DAY_FILE, 'rb') as file:
    first_sale_days = pickle.load(file)

with open(SELL_PRICE_INDICES_FILE, 'rb') as file:
    item_price_indices = pickle.load(file) # length of 3,049

sell_price_df = pd.read_csv(SELL_PRICES_DATA_PATH)

sample_weights = np.array(pd.read_csv(WEIGHTS_PATH).spl_scaled_weight)

## CURRENT RUNTIME: 18.2 / 2
## 2nd optimization: 15.1 / 2 (or 16.6, 15.4)
## 3rd optimization: 8.8 / 2 (or 9.5)

class TimeSeriesSequence(Sequence):
    # time_series: (n, MAX_SERIES_LENGTH)
    # item_store_ids: (n) [tuple of (item_id, store_id)]
    # static_features: (n, num_static_labels). One-hot encoded
    # day_labels: (MAX_SERIES_LENGTH + x, num_day_labels). Binary
    # shuffle: boolean whether to shuffle data or not
    # use_weights: boolean whether to include sample weights in output or not
    def __init__(self, time_series, static_features, day_labels, batch_size, should_shuffle=True, use_weights=True):
        self.series_indices = [k for k in range(time_series.shape[0])]
        if should_shuffle:
            seed(50)
            shuffle(self.series_indices)

        self.time_series = time_series
        self.static_features = np.reshape(static_features, (static_features.shape[0], 1, static_features.shape[1]))
        self.batch_size = batch_size
        self.tiled_day_labels = np.tile(np.reshape(day_labels, (1,) + day_labels.shape), (self.batch_size, 1, 1))
        self.use_weights = use_weights

    def __len__(self):
        return math.ceil(len(self.time_series) / self.batch_size)

    # returns tuple of 2 or 3
    # x: [(batch_size, MAX_SERIES_LENGTH, 1 + num_static_labels + num_day_labels), (batch_size, MAX_SERIES_LENGTH, HORIZON_LENGTH, num_static_labels + num_day_labels)]
    # y: {'train_forecasts': (batch_size, MAX_SERIES_LENGTH - HORIZON_LENGTH, HORIZON_LENGTH, num_quantiles)}
    # w: (batch_size,) [optional]
    def __getitem__(self, idx): # TODO back to __ at start
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        batch_indices = self.series_indices[start:end]
        actual_size = self.batch_size if end < len(self.time_series) else len(self.time_series) - idx * self.batch_size

        tiled_static_features = np.tile(self.static_features[batch_indices], (1, MAX_SERIES_LENGTH + HORIZON_LENGTH, 1))
        tiled_day_labels = self.tiled_day_labels[:actual_size]

        features_concats = [
            tiled_day_labels[:, :(MAX_SERIES_LENGTH + HORIZON_LENGTH)], 
            tiled_static_features[:, :(MAX_SERIES_LENGTH + HORIZON_LENGTH)]
        ]
        features = np.concatenate(features_concats, axis=-1)

        time_series_batch = self.time_series[batch_indices, :MAX_SERIES_LENGTH]
        x_batch = np.reshape(time_series_batch, time_series_batch.shape + (1,))
        x_batch_concats = [x_batch, features[:, :MAX_SERIES_LENGTH]]
        x_batch = np.concatenate(x_batch_concats, axis=-1)

        y_batch = np.zeros((actual_size, LOSS_SIZE, HORIZON_LENGTH, len(QUANTILES)))
        val_y_batch = np.zeros((actual_size, 1, HORIZON_LENGTH, len(QUANTILES)))
        pred_features = np.zeros((actual_size, MAX_SERIES_LENGTH, HORIZON_LENGTH, features.shape[-1]))
        for i in range(actual_size):
            feature_series = features[i]

            for j in range(1, MAX_SERIES_LENGTH+1):
                should_expand = j-1 < LOSS_SIZE
                is_val_point = j == MAX_SERIES_LENGTH

                for k in range(HORIZON_LENGTH):
                    pred_features[i,j-1,k,:] = feature_series[j+k]
                    if should_expand:
                        y_batch[i,j-1,k,:] = self.time_series[i,j+k]
                    if is_val_point:
                        val_y_batch[i,0,k,:] = self.time_series[i,j+k]
        # runtime now: 4, 4.1, 4.4, 4.5
        # with lists: 7.7, 7.1

        full_batch = (
            {'time_series': x_batch, 'predict_features': pred_features},
            {'train_forecasts': y_batch, 'quantile_forecasts': val_y_batch}
        )
        if self.use_weights:
            return full_batch + (sample_weights[batch_indices],)
        return full_batch
