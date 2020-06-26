import numpy as np
import math
from time import time
import tensorflow
from tensorflow.keras.utils import Sequence

from constants import LOSS_SIZE, QUANTILES, HORIZON_LENGTH, MAX_SERIES_LENGTH
# from tqdm import tqdm

class TimeSeriesSequence(Sequence):
    # time_series: (n, MAX_SERIES_LENGTH)
    # static_features: (n, num_static_labels). One-hot encoded
    # day_labels: (MAX_SERIES_LENGTH + x, num_day_labels). Binary
    # weights: (n,). Leave as None for no sample weighting
    def __init__(self, time_series, static_features, day_labels, batch_size, weights=None):
        self.time_series = time_series
        self.static_features = np.reshape(static_features, (static_features.shape[0], 1, static_features.shape[1]))
        self.day_labels = np.reshape(day_labels, (1,) + day_labels.shape)
        self.batch_size = batch_size
        self.weights = weights

    def __len__(self):
        return math.ceil(len(self.time_series) / self.batch_size)

    # returns tuple of 2 or 3
    # x: [(batch_size, MAX_SERIES_LENGTH, 1 + num_static_labels + num_day_labels), (batch_size, MAX_SERIES_LENGTH, HORIZON_LENGTH, num_static_labels + num_day_labels)]
    # y: {"train_forecasts": (    "     , MAX_SERIES_LENGTH - HORIZON_LENGTH, HORIZON_LENGTH, num_quantiles)}
    # w: (batch_size,) [optional]
    def __getitem__(self, idx): # TODO back to __ at start
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        actual_size = self.batch_size if end < len(self.time_series) else len(self.time_series) - idx * self.batch_size

        tiled_static_features = np.tile(self.static_features[start:end], (1, MAX_SERIES_LENGTH + HORIZON_LENGTH, 1))
        tiled_day_labels = np.tile(self.day_labels, (actual_size, 1, 1))

        time_series_batch = self.time_series[start:end]
        x_batch = np.reshape(time_series_batch, time_series_batch.shape + (1,))
        x_batch = np.concatenate([x_batch, tiled_day_labels[:, :MAX_SERIES_LENGTH], tiled_static_features[:, :MAX_SERIES_LENGTH]], axis=-1)

        features = np.concatenate([tiled_day_labels[:, :(MAX_SERIES_LENGTH + HORIZON_LENGTH)], tiled_static_features], axis=-1)
        pred_features = []
        for i in range(features.shape[0]):
            series = features[i]
            preds = []
            pred_features.append(preds)
            for j in range(1, MAX_SERIES_LENGTH+1):
                preds_sublist = []
                preds.append(preds_sublist)
                for k in range(HORIZON_LENGTH):
                    preds_sublist.append(series[j+k])
        pred_features = np.array(pred_features)

        # before1 = time()
        y_batch = []
        for i in range(time_series_batch.shape[0]):
            series = time_series_batch[i]
            augmented_series = []
            y_batch.append(augmented_series)
            for j in range(1, LOSS_SIZE + 1):
                augmented_series.append([len(QUANTILES) * [series[j+k]] for k in range(HORIZON_LENGTH)])
        y_batch = np.array(y_batch)
        # after1 = time()
        # print("time to create y_batch: {}".format(after1 - before1))

        full_batch = ([x_batch, pred_features], {'train_forecasts': y_batch})
        if self.weights is not None:
            return full_batch + (self.weights[start:end],)
        return full_batch
