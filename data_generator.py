import numpy as np
import pandas as pd
import pickle
import math
from time import time
import tensorflow
from tensorflow.keras.utils import Sequence

from constants import LOSS_SIZE, QUANTILES, HORIZON_LENGTH, MAX_SERIES_LENGTH, SELL_PRICES_DATA_PATH, FIRST_DAY_FILE, SELL_PRICE_INDICES_FILE, FIRST_WEEK
# from tqdm import tqdm

with open(FIRST_DAY_FILE, 'rb') as file:
    first_sale_days = pickle.load(file)

with open(SELL_PRICE_INDICES_FILE, 'rb') as file:
    item_price_indices = pickle.load(file) # length of 3,049

sell_price_df = pd.read_csv(SELL_PRICES_DATA_PATH)

class TimeSeriesSequence(Sequence):
    # time_series: (n, MAX_SERIES_LENGTH)
    # item_store_ids: (n) [tuple of (item_id, store_id)]
    # static_features: (n, num_static_labels). One-hot encoded
    # day_labels: (MAX_SERIES_LENGTH + x, num_day_labels). Binary
    # weights: (n,). Leave as None for no sample weighting
    def __init__(self, time_series, item_store_ids, static_features, day_labels, batch_size, weights=None):
        self.time_series = time_series
        self.item_store_ids = item_store_ids
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

        prices = np.zeros((actual_size, MAX_SERIES_LENGTH + HORIZON_LENGTH))
        for k in range(actual_size):
            item_store_ids = self.item_store_ids[start+k]
            price_start_idx, price_stop_idx = item_price_indices[item_store_ids]
            sell_prices = sell_price_df.iloc[price_start_idx:price_stop_idx]
            start_day = (sell_prices.iloc[0].wm_yr_wk - FIRST_WEEK) * 7
            for j in range(len(sell_prices) - 1):
                week_beginning = start_day + 7*j
                prices[k, week_beginning:week_beginning + 7] = sell_prices.iloc[j].sell_price
            prices[k,-2] = sell_prices.iloc[len(sell_prices)-2].sell_price
            prices[k,-1] = sell_prices.iloc[len(sell_prices)-1].sell_price # all weeks have 7 days except last (2 days)
        prices = np.reshape(prices, prices.shape + (1,))

        for k in range(actual_size):
            first_sale_day = first_sale_days[k+idx*self.batch_size]
            tiled_static_features[k, 0:first_sale_day, :] = 0
            tiled_day_labels[k, 0:first_sale_day, :] = 0
            prices[k, 0:first_sale_day] = 0

        features_concats = [
            prices[:, :(MAX_SERIES_LENGTH + HORIZON_LENGTH)],
            tiled_day_labels[:, :(MAX_SERIES_LENGTH + HORIZON_LENGTH)], 
            tiled_static_features
        ]
        features = np.concatenate(features_concats, axis=-1)

        time_series_batch = self.time_series[start:end]
        x_batch = np.reshape(time_series_batch, time_series_batch.shape + (1,))
        x_batch_concats = [x_batch, features[:, :MAX_SERIES_LENGTH]]
        x_batch = np.concatenate(x_batch_concats, axis=-1)

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
