import tensorflow as tf

DTYPE = tf.float32
HORIZON_LENGTH = 28
QUANTILES = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]
INCLUDE_VALIDATION_METRIC = True
MAX_SERIES_LENGTH = 1941
if INCLUDE_VALIDATION_METRIC:
    MAX_SERIES_LENGTH -= HORIZON_LENGTH
LOSS_SIZE = MAX_SERIES_LENGTH - HORIZON_LENGTH

ALL_FEATURES_DATA_PATH = './data/aggregate_features.csv'
AGGREGATE_EVALUATION_SALES_PATH = './data/aggregate_sales_train_evaluation.csv'
# FULL_SALES_DATA_PATH = './data/sales_train_evaluation.csv'
WEIGHTS_PATH = './data/evaluation_weights.csv'
CALENDAR_DATA_PATH = './data/calendar.csv' # calendar goes to day 1969
# SELL_PRICES_DATA_PATH = './data/sell_prices.csv'
FIRST_DAY_FILE = './data/first_days.p'
FIRST_WEEK = 11101