import tensorflow as tf

DTYPE = tf.float32
HORIZON_LENGTH = 28
QUANTILES = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]
MAX_SERIES_LENGTH = 1913 # len(sales_df.columns) - 6
LOSS_SIZE = MAX_SERIES_LENGTH - HORIZON_LENGTH

SALES_DATA_PATH = './data/sales_train_validation.csv'
CALENDAR_DATA_PATH = './data/calendar.csv' # calendar goes to day 1969
SELL_PRICES_DATA_PATH = './data/sell_prices.csv'
SELL_PRICE_INDICES_FILE = './data/sell_price_indices.p'
FIRST_DAY_FILE = './data/first_days.p'
FIRST_WEEK = 11101