import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

from constants import SELL_PRICES_DATA_PATH, CALENDAR_DATA_PATH, SELL_PRICE_INDICES_FILE

sell_prices_df = pd.read_csv(SELL_PRICES_DATA_PATH)
item_sell_price_indices = {}
start = True
prev_id = ('', '')
for k in tqdm(range(len(sell_prices_df))):
    row = sell_prices_df.iloc[k]
    curr_id = (row.item_id, row.store_id)
    if curr_id == prev_id:
        continue
    if not start:
        item_sell_price_indices[prev_id][1] = k-1
    item_sell_price_indices[curr_id] = [k, len(sell_prices_df) - 1]
    prev_id = curr_id

with open(SELL_PRICE_INDICES_FILE, 'wb') as file:
    pickle.dump(item_sell_price_indices, file, protocol=pickle.HIGHEST_PROTOCOL)

# with open(SELL_PRICE_INDICES_FILE, 'rb') as file:
#     data = pickle.load(file)

