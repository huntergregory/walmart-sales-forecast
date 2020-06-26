import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

from constants import SALES_DATA_PATH, FIRST_DAY_FILE

sales_df = pd.read_csv(SALES_DATA_PATH)
sales_df = sales_df.iloc[:, 6:]

first_days = []
for k in tqdm(range(len(sales_df))):
    row = np.array(sales_df.iloc[k])
    for j, sale in enumerate(row):
        if sale > 0:
            first_days.append(j)
            break

print(len(first_days) == len(sales_df))

with open(FIRST_DAY_FILE, 'wb') as file:
    pickle.dump(first_days, file, protocol=pickle.HIGHEST_PROTOCOL)

# with open(SELL_PRICE_INDICES_FILE, 'rb') as file:
#     data = pickle.load(file)
