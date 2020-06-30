import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import pickle

# Load the functions
import sys
sys.path.insert(1, '../../src')
from preprocessing import*
from quantile_regression_functions import*
from model_evaluation import*

# Read and combine the data
data_dir = "../../data/05DC001_Q_1996to2019_20200602.csv"
flow = read_hourly_data(data_dir)

dfs = []
for i in range(4):
    narr_dir = f'../../data/NARR_csvs/05DC001_q{i+1}_3hour.csv'
    flow_new = add_hourly_narr(flow, narr_dir)
    dfs.append(flow_new.rename(columns={'apcp':f'q{i+1}_apcp', 
                                        'ssrun':f'q{i+1}_ssrun', 
                                        'snom':f'q{i+1}_snom', 
                                        'soilm':f'q{i+1}_soilm'}))

flow_new = dfs[0].join([dfs[1], dfs[2], dfs[3]]).drop(['Value_y','year_y','day_y','hour_y'], axis =1)
flow_new = flow_new.loc[:,~flow_new.columns.duplicated()].rename(columns={'Value_x':'Value',
                                                                          'year_x':'year',
                                                                          'day_x':'day',
                                                                          'hour_x':'hour'})

# Split the data
df_train, df_val, df_test = choose_month_and_split(flow_new, val_size=350*24, test_size=205*24, start_month=4, end_month=10)

# Save the splitted data
with open('../../data/pickle/qr_hourly_train.pickle', 'wb') as handle:
    pickle.dump(df_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../../data/pickle/qr_hourly_val.pickle', 'wb') as handle:
    pickle.dump(df_val, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../../data/pickle/qr_hourly_test.pickle', 'wb') as handle:
    pickle.dump(df_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
