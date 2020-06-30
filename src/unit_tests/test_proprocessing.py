import numpy as np
import pandas as pd
import datetime

from sklearn.model_selection import train_test_split

import sys
sys.path.insert(1, '..')
from preprocessing import *

# Test read_data function
flows = read_data("../../data/station_flowrate.csv")
assert flows.index.name == 'date', "The index name of the dataframe should be 'date'!"
assert isinstance(flows, pd.core.frame.DataFrame), "The data tpye should be pandas DataFrame!"

# Test choose_gauge function
col = '05DF001_flow_m3s-1'
gauge = choose_gauge(flows, col)
assert all(gauge.columns == ['flow_record', 'log_flow', 'year', 'day']), "The columns of the dataframe is wrong!"
assert all(np.log(gauge['flow_record']) == gauge['log_flow']), "The values in log_flow column are wrong!"

# Test add_narr function
narr_dir = "../../data/NARR_csvs/05DF001.csv"
gauge_new = add_narr(gauge, narr_dir)
assert gauge_new.index.name == 'date', "The index name of the dataframe should be 'date'!"
assert set(['apcp', 'crain', 'snom', 'soilm', 'ssrun']).issubset(gauge_new.columns), "The dataframe should contain all the NARR columns!"

# Test choose_month_and_split function
df_train, df_val, df_test = choose_month_and_split(gauge_new, val_size=214, test_size=214, start_month=4, end_month=10)
assert len(df_val) == 214, "The size of the validation set should be 214!"
assert len(df_test) == 214, "The size of the test set should be 214!"
assert all(df_val.index.month > 3), "The result should not contain the data in the first three months in a year!"
assert all(df_val.index.month < 11), "The result should not contain the data in the last two months in a year!"

# Test lag_df function
lagged_train = lag_df(df_train, lag=2, cols=['flow_record', 'apcp'])
assert set(['apcp1', 'apcp2', 'flow_record1', 'flow_record2']).issubset(lagged_train.columns), "The dataframe should contain the lagged columns!"
assert all(lagged_train['apcp'].shift(1).dropna() == lagged_train['apcp1'].dropna()), "The column is not correctly lagged!"

# Test the lag function
lagged_train_new, lagged_cols = lag(df=df_train, lag=2, cols=['flow_record', 'apcp'])
assert set(['apcp1', 'apcp2', 'flow_record1', 'flow_record2']).issubset(lagged_train_new.columns), "The dataframe should contain the lagged columns!"
assert all(lagged_train_new['apcp'].shift(1).dropna() == lagged_train_new['apcp1'].dropna()), "The column is not correctly lagged!"
assert lagged_cols == ['flow_record1', 'flow_record2', 'apcp1', 'apcp2'], "The names of lagged columns are not correctly returned!"

# Test read_hourly_data
data_dir = "../../data/05DC001_Q_1996to2019_20200602.csv"
flow = read_hourly_data(data_dir)
assert flow.index.name == 'time', "The index name of the dataframe should be 'time'!"
assert isinstance(flow, pd.core.frame.DataFrame), "The data tpye should be pandas DataFrame"
assert all(flow.columns == ['date', 'hour', 'Value', 'month', 'minute', 'year', 'day']), "The columns of the dataframe is wrong!"

# Test add_houly_narr
narr_dir = f'../../data/NARR_csvs/05DC001_q1_3hour.csv'
flow_new = add_hourly_narr(flow, narr_dir)
assert flow_new.index.name == 'time', "The index name of the dataframe should be 'date'!"
assert set(['apcp', 'snom', 'soilm', 'ssrun']).issubset(flow_new.columns), "The dataframe should contain all the NARR columns!"