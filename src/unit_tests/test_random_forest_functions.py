import numpy as np
import pandas as pd
import datetime
import pickle

import sys
sys.path.insert(1, '..')
from preprocessing import *
from model_evaluation import *
from random_forest_functions import *

# Load the test data
flows = read_data("../../data/station_flowrate.csv")
col = '05DF001_flow_m3s-1'
gauge = choose_gauge(flows, col)
narr_dir = "../../data/NARR_csvs/05DF001.csv"
gauge_new = add_narr(gauge, narr_dir)
df_train, df_val, df_test = choose_month_and_split(gauge_new, val_size=214, test_size=214, start_month=4, end_month=10)

# Test rf_quantile
# Generate the synthetic data
N = 100
x = np.arange(0, 1, 1/N)
y = np.random.normal(2*x + 1, 1)
# Reshape the predictor to train the sklearn RandomForestRegressor model
x_new = x.reshape(N,1)
rf = RandomForestRegressor(n_estimators=1000, n_jobs=-1) 
rf.fit(x_new, y)
quantiles = np.arange(.01, .99, .02)
# Make the prediction when x is 1
quantile1 = rf_quantile(rf, [[1]], quantiles)
assert len(quantile1) == 49, "The length of the result is wrong!"

# Test RF_model_origin_flow
result1, imp1 = RF_model_origin_flow(df_train, df_val.iloc[:1,], predictor_cols=['flow_record1'], label='flow_record', n_lag=1)
assert df_val.iloc[:1,].index[0] == list(result1.keys())[0], "The prediction should be on the first day in the validation set!"
assert len(result1[df_val.iloc[:1,].index[0]]) == 49, "The size of the prediction is wrong!"
for i in range(48):
    assert result1[df_val.iloc[:1,].index[0]][i] <= result1[df_val.iloc[:1,].index[0]][i+1], "The lower quantiles should be less than the higher quantiles!"
assert imp1.sum() == 1, "The sum of the feature importance should be 1."

# Test RF_multi_step_predict
predictors=[
    ['flow_record1', 'flow_record2'],
    ['flow_record1', 'flow_record2'],
    ['flow_record1', 'flow_record2']
]
multi_res = RF_multi_step_predict(df_train, label='flow_record', predictors=predictors, n_lag=2, N=2, time_steps=3)
assert len(multi_res) == 2, "The length of the result should be the same as the number of iterations!"
assert len(multi_res.columns) == 3, "The number of columns of the result should be the same as the number of time steps!"