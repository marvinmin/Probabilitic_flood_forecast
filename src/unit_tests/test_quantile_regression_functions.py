import numpy as np
import pandas as pd
import datetime
import pickle

import sys
sys.path.insert(1, '..')
from preprocessing import *
from model_evaluation import *
from quantile_regression_functions import *

# Load the test data
flows = read_data("../../data/station_flowrate.csv")
col = '05DF001_flow_m3s-1'
gauge = choose_gauge(flows, col)
narr_dir = "../../data/NARR_csvs/05DF001.csv"
gauge_new = add_narr(gauge, narr_dir)
df_train, df_val, df_test = choose_month_and_split(gauge_new, val_size=214, test_size=214, start_month=4, end_month=10)

# Test QR_model_origin_flow
result1 = QR_model_origin_flow(df_train, df_val.iloc[:1,], predictor_cols=['flow_record1'], label='flow_record', n_lag=1)
assert df_val.iloc[:1,].index[0] == list(result1.keys())[0], "The prediction should be on the first day in the validation set!"
assert len(result1[df_val.iloc[:1,].index[0]]) == 49, "The size of the prediction is wrong!"
for i in range(48):
    assert result1[df_val.iloc[:1,].index[0]][i] <= result1[df_val.iloc[:1,].index[0]][i+1], "The lower quantiles should be less than the higher quantiles!"

# Test QR_multi_step_predict
predictors=[
    ['flow_record1', 'flow_record2'],
    ['flow_record1', 'flow_record2'],
    ['flow_record1', 'flow_record2']
]
multi_res = QR_multi_step_predict(df_train, label='flow_record', predictors=predictors, n_lag=2, N=2, time_steps=3)
assert len(multi_res) == 2, "The length of the result should be the same as the number of iterations!"
assert len(multi_res.columns) == 3, "The number of columns of the result should be the same as the number of time steps!"
