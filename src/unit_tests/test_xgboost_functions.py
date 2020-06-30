import numpy as np
import pandas as pd
import datetime
import pickle

import sys
sys.path.insert(1, '..')
from preprocessing import *
from model_evaluation import *
from xgboost_functions import *

# Load the test data
flows = read_data("../../data/station_flowrate.csv")
col = '05DF001_flow_m3s-1'
gauge = choose_gauge(flows, col)
narr_dir = "../../data/NARR_csvs/05DF001.csv"
gauge_new = add_narr(gauge, narr_dir)
df_train, df_val, df_test = choose_month_and_split(gauge_new, val_size=214, test_size=214, start_month=4, end_month=10)

# Test gb_quantile
# Generate the synthetic data
N = 100
x = np.arange(0, 1, 1/N)
y = np.random.normal(2*x + 1, 1)

# Reshape the predictor to train the sklearn GradientBoostingRegressor model
x_new = x.reshape(N,1)

quantiles = np.arange(.01, .99, .02)
gb_quanitle1 = gb_quantile(x_new, y, [[1]], 0.5)
assert isinstance(gb_quanitle1[0], float), "The result should be numeric!"

# Test gb_model
result1, imp1 = gb_model(df_train, df_val.iloc[:1,], predictor_cols=['flow_record'], label='flow_record', n_lag=1)
assert df_val.iloc[:1,].index[0] == list(result1.keys())[0], "The prediction should be on the first day in the validation set!"
assert len(result1[df_val.iloc[:1,].index[0]]) == 49, "The size of the prediction is wrong!"

