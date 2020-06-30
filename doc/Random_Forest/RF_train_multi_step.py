import numpy as np
import pandas as pd

import pickle

# Load the functions
import sys
sys.path.insert(1, '../../src')
from preprocessing import*
from random_forest_functions import*
from model_evaluation import*

# Read the flow rates data and select the gauge of interest
flows = read_data("../../data/station_flowrate.csv")
col = '05DF001_flow_m3s-1'
gauge = choose_gauge(flows, col)

# Attach NARR data to the gauge
narr_dir = "../../data/NARR_csvs/05DF001.csv"
gauge_new = add_narr(gauge, narr_dir)

# Choose the summer month and split the data
df_train, df_val, df_test = choose_month_and_split(gauge_new, val_size=214, test_size=214, start_month=4, end_month=10)
# Append the training set so that we predict on a day in May
train = df_train.append(df_val.iloc[:30,:])

# Predictors for the model without NARR data
predictors = [
    ['flow_record1', 'flow_record2', 'flow_record3'],
    ['flow_record1', 'flow_record2', 'flow_record3'],
    ['flow_record1', 'flow_record2', 'flow_record3'],
    ['flow_record1', 'flow_record2', 'flow_record3'],
    ['flow_record1', 'flow_record2', 'flow_record3'],
    ['flow_record1', 'flow_record2', 'flow_record3']
]

# Fit the random forest model with above predictors and predict 6 steps ahead without NARR
res_six_ahead = RF_multi_step_predict(train, label='flow_record', predictors=predictors, n_lag=3, N=100, time_steps=6)

# Save the results to the pickle files for future use.
with open('../../data/pickle/res_six_ahead.pickle', 'wb') as handle:
    pickle.dump(res_six_ahead, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Predictors for the model with NARR data
predictors_narr = [
    ['flow_record1', 'soilm1'],
    ['flow_record1', 'soilm2'],
    ['flow_record1', 'soilm3'],
    ['flow_record1', 'soilm4'],
    ['flow_record1', 'soilm5'],
    ['flow_record1', 'soilm6']
]

# Fit the random forest model with above predictors and predict 6 steps ahead with NARR
res_six_ahead_narr = RF_multi_step_predict(train, label='flow_record', predictors=predictors_narr, n_lag=1, N=100, time_steps=6)

# Save the results to the pickle files for future use.
with open('../../data/pickle/res_six_ahead_narr.pickle', 'wb') as handle:
    pickle.dump(res_six_ahead_narr, handle, protocol=pickle.HIGHEST_PROTOCOL)