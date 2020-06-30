import numpy as np
import pandas as pd
import datetime

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

# Read training and validation sets from the saved pickle files
with open('../../data/pickle/qr_hourly_train.pickle', 'rb') as handle:
    df_train = pickle.load(handle)
with open('../../data/pickle/qr_hourly_val.pickle', 'rb') as handle:
    df_val = pickle.load(handle)

# Train three models using different predictors
# Only use flow data
predictors = [
    ['Value1', 'Value2', 'Value3'],
    ['Value1', 'Value2', 'Value3'],
    ['Value1', 'Value2', 'Value3'],
    ['Value1', 'Value2', 'Value3'],
    ['Value1', 'Value2', 'Value3'],
    ['Value1', 'Value2', 'Value3']
]

without_NARR = QR_multi_step_predict(df_train, label='Value', predictors = predictors, n_lag=3, N=100, time_steps=6, quantiles=np.arange(.01, .99, .02))
# Save the results
with open('../../data/pickle/qr_without_NARR.pickle', 'wb') as handle:
    pickle.dump(without_NARR, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Add precipitation and surface runoff 
apcp_ssrun_predictors = [
    ['Value1', 'Value2', 'Value3', 'q1_apcp3', 'q1_ssrun3'],
    ['Value1', 'Value2', 'Value3', 'q1_apcp4', 'q1_ssrun4'],
    ['Value1', 'Value2', 'Value3', 'q1_apcp5', 'q1_ssrun5'],
    ['Value1', 'Value2', 'Value3', 'q1_apcp6', 'q1_ssrun6'],
    ['Value1', 'Value2', 'Value3', 'q1_apcp7', 'q1_ssrun7'],
    ['Value1', 'Value2', 'Value3', 'q1_apcp8', 'q1_ssrun8']
]
NARR = QR_multi_step_predict(df_train, label='Value', predictors = apcp_ssrun_predictors, n_lag=3, N=100, time_steps=6, quantiles=np.arange(.01, .99, .02))
# Save the results
with open('../../data/pickle/qr_with_NARR.pickle', 'wb') as handle:
    pickle.dump(NARR, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Add soil moisture
apcp_ssrun_soilm_predictors = [
    ['Value1', 'Value2', 'Value3', 'q1_apcp3', 'q1_ssrun3', 'q1_soilm1'],
    ['Value1', 'Value2', 'Value3', 'q1_apcp4', 'q1_ssrun4', 'q1_soilm2'],
    ['Value1', 'Value2', 'Value3', 'q1_apcp5', 'q1_ssrun5', 'q1_soilm3'],
    ['Value1', 'Value2', 'Value3', 'q1_apcp6', 'q1_ssrun6', 'q1_soilm4'],
    ['Value1', 'Value2', 'Value3', 'q1_apcp7', 'q1_ssrun7', 'q1_soilm5'],
    ['Value1', 'Value2', 'Value3', 'q1_apcp8', 'q1_ssrun8', 'q1_soilm6']
]

NARR_solim = QR_multi_step_predict(df_train, label='Value', predictors = apcp_ssrun_soilm_predictors, n_lag=3, N=100, time_steps=6, quantiles=np.arange(.01, .99, .02))
# Save the results
with open('../../data/pickle/qr_with_NARR_soilm.pickle', 'wb') as handle:
    pickle.dump(NARR_solim, handle, protocol=pickle.HIGHEST_PROTOCOL)