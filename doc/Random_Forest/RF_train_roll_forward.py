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

# Null model
res_null_model = {}
for i in range(len(df_val)):
    res_null_model[df_val.index[i]] = np.percentile(df_train[df_train['day'] == df_val.day.values[i]].flow_record.values, np.arange(.01, .99, .02)*100)

# Save the results to the pickle files for future use.
with open('../../data/pickle/res_null_model.pickle', 'wb') as handle:
    pickle.dump(res_null_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Fit the random forest model with chosen predictors and predict on the validation set without NARR
# Please note that, the number of lags used as predictors needs to be as the same as the parameter `n_lag`.
res_origin_flow, feat_imp = RF_model_origin_flow(df_train.iloc[-428:,], df_val,
                                                 predictor_cols=['flow_record1', 'flow_record2', 'flow_record3'],
                                                 label='flow_record', n_lag=3)

# Save the predicted quantiles and average feature importance to the pickle files for future use.
with open('../../data/pickle/res_origin_flow_daily.pickle', 'wb') as handle:
    pickle.dump(res_origin_flow, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('../../data/pickle/feat_imp_daily.pickle', 'wb') as handle:
    pickle.dump(feat_imp, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Fit the random forest model with chosen predictors and predict on the validation set with NARR
# Please note that, the max number of lags used as predictors needs to be as the same as the parameter `n_lag`.
res_narr_origin_flow, feat_narr_imp = RF_model_origin_flow(df_train.iloc[-428:,], df_val,
                                                      predictor_cols=['flow_record1', 'apcp1', 'apcp2', 'apcp3', 'apcp4', 'soilm1'],
                                                      label='flow_record', n_lag=4)

# Save the predicted quantiles and average feature importance to the pickle files for future use.
with open('../../data/pickle/narr_res_origin_flow_daily.pickle', 'wb') as handle:
    pickle.dump(res_narr_origin_flow, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('../../data/pickle/narr_feat_imp_daily.pickle', 'wb') as handle:
    pickle.dump(feat_narr_imp, handle, protocol=pickle.HIGHEST_PROTOCOL)
