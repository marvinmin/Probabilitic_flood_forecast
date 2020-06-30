import numpy as np
import pandas as pd

# import functions
import sys
sys.path.insert(1, '../../src')
from preprocessing import*
from xgboost_functions import*
from model_evaluation import*

# Read the flow rates data and select the gauge of interest
flows = read_data("../../data/station_flowrate.csv")
col = '05DF001_flow_m3s-1'
gauge = choose_gauge(flows, col)

narr_dir = "../../data/narr_csvs/05DF001.csv"
gauge_new = add_narr(gauge, narr_dir)

# Choose the summer month and split the data
df_train, df_val, df_test = choose_month_and_split(gauge_new, val_size=214, test_size=214, start_month=4, end_month=10)

# null model
res_null_model = {}
for i in range(len(df_val)):
    res_null_model[df_val.index[i]] = np.percentile(df_train[df_train['day'] == df_val.day.values[i]].flow_record.values, np.arange(.01, .99, .02)*100)

# Save the results to the pickle files for future use.
with open('../../data/pickle/res_null_model_gb.pickle', 'wb') as handle:
    pickle.dump(res_null_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# model without narr feature
gb_flow_only_results, gb_flow_only_fi = gb_model(df_train.iloc[-428:,], df_val, label = 'flow_record', predictor_cols = ['flow_record'])

# save the outputs as pickle file
with open('../data/pickle/gb_flow_only_results.pickle', 'wb') as handle:
    pickle.dump(gb_flow_only_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('../data/pickle/gb_flow_only_fi.pickle', 'wb') as handle:
    pickle.dump(gb_flow_only_fi, handle, protocol=pickle.HIGHEST_PROTOCOL)

# model with narr feature added
gb_narr_results, gb_narr_fi = gb_model(df_train.iloc[-428:,], df_val, label = 'flow_record', predictor_cols = ['flow_record', 'apcp', 'soilm'])

# save the outputs as pickle file
with open('../data/pickle/gb_narr_results.pickle', 'wb') as handle:
    pickle.dump(gb_narr_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('../data/pickle/gb_narr_fi.pickle', 'wb') as handle:
    pickle.dump(gb_narr_fi, handle, protocol=pickle.HIGHEST_PROTOCOL)