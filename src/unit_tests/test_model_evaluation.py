import numpy as np
import pandas as pd
import datetime
import pickle

import altair as alt

import sys
sys.path.insert(1, '..')
from preprocessing import *
from model_evaluation import *

# Load the test data
flows = read_data("../../data/station_flowrate.csv")
col = '05DF001_flow_m3s-1'
gauge = choose_gauge(flows, col)
narr_dir = "../../data/NARR_csvs/05DF001.csv"
gauge_new = add_narr(gauge, narr_dir)
df_train, df_val, df_test = choose_month_and_split(gauge_new, val_size=214, test_size=214, start_month=4, end_month=10)

# Test get_quantile_loss
assert get_quantile_loss(0.5, 0.5, 0) == 0.25, "The result of the quantile loss is wrong!"
assert get_quantile_loss(0.5, 0.5, 1) == 0.25, "The result of the quantile loss is wrong!"
assert get_quantile_loss(0.5, 0.5, 0.5) == 0, "The result of the quantile loss is wrong!"

# Test avg_quantile_loss
with open('../../data/pickle/res_null_model.pickle', 'rb') as handle:
    res_null_model = pickle.load(handle)

avg_loss = avg_quantile_loss(res_null_model, df_val, 'flow_record')
assert isinstance(avg_loss, float), "The result should be numeric!"

# Test plot_quantiles
plot1 = plot_quantiles(res_null_model, df_val)
assert isinstance(plot1, alt.vegalite.v4.api.LayerChart), "The type of result should be Altair object!"

# Test avg_loss_multiple_step
with open('../../data/pickle/res_six_ahead.pickle', 'rb') as handle:
    res_six_ahead = pickle.load(handle)
multi_loss = avg_loss_multiple_step(res_six_ahead, df_val.iloc[30:,], label='flow_record')
assert isinstance(multi_loss, float), "The result should be numeric!"

# Test plot_multiple_step_prediction
train = df_train.append(df_val.iloc[:30,:])
plot2 = plot_multiple_step_prediction(train, df_val.iloc[30:,], res_six_ahead, hourly=False)
assert isinstance(plot2, alt.vegalite.v4.api.LayerChart), "The type of result should be Altair object!"