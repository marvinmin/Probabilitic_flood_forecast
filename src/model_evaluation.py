import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import datetime
import xarray as xr
import zarr

from sklearn.model_selection import train_test_split

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import altair as alt

def get_quantile_loss(q, y, f):
    """
    Calculate the quantile loss of a point at different quantiles
    
    Parameters
    ----------
    q: float or array-like of float
        The precentiles to be evaluated, e,g, 0.5
    y: float
        The true value of the point
    f: float or array-like of float
        The fitted or predicted quantiles of the point 
    
    Return
    ------
    float or array-like of float
        The quantile loss of the point at different quantiles
    """
    e = y - f
    return np.maximum(q * e, (q - 1) * e)

def avg_quantile_loss(res, df, label):
    """
    Get average quantile loss on a dataset for the one-step ahead prediction

    Parameters
    ----------
    res: dict
        The dictionary with dates as keys and precited quantiles as values
    df: pandas.DataFrame
        The dataframe with the same dates as index
    label: str
        The name of true flow in the dataset df

    Return
    ------
    float
        The average quantile loss for the prediction
    """
    # Define the quantiles to be evaluated
    quantiles=np.arange(.01, .99, 1/(1+len(res[list(res.keys())[0]])))
    result = 0
    for k, v in res.items():
        # Get the true flow rate
        true_flow = df.loc[k, label]
        # Get the average quantile loss for all the quantiles
        result += get_quantile_loss(quantiles, true_flow, v).mean()
    # Get the average quanlile loss for the whole dataset
    result /= len(res)
    return round(result, 4)

def plot_quantiles(res, df_val, label="flow_record", set_title = "Predicted Quantiles vs. True Daily Flow", width=900, height=400):
    """
    Plot the predicted quantiles as the grey lines and grey points 
    with the true flow rates as the red line.
    Parameters
    ----------
    res: dict
        The dictionary with dates as keys and precited quantiles as values
    df_val: pandas.DataFrame
        The validation dataset
    label: str
        The name of label in the validation set, default: "flow_record"
    set_title: str
        The title of the output plot, default: "Predicted Quantiles vs. True Daily Flow"
    width: int
        The width of the plot, default: 900
    height: int 
        The height of the plot, default: 400
    
    Return
    Altair object
        The plot of the predicted quantiles and true flow rates
    """
    df_res = pd.DataFrame(res) 
    # Get the true flow
    true_flow_df = df_val.loc[df_res.columns, [label]].reset_index().rename(columns = {'index': 'date'})

    # Add column of the date
    df_res_t = df_res.T.reset_index().rename(columns = {'index':'date'})

    df_res_melt = df_res_t.melt(id_vars = 'date')
    # Add the legend column
    df_res_melt['legend'] = 'quantile predictions'
    true_flow_df['legend'] = 'true flow'
    true_flow_df.rename(columns = {label: 'value'}, inplace = True)

    # Combine the two dataframe
    combined_df = df_res_melt.append(true_flow_df)

    chart_true_flow = alt.Chart(combined_df.query('legend == "true flow"')).mark_line().encode(
        alt.X('date', title = 'Date'),
        alt.Y('value', title = 'Flow (m^3/s)'),
        color = alt.Color('legend:N', scale=alt.Scale(range=['red', 'grey'], domain = ['true flow', 'quantile predictions']), legend=alt.Legend(title="Flows by color", orient = 'left'))
    )

    # Base chart of the first quantile line
    chart = alt.Chart(combined_df.query('variable == 0')).mark_line(opacity=0.2).encode(
        alt.X('date', title = 'Date'),
        alt.Y('value', title = 'Flow (m^3/s)'),
        color = alt.Color('legend:N', scale=alt.Scale(range=['red', 'grey'], domain = ['true flow', 'quantile predictions']))
    )

    # Add other quantile lines
    for i in range(1,len(df_res_t)):
        chart1 = alt.Chart(combined_df.query(f'variable == {i}')).mark_line(opacity=0.2).encode(
            alt.X('date', title = 'Date'),
            alt.Y('value', title = 'Flow (m^3/s)'),
            color = alt.Color('legend:N', scale=alt.Scale(range=['red', 'gray'], domain = ['true flow', 'quantile predictions']))
        )

        chart = (chart + chart1).properties(
        width=width,
        height=height)

    # Add points on the quantile lines for the predicted quantiles
    chart_point = alt.Chart(combined_df.query('legend != "true flow"')).mark_circle(opacity=0.2, color = 'gray').encode(
        alt.X('date', title = 'Date'),
        alt.Y('value', title = 'Flow (m^3/s)', )
    ).properties(title = set_title)
    
    return chart + chart_point + chart_true_flow

def avg_loss_multiple_step(res, df_val, label, quantiles=np.arange(.01, .99, .02)):
    """
    Take the result of the multi-setp prediction, calculate the average quantile loss 
    
    Parameters
    ----------
    res: pandas.DataFrame
        The dataframe of the result from the `RF_multi_step_predict` or `QR_multi_step_predict` function
    df_val: pandas.DataFrame
        The validation dataset
    label: str
        The name of the column represents the flow rate
    quantiles: array-like of float (default:np.arange(.01, .99, .02))
        The sequence of percentiles to compute
        
    Return
    ------
    float:
        The calculated average quantile loss
    """
    # get the true flow value
    true_flow = []
    for i in range(len(res.columns)):
        true_flow.append(df_val[label].values[i])
   
    # get the quantile results for each prediction step
    quantile_result = np.zeros(len(res.columns))
    for q in quantiles:
        quantile_result += get_quantile_loss(q, true_flow, res.quantile(q))
    quantile_result = quantile_result/len(quantiles)
        
    # get the average quantile loss
    avg_quantile_loss = quantile_result.sum()
    return round(avg_quantile_loss, 4)


def plot_multiple_step_prediction(df_train, df_val, res_df, hourly=True, width=600, height=400):
    """
    Plot the predicted quantiles as the grey lines and grey points 
    with the true flow rates as the red line for multiple-step prediction.
    Parameters
    ----------
    df_train: pandas.DataFrame
        The training dataset
    df_val: pandas.DataFrame
        The validation dataset
    res_df: pandas.DataFrame
        The dataframe for the result of multiple-step prediction
    hourly: bool
        The boolean variable for whether the data is hourly data, default: True
    width: int
        The width of the plot, default: 900
    height: int 
        The height of the plot, default: 400
    
    Return
    Altair object
        The plot of the predicted quantiles and true flow rates
    """
    columns = {}
    true_flow = []
    # Set label and timedelta as the hourly or daily data
    if hourly:
        label = 'Value'
        delta = 'H'
    else:
        label = 'flow_record'
        delta = 'D'
    # Get the true flow rates
    for i in range(len(res_df.columns)):
        columns[i] = str(i+1)
        true_flow.append(df_val.loc[df_train.index[-1] + pd.Timedelta(str(i+1)+delta), label])
    # Change the start time of timestep from 0 to 1 for both results and true flow rates
    res_df = res_df.rename(columns = columns)   
    true_flow_df = pd.DataFrame({"true_flow": true_flow}).reset_index()
    true_flow_df['index'] = true_flow_df['index'] + 1
    
    # Melt the data frame and set the legend column for both dataframes
    df_res_melt = res_df.T.reset_index().rename(columns = {'index':'step'}).melt(id_vars = 'step')
    df_res_melt['legend'] = 'quantile predictions'
    
    true_flow_df = true_flow_df.rename(columns={'index': 'step', 'true_flow': 'value'})
    true_flow_df['legend'] = 'true flow'
    
    # Combine the two dataframes
    combined_df = df_res_melt.append(true_flow_df)

    chart_true_flow = alt.Chart(combined_df.query('legend == "true flow"')).mark_line().encode(
        alt.X('step:N', title = 'Step', axis=alt.Axis(labelAngle=0)),
        alt.Y('value', title = 'Flow (m^3/s)', scale=alt.Scale(zero=False)),
        color = alt.Color('legend:N', scale=alt.Scale(range=['red', 'grey'], domain = ['true flow', 'quantile predictions']), legend=alt.Legend(title="Flows by color", orient = 'left'))
    )
    # Base chart of the first quantile line
    chart = alt.Chart(combined_df.query('variable == 0')).mark_line(opacity=0.2).encode(
        alt.X('step:N', title = 'Step', axis=alt.Axis(labelAngle=0)),
        alt.Y('value', title = 'Flow (m^3/s)', scale=alt.Scale(zero=False)),
        color = alt.Color('legend:N', scale=alt.Scale(range=['red', 'grey'], domain = ['true flow', 'quantile predictions']))
    )

    # Add other quantile lines
    for i in range(1,len(res_df)):
        chart1 = alt.Chart(combined_df.query(f'variable == {i}')).mark_line(opacity=0.2).encode(
        alt.X('step:N', title = 'Step', axis=alt.Axis(labelAngle=0)),
        alt.Y('value', title = 'Flow (m^3/s)', scale=alt.Scale(zero=False)),
        color = alt.Color('legend:N', scale=alt.Scale(range=['red', 'gray'], domain = ['true flow', 'quantile predictions']))
        )

        chart = (chart + chart1).properties(
        width=width,
        height=height,
        title = f'{len(res_df.columns)} Steps Ahead Prediction')
    
    # Add points on the quantile lines for the predicted quantiles
    chart_point = alt.Chart(combined_df.query('legend != "true flow"')).mark_circle(opacity=0.2, color = 'gray').encode(
        alt.X('step:N', title = 'Step', axis=alt.Axis(labelAngle=0)),
        alt.Y('value', title = 'Flow (m^3/s)', scale=alt.Scale(zero=False))
    )
    return chart + chart_true_flow + chart_point    
