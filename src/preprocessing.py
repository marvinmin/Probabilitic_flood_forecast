import numpy as np
import pandas as pd
import datetime

from sklearn.model_selection import train_test_split

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def read_data(file_dir):
    """
    Read the data and set the index to be the date with the pandas datetime type
    
    Parameter
    ---------
    file_dir: str
        The directory of the data to be read in (with .csv in the end)
    
    Return
    ------
    pandas.DataFrame
        The data whose index is the date with pandas datetime type and columns are 
        the flow rate records for each gauge.
    """
    flows = pd.read_csv(file_dir)
    # Rename the date column
    flows = flows.rename(columns={'Unnamed: 0': "date"})
    # Transfer to pandas datetime format
    flows['date'] = pd.to_datetime(flows['date'])
    # Set the date columns as index
    flows = flows.set_index('date')
    return flows

def choose_gauge(flows, col):
    """
    Choose the gauge to explore, drop NAs and filter positive flow.
    Also generate columns that present the log of the flow rate,
    the year and the Julian day of the record.
    
    Parameters
    ----------
    flows: pandas.DataFrame
        The flow river data read by `read_data` function
    col: str
        The name of the gauge (One of the column names in flows)
        
    Return
    ------
    pandas.DataFrame
        The gauge with date as the index and flow rates, log flow rates,
        year and Julian day as the columns.
    """
    gauge = flows.loc[:,[col]].dropna()
    # Rename the flow rate columns
    gauge = gauge.rename(columns={col: "flow_record"})
    # Only consider the positive flow rate
    gauge = gauge[gauge['flow_record'] > 0]
    # Take the log of the flow rate in case of future need
    gauge['log_flow'] = np.log(gauge['flow_record'])
    # Add year and Julian day columns for future use
    gauge['year'] = gauge.index.year
    gauge['day'] = gauge.index.dayofyear
    return gauge

def add_narr(gauge, narr_dir):
    """
    Add NARR data to the existing gauge dataframe

    Parameters
    ----------
    gauge: pandas.DataFrame
        The gauge with date as the index and flow rates, log flow rates,
        year and Julian day as the columns.
    narr_dir: str
        The directory of the NARR data to be read in (with .csv in the end)

    Return
    ------
    pandas.DataFrame
        The gauge with date as the index and flow rates, log flow rates,
        year, Julian day and NARR features as the columns.
    """
    narr_df = pd.read_csv(narr_dir)
    # Make the date column as the same format as the gauge dataframe
    narr_df = narr_df.rename(columns = {"time":"date"})
    narr_df['date'] = pd.to_datetime(narr_df['date'])

    gauge_df = gauge.reset_index()

    # Merge the flow rate dataframe with the NARR dataframe by the date
    gauge_new = gauge_df.merge(narr_df, on = 'date')
    # Set the data as the index
    gauge_new = gauge_new.set_index('date')
    
    return gauge_new

def choose_month_and_split(gauge, val_size=214, test_size=214, start_month=4, end_month=10):
    """
    Choose the months of interest and split the data into train, validation, test set.

    Parameters
    ----------
    gauge: pandas.DataFrame
        The gauge with date as the index and flow rates, log flow rates,
        year, Julian day and NARR features as the columns.
    val_size: int (default: 214)
        The absolute number of validation samples
    test_size: int (default: 214)
        The absolute number of test samples
    start_month: int (default: 4 (April))
        The start month of the year
    end_month: int (default: 10 (October))
        The end month of the year
    
    Return
    ------
    tuple: (df_train, df_val, df_test)
        df_train: pandas.DataFrame
            The training dataframe
        df_val: pandas.DataFrame
            The validation dataframe
        df_test: pandas.DataFrame
            The test dataframe
    """
    # Choose the start month and end month of interest
    gauge = gauge[(gauge.index.month >= start_month) & (gauge.index.month <= end_month)]

    # Split the data without shuffle
    df_train_val, df_test = train_test_split(gauge, test_size=test_size, shuffle=False)
    df_train, df_val = train_test_split(df_train_val, test_size=val_size, shuffle=False)

    return df_train, df_val, df_test


def read_hourly_data(file_dir):
    """
    Read the houlry data and set the index to be the time with the pandas datetime type
    
    Parameter
    ---------
    file_dir: str
        The directory of the data to be read in (with .csv in the end)
    
    Return
    ------
    pandas.DataFrame
        The data whose index is the time with pandas datetime type and columns are 
        the flow rate records for each gauge.
    """
    # Read the data and skip the meta data part, only use the first three columns
    flow = pd.read_csv(file_dir, skiprows=14, usecols = [0, 1, 2])
    # Drop NAs
    flow = flow.dropna()
    # Set the index as time and use the pandas datetime format
    flow['time'] = pd.to_datetime(flow['ISO 8601 UTC'])
    flow = flow.set_index('time')
    # Choose only the flow rate column
    flow = flow[['Value']]

    # Get the data and time related columns
    flow['date'] = flow.index.date
    flow['hour'] = flow.index.hour
    flow['month'] = flow.index.month
    flow['minute'] = flow.index.minute
    flow['year'] = flow.index.year
    flow['day'] = flow.index.dayofyear
    # Because there are some timestamp with minutes as 56,57,58,59, we treat these timestamp as next hour
    flow.loc[flow.minute > 55, 'hour'] += 1

    # Use the max flow in an hour as the hourly data
    grouped_flow = flow.groupby(by=['date', 'hour']).max()

    # Set the data and time as the index
    grouped_flow = grouped_flow.reset_index()
    grouped_flow['date'] = pd.to_datetime(grouped_flow['date'], utc=True)
    grouped_flow['time'] = grouped_flow['date'] + (grouped_flow['hour'].astype(str) + 'H').apply(pd.Timedelta)
    grouped_flow = grouped_flow.set_index('time')
    return grouped_flow

def add_hourly_narr(grouped_flow, narr_dir):
    """
    Add NARR data to the existing gauge dataframe

    Parameters
    ----------
    grouped_flow: pandas.DataFrame
        The gauge with time as the index and flow rates, date, hour, month, minute,
        year and Julian day as the columns.
    narr_dir: str
        The directory of the NARR data to be read in (with .csv in the end)

    Return
    ------
    pandas.DataFrame
        The gauge with time as the index and flow rates, hour, year, Julian day 
        and NARR features as the columns.
    """
    narr = pd.read_csv(narr_dir)
    # Set the time column as the same format
    narr['time'] = pd.to_datetime(narr['time'], utc=True)
    narr = narr.set_index('time')
    # Change the 3-hourly NARR data to hourly data by interploting with the nearest value
    narr = narr.reindex(pd.date_range(start=narr.index.min(),end=narr.index.max(),freq='1H')
                       ).interpolate(method='nearest').reset_index().rename(columns={"index": "time"})
    grouped_flow = grouped_flow.reset_index()
    # Merge the two columns by time and set the time as index
    flow_new = grouped_flow.merge(narr, on = 'time').set_index('time')
    flow_new = flow_new.loc[:,['Value', 'year', 'day', 'hour', 'apcp', 'ssrun', 'snom', 'soilm']]
    return flow_new

def lag_df(df, lag, cols):
    """
    Get the lags of columns of a dataframe.

    Parameters
    ----------
    df: pandas.DataFrame
        The dataframe of interest
    lag: int
        The number of lags to get
    cols: list of strings
        The list of column names , whose lags are needed
    
    Return
    ------
    pandas.DataFrame
        The dataframe with lags of the columns added. The name of the lagged columns is combination 
        of original column and the number of lag, e.g., 'apcp2' means 2 lags of apcp.
    """
    return df.assign(**{f"{col}{n}": df[col].shift(n) for n in range(1, lag + 1) for col in cols})

def lag(lag, df, cols):
    """
    shift the given column in the dataframe with the given lag number,
    return the lagged data frame and its list of the column names
    
    Parameters
    ----------
    lag: int
        The number of lags to get
    df: pandas.DataFrame
        The data frame of interest 
    cols: list of strings
        The list of column names, whose lags are needed
    
    Return
    ------
    tuple:(df, list_of_cols)
        df: pandas.DataFrame
                The dataframe with lags of the columns added. The name of the lagged columns is combination 
                of original column and the number of lag, e.g., 'apcp2' means 2 lags of apcp.
        list_of_cols: list
                The list of the column names in the lagged data frame
    """
    # lag the data frame according to the provided lag number
    lagged_df = df.assign(**{f"{col}{n}": df[col].shift(n) for n in range(1, lag + 1) for col in cols})
    
    # get the columns
    col_list = []
    for col in cols:
        for n in range(1, lag + 1):
            col_list.append(f"{col}{n}")
                
    return lagged_df, col_list