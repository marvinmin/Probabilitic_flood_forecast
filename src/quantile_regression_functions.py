import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import datetime

from sklearn.model_selection import train_test_split

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from preprocessing import*

def QR_model_origin_flow(df_train, df_val, predictor_cols, label, n_lag=3, quantiles=np.arange(.01, .99, .02)):
    """
    Train the Quantile Regression model on the train set and make the roll-forward 
    one-step ahead quantile prediction on the validation set using the origin 
    flow rates as the response.

    Parameters
    ----------
    df_train: pandas.DataFrame
        The training dataset
    df_val: pandas.DataFrame
        The validation dataset
    predictor_cols: list of strings
        The list of names of predictors to be used to train the model
    label: str
        The name of label in the training set
    n_lag: int (default:3)
        The number of lags to fit the model
    quantiles: array-like of float (default:np.arange(.01, .99, .02))
        The sequence of percentiles to compute
    
    Return
    ------
    result: dict
        The dictionary with dates as keys and precited quantiles as values
    """
    # Copy the training set to a temp dataframe
    df_temp = df_train.copy()
    # The dictionary to store the result, the key of the dictionary would be the dates (The index of the validation set)
    result = {}
    for k in range(len(df_val)):
        # Get the lags of all the columns in the training set and drop NAs
        df_temp = lag_df(df_temp, lag = n_lag, cols = df_temp.columns).dropna()
        
        # Train the quantile regression model 
        model_list = []
        seperator = '+'
        quantreg = smf.quantreg(f'{label}~{seperator.join(predictor_cols)}', data = df_temp)
        for q in quantiles:
            model_list.append(quantreg.fit(q=q, max_iter=2000))

        # Initialize the dataframe for predictors to do the prediction,
        # the index of the dataframe should be the same as the row from 
        # the validation set that we want to predict
        df_pred = pd.DataFrame(df_val.loc[df_val.index[k],:]).T
        
        # Get all the needed lags from the last few rows of the training set
        for col in df_temp.columns:
            for i in range(n_lag):
                df_pred[col+f'{i + 1}'] = df_temp[col].values[-1-i]
        # Select the chosen predictors
        df_pred = df_pred.loc[:,predictor_cols]
        
        # Predict and add the predicted quantiles into the result dictionary
        pred_quantiles = np.concatenate([model_list[m].predict(df_pred) for m in range(len(quantiles))])        
        result[df_val.index[k]] = pred_quantiles

        # Update the training set by appending the row from the validation set that has been predited
        df_train = df_train.append(df_val.iloc[k,:])
        df_temp = df_train.copy()

    return result

def QR_multi_step_predict(df_train, label, predictors, n_lag=3, N=100, time_steps=6, quantiles=np.arange(.01, .99, .02)):
    """
    fit and predict multiple steps ahead quantile using quantile regression with NARR data 
    
    parameters: 
    -----------
    df_train: pandas.DataFrame
        The training dataset
    label: str
        The name of label in the training set
    predictors: list of lists of strings
        The list of lists of names of predictors to be used to train each model
    n_lag: int
        The maximum number of lags of features needed in the first step ahead prediction
    N: int (default:100)
        The number of samples to form a distribution
    time_steps: int (default:6)
        The number of time steps to predict
    quantiles: array-like of float (default:np.arange(.01, .99, .02))
        The sequence of percentiles to compute
        
    Return:
    -------
    pandas.DataFrame
        the results of simulated values
    """
    # lag the flow rate
    df_temp = lag_df(df_train, lag = time_steps+n_lag-1, cols = df_train.columns).dropna()

    # fit the models
    # a list to store models
    model_list = []
    
    if all(predictor == predictors[0] for predictor in predictors):
        # If only the lags of flow rate are used as predictors, only need to fit one model
        quantile_list = []
        seperator = '+'
        predictor_cols = predictors[0]
        quantreg = smf.quantreg(f'{label}~{seperator.join(predictor_cols)}', data = df_temp)
        for q in quantiles:
            quantile_list.append(quantreg.fit(q=q, max_iter=2000))
        model_list = [quantile_list] *time_steps
    else:
        # Fit multiple models
        for i in range(time_steps):
            quantile_list = []
            seperator = '+'
            predictor_cols = predictors[i]
            quantreg = smf.quantreg(f'{label}~{seperator.join(predictor_cols)}', data = df_temp)
            for q in quantiles:
                 quantile_list.append(quantreg.fit(q=q, max_iter=2000))
            model_list.append(quantile_list)
    
    # initialize the result array to store the result after simulation
    result = np.zeros((time_steps, N))
    for n in range(N):
        temp_row = [df_train.loc[:, label][-1]]

        for i in range(time_steps):            
            predictor_cols = predictors[i]
            predictions = lag_df(df_train.append(pd.DataFrame({label: temp_row}), ignore_index=True), 
                                    lag = time_steps+n_lag-1, 
                                    cols = df_train.columns)[-1:].loc[:,predictor_cols]
            pred_flow = np.concatenate([model_list[i][m].predict(predictions) for m in range(len(quantiles))])

            result[i, n] = np.random.choice(pred_flow)
            temp_row[-1] = result[i, n]
            temp_row.append(result[i, n])
            
    result_df = pd.DataFrame(result.T)
    return result_df
       