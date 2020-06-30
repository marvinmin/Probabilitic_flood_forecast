import numpy as np
import pandas as pd
import datetime

from sklearn.model_selection import train_test_split

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.ensemble import RandomForestRegressor

from preprocessing import*

def rf_quantile(model, X, quantiles):
    """
    Calculate the quantiles of the random forest model
    
    Parameters
    ----------
    model: sklearn RandomForestRegressor model
        The fitted model
    X: pandas.DataFrame
        The input of the `predict` method
    quantiles: array-like of float
        Sequence of percentiles to compute
    
    Return
    ------
    array
        The sequence of calculated quantiles.
    """
    # An empty list to store all the outputs of all the trees
    rf_preds = []
    # Get the output for every tree in the forest
    for estimator in model.estimators_:
        rf_preds.append(estimator.predict(X))
    
    # Make the list to an 1-D array
    rf_preds = np.concatenate(rf_preds)

    # Return the precentiles
    return np.percentile(rf_preds, quantiles*100)



def RF_model_origin_flow(df_train, df_val, predictor_cols, label, n_lag=3, quantiles=np.arange(.01, .99, .02), N_ESTIMATORS=1000):
    """
    Train the Random forest model on the train set and make the roll-forward 
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
    N_ESTIMATORS: int (default:1000)
        The number of trees in the random forest
    
    Return
    ------
    tuple: (result, avg_importance)
        result: dict
            The dictionary with dates as keys and precited quantiles as values
        avg_importance: array-like of float
            The average importance of the predictors in the model
    """
    # Copy the training set to a temp dataframe
    df_temp = df_train.copy()
    # The dictionary to store the result, the key of the dictionary would be the dates (The index of the validation set)
    result = {}
    # An empty list to store the feature inportance of all the fitted forests
    importance = []
    # Fit a random forest model for every row (prediction) on the validation set
    for k in range(len(df_val)):
        # Get the lags of all the columns in the training set and drop NAs
        df_temp = lag_df(df_temp, lag = n_lag, cols = df_temp.columns).dropna()
        # Select the chosen predictors from the lagged datframe as the input
        df_input = df_temp.loc[:,predictor_cols]
        # Choose the column to be the labels
        labels = df_temp.loc[:, label]

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
        
        # Fit a random forest regression model, use the default `max_depth` and `min_samples_split`
        # to make sure each leaf in any tree of the forest has only one value
        rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=3, n_jobs=-1) 
        rf.fit(df_input, labels)
        # Append the feature importances to the list
        importance.append(rf.feature_importances_)
        # Predict and add the predicted quantiles into the result dictionary
        pred_quantiles = rf_quantile(rf, df_pred, quantiles)         
        result[df_val.index[k]] = pred_quantiles

        # Update the training set by appending the row from the validation set that has been predited
        df_train = df_train.append(df_val.iloc[k,:])
        df_temp = df_train.copy()
        
    return result, np.mean(importance, axis=0)




def RF_multi_step_predict(df_train, label, predictors, n_lag=1, N=100, time_steps=6, quantiles=np.arange(.01, .99, .02), N_ESTIMATORS=1000):
    """
    Fit and predict multiple steps ahead quantiles using RandomForestRegressor by simulation
    
    Parameters
    ----------
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
    N_ESTIMATORS: int (default:1000)
        The number of trees in the random forest
        
    Return
    ------
    pandas.Dataframe
        The simulated flow rates for `time_steps` ahead
    
    Example
    -------
    >>> predictors = [
            ['flow_record1','flow_record2', 'soilm1', 'apcp4'],
            ['flow_record1','flow_record2', 'soilm2', 'apcp5']
        ]
    >>> RF_daily_predict(train, label='flow_record', predictors=predictors, n_lag=4, N=100, time_steps=2)
    """
    # Get all the lags
    df_temp = lag_df(df_train, lag = time_steps+n_lag-1, cols = df_train.columns).dropna()
    # Get the response
    labels = df_temp.loc[:, label]
    
    # List to store the fitted model
    models = []
    if all(predictor == predictors[0] for predictor in predictors):
        # If only the lags of flow rate are used as predictors, only need to fit one model
        df_input = df_temp.loc[:,predictors[0]]
        rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=3, n_jobs=-1) 
        rf.fit(df_input, labels)
        models = [rf] * time_steps
    else:
        # Fit multiple models
        for i in range(time_steps):
            predictor_cols = predictors[i]
            df_input = df_temp.loc[:,predictor_cols]
            rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=3, n_jobs=-1) 
            rf.fit(df_input, labels)
            models.append(rf)
    
    # Array to store the result
    result_rf = np.zeros((time_steps, N))
    for n in range(N):
        # Initialize a list to store the flow rate drawn from the predicted distribution with the last value
        # from the training set
        temp_list = [df_train.loc[:, label][-1]]
        for i in range(time_steps):            
            predictor_cols = predictors[i]
            # Get the predictors by adding new rows to the training set and calculate the lags
            preds = lag_df(df_train.append(pd.DataFrame({label: temp_list}), ignore_index=True), 
                                 lag = time_steps+n_lag-1, 
                                 cols = df_train.columns)[-1:].loc[:,predictor_cols]
            # Make the prediction for the ith step using the ith model
            pred_quantiles = rf_quantile(models[i], preds, quantiles)
            # Random draw a flow rate from the predicted distribution as one predictor for the next step
            result_rf[i, n] = np.random.choice(pred_quantiles)
            # Update the last value of the list with the drawn dample
            temp_list[-1] = result_rf[i, n]
            # Add the sample to the end of the list
            temp_list.append(result_rf[i, n])
                
    result_rf_df = pd.DataFrame(result_rf.T)
    return result_rf_df