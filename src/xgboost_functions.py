import numpy as np
import pandas as pd
import datetime

from sklearn.model_selection import train_test_split

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.ensemble import GradientBoostingRegressor

from preprocessing import*

def gb_quantile(X_train, train_labels, X, q):
    """
    Fit the Gradient Boosting Regressor on the training set and the label,
    get the one-step ahead prediction on the validation set with a single quantile
    with the giving input
    
    Parameters
    ----------
    df_train: pandas.DataFrame
        The training dataset
    train_label: pandas.DataFrame
        The target in the training set
    X: pandas.DataFrame
        The input of hte `predict` method
    q: array-like float
        The alpha-quantile of the quantile loss function
    
    Return
    ------
    float
        The predicted flow record
    """
    # set the gradient boosted tree with the right parameter
    gbf = GradientBoostingRegressor(loss='quantile', alpha=q,
                                             n_estimators=100,
                                             max_depth=3,
                                             learning_rate=0.1, min_samples_leaf=9,
                                             min_samples_split=9)
    # fit the gradient boosted tree and predict the value
    gbf.fit(X_train, train_labels)
    
    return gbf.predict(X)



def gb_model(df_train, df_val, label, predictor_cols, n_lag=3, quantiles=np.arange(.01, .99, .02), N_ESTIMATORS=1000):
    """
    Train the Gradient Boosted Trees on the train set and make the roll-forward 
    one-step ahead quantile prediction on the validation set using the origin 
    flow rates as the response.
    
    Parameters
    ----------
    df_train: pandas.DataFrame
        The training dataset
    df_val: pandas.DataFrame
        The validation dataset
    predictor_cols: list of strings
        The list of columns to be lagged and then to train the model
    label: str
        The name of label in the training set
    n_lag: int (default:3)
        The number of lags to fit the model
    quantiles: array-like of float (default:np.arange(.01, .99, .02))
        The sequence of quantiles to compute
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
    
    # Initializing a dictionary to store the result, the key of the dictionary would be the dates (The index of the validation set)
    result = {}
    
    # An empty list to store the feature inportance of all the fitted forests
    importance = []
    
    # Fit a gradient boosted trees model for every row (prediction) on the validation set
    for k in range(len(df_val)):
        
        # Get the lags of columns in `predictor_cols` in the training set and drop NAs
        # also get the list of lagged columns to be used later
        df_temp, lag_list = lag(n_lag, df_temp, predictor_cols)
        df_temp = df_temp.dropna()
        
        # Select the chosen predictors from the lagged dataframe as the input
        df_input = df_temp.loc[:,lag_list]
        
        # Choose the column to be the labels
        labels = df_temp.loc[:, label]
        
        # Initialize the dataframe for predictors to do the prediction,
        # the index of the dataframe should be the same as the row from 
        # the validation set that we want to predict
        df_pred = pd.DataFrame(df_val.loc[df_val.index[k], :]).T
        
        # Get all the needed lags from the last few rows of the training set
        for col in predictor_cols:
            for i in range(n_lag):
                df_pred[col+f'{i+ 1}'] = df_temp[col].values[-1-i]
        
        # Select the chosen lagged predictors
        df_pred = df_pred.loc[:,lag_list]
        
        # fit a gradient boosted model for every quantile
        pred_flow = np.concatenate(
            [gb_quantile(df_input, labels, df_pred, q) for q in quantiles])
        
        # append the predicted quantiles into the result dictionary
        result[df_val.index[k]] = pred_flow
        
        # fitting a 50 quantile model to get the feature importance
        gbf_50 = GradientBoostingRegressor(loss='quantile', alpha=0.5,
                                             n_estimators=100,
                                             max_depth=3,
                                             learning_rate=0.1, min_samples_leaf=9,
                                             min_samples_split=9)
        gbf_50.fit(df_input, labels)
        
        # get the importance of the features 
        importance.append(gbf_50.feature_importances_)
        
        # update the training set by appending the row from the validatio nset that has been predicted
        df_train = df_train.append(df_val.iloc[k,:])
        df_temp = df_train.copy()

    return result, np.mean(importance, axis = 0)