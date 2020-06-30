# Linear Quantile Regression

This folder contains a notebook and two scripts to demenstrate how to train the linear quantile regression models and make multi-step ahead probabilistic forecast on the river flow.

In the notebook, [`linear_quantile_regression.ipynb`](https://github.ubc.ca/MDS-2019-20/DSCI_591-BGC/blob/data_product/doc/Linear_Quantile_Reg/linear_quantile_regression.ipynb), the multi-step predictions are demenstrated. A walk through is provided and the results of the linear quantile regression model are also shown.

**NOTE**: You **DO NOT** need to run the two scripts in this folder to run the notebook. The procedure of training the models can be time consuming.

The script, [`QR_preprocessing.py`](https://github.ubc.ca/MDS-2019-20/DSCI_591-BGC/blob/data_product/doc/Linear_Quantile_Reg/QR_preprocessing.py), is to read and combine the data, as well as splitting the data into training, validation and test sets. It can be changed when training the models for a new gauge.

The models can be re-trained by running `python QR_train.py` in the terminal. The parameters and predictors of the models can also be modified in this script.