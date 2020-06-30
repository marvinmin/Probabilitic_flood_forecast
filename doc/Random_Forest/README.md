# Qauntile Regression with Random Forest

This folder contains a notebook and two scripts to demenstrate how to use random forest model to perform quantile regression and make probabilistic forecast on the river flow.

In the notebook, [`RF_model.ipynb`](https://github.ubc.ca/MDS-2019-20/DSCI_591-BGC/blob/data_product/doc/Random_Forest/RF_model.ipynb), the roll forward and multi-step predictions are both demenstrated. A walk through is provided and the results of the random forest model are also shown. Sub-conclusions can also be found at the end of each section.

**NOTE**: You **DO NOT** need to run the two scripts in this folder to run the notebook. The procedure of training the models can be time consuming.

The models can be re-trained by running `python RF_train_roll_forward.py` and `python RF_train_multi_step.py` in the terminal. The parameters and predictors of the models can also be modified in these scripts.