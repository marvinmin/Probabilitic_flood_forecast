# Qauntile Regression with Gradient Boosted Trees

This folder contains a notebook and one script to demenstrate how to use gradient boosted trees model to perform quantile regression and make probabilistic forecast on the river flow.

In the notebook, [`gradient-boosted-trees_model.ipynb`](https://github.ubc.ca/MDS-2019-20/DSCI_591-BGC/blob/data_product/doc/Gradient_Boosted_Trees/gradient-boosted-trees_model.ipynb), the roll forward prediction is demenstrated. A walk through is provided and the results of the gradient boosted trees are also shown. Sub-conclusions can also be found at the end of each section.

**NOTE**: You **DO NOT** need to run the two scripts in this folder to run the notebook. The procedure of training the models can be time consuming.

The models can be re-trained by running `python gb_train_roll_forward.py` in the terminal. The parameters and predictors of the models can also be modified in the script.