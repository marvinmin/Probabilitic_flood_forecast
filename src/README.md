# Function Scripts

This folder contains multiple scripts that include helper and main functions to run the models and a `unit_tests` folder. Here is a brief introduction to each script. A more detailed description of functions can be found in the scripts.

[`model_evaluation.py`](https://github.ubc.ca/MDS-2019-20/DSCI_591-BGC/blob/data_product/src/model_evaluation.py) contains functions that evaluate model performances (`get_quantile_loss`, ` avg_quantile_loss`) and plotting functions(`plot_quantiles`, `plot_multiple_step_prediction`).  

[`preprocessing.py`](https://github.ubc.ca/MDS-2019-20/DSCI_591-BGC/blob/data_product/src/preprocessing.py) contains functions that are used for preprocess the data (`read_data`, `choose_gauge`, `add_narr`, `choose_month_and_split`, `read_hourly_data`, `add_hourly_narr`, `lag_df`, `lag`).

[`spatial_functions.py`](https://github.ubc.ca/MDS-2019-20/DSCI_591-BGC/blob/data_product/src/spatial_functions.py) contains functions process the spatial (NARR) data (`create_ploygon_grid`, `get_cells_and_distances`, `plot_distance`, `get_NARR_dataframe`).

[`quantile_regression_functions.py`](https://github.ubc.ca/MDS-2019-20/DSCI_591-BGC/blob/data_product/src/quantile_regression_functions.py) contains functions that fit and predict one-step ahead (`QR_model_origin_flow`) and multi-step ahead (`QR_multi_step_predict`) quantiles using the quantile regression model with spatial data.

[`random_forest_functions.py`](https://github.ubc.ca/MDS-2019-20/DSCI_591-BGC/blob/data_product/src/random_forest_functions.py) contains functions that fit and predict one-step ahead (`RF_model_origin_flow`) and multi-step ahead (`RF_multi_step_predict`) quantiles using the random forest model with spatial data. Helper function (`rf_quantile`) calculates the quantiles of the random forest model is also included.

[`xgboost_functions.py`](https://github.ubc.ca/MDS-2019-20/DSCI_591-BGC/blob/data_product/src/xgboost_functions.py) contains functions that fit (`gb_quantile`) and predict one-step ahead (`gb_model`) quantiles using gradient boosted tree with spatial data. 

[`exact_prob_functions.r`](https://github.ubc.ca/MDS-2019-20/DSCI_591-BGC/blob/data_product/src/exact_prob_functions.r) contains R functions (`quantiles_to_points`, `condense_points`, `create_input_data`) that used for preprocessing, deterministic flows. 
