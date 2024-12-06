import os

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import external_data.example_estimator as ex 

problem_title = "Bike count prediction"
_target_column_name = "log_bike_count"
# A type (class) which will be used to create wrapper objects for y_pred


def get_cv(X, y, random_state=0):
    cv = TimeSeriesSplit(n_splits=8)
    rng = np.random.RandomState(random_state)

    for train_idx, test_idx in cv.split(X):
        # Take a random sampling on test_idx so it's that samples are not consecutives.
        yield train_idx, rng.choice(test_idx, size=len(test_idx) // 3, replace=False)


def get_train_data(path="data/train.parquet"):
    data = pd.read_parquet(path)
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    
    data = ex._merge_external_data(data)
    
    # Dropping irrelevant columns
    data = data.drop(columns=["coordinates", "counter_id", "counter_name", "site_name",
                                            "counter_installation_date","counter_technical_id"])
    
    data = ex.calculate_sunrise_sunset_astral(data)
    data = ex._encode_dates(data)
    
    if _target_column_name in data.columns:
        y_array = data[_target_column_name].values
        X_df = data.drop([_target_column_name, "bike_count"], axis=1)
        
        return X_df, y_array
    else:
        return data