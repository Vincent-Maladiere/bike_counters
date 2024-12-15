from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from vacances_scolaires_france import SchoolHolidayDates
from jours_feries_france import JoursFeries
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline

import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

# Utility functions

def date_encoder(df, date_col="date"):
    df = df.copy()
    """Encodes date-related features."""
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["weekday"] = df[date_col].dt.weekday
    df["hour"] = df[date_col].dt.hour
    # df.drop(columns=[date_col], inplace=True)
    return df

def cyclical_encoding(df, col, max_val):
    df = df.copy()
    df[col + "_sin"] = np.sin(2 * np.pi * df[col] / max_val)
    df[col + "_cos"] = np.cos(2 * np.pi * df[col] / max_val)
    df.drop(columns=[col], inplace=True)
    return df

def prepare_data(df_orig, external_df):
    df_orig['index'] = np.arange(df_orig.shape[0])

    # Handle missing values in the external dataset
    external_df.drop(columns=external_df.columns[(external_df.isnull().sum() / len(external_df)) > 0.1], inplace=True)
    external_df.fillna(external_df.median(numeric_only=True), inplace=True)

    # drop columns with zero or one unique value
    external_df.drop(columns=external_df.columns[external_df.nunique() == 0], inplace=True)
    external_df.drop(columns=external_df.columns[external_df.nunique() == 1], inplace=True)

    # Ensure datetime compatibility
    df_orig["date"] = df_orig["date"].astype('datetime64[us]')
    external_df["date"] = external_df["date"].astype('datetime64[us]')

    # Merge datasets
    df = pd.merge_asof(df_orig.sort_values("date"), external_df.sort_values("date"), on="date")

    # Add quarantine dates
    df["quarantine1"] = ((df["date"] >= "2020-10-30") & (df["date"] <= "2020-12-14")).astype(int)
    df["quarantine2"] = ((df["date"] >= "2020-04-03") & (df["date"] <= "2020-05-02")).astype(int)

    # Add holiday information
    school_holidays = SchoolHolidayDates()
    jours_feries = JoursFeries()

    # df['holidays'] = df.apply(
    #     lambda row: 1 if school_holidays.is_holiday_for_zone(row['date'].date(), 'C') else 0, axis=1
    # )

    # df['bank_holiday'] = df.apply(
    #     lambda row: 1 if jours_feries.is_bank_holiday(row['date'].date()) else 0, axis=1
    # )

    # Drop highly correlated columns
    df.drop(columns=["pres", "raf10", "rafper", "td", "w2"], axis=1, inplace=True)

    df.drop(columns=["counter_id", "site_id", "counter_installation_date", "counter_technical_id", "coordinates"], axis=1, inplace=True)
    # df = date_encoder(df)
    
    df = df.sort_values("index")

    return df

def create_preprocessor():
    """
    Creates a preprocessing pipeline that handles numerical, categorical, and date features.
    Returns a ColumnTransformer object that can be used in a sklearn pipeline.
    """
    # Numerical features
    numerical_features = ["tend", "cod_tend", "dd", "ff", "t", "u", "vv", "ww", "w1", "n", 
                         "nbas", "tend24", "etat_sol", "ht_neige", "rr1", "rr3", "rr6", 
                         "rr12", "rr24", "latitude", "longitude"]
    
    # Categorical features
    categorical_features = ["counter_name", "site_name"]
    
    # Date features
    cyclical_features = ["month", "day", "hour", "weekday"]
    
    def cyclical_transform(X):
        X = X.copy()
        # Cyclical encoding for temporal features
        X = cyclical_encoding(X, "month", 12)
        X = cyclical_encoding(X, "weekday", 7)
        X = cyclical_encoding(X, "hour", 24)
        X = cyclical_encoding(X, "day", 31)
        return X
    
    def date_transform(X):
        X = X.copy()
        return date_encoder(X)
    
    # Preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder='passthrough'
    )
    
    # Preprocessing pipeline
    full_preprocessor = Pipeline([
        ('date_encoding', FunctionTransformer(date_transform)),
        ('cyclical_encoding', FunctionTransformer(cyclical_transform)),
        ('column_transformer', preprocessor),
    ])
    
    return full_preprocessor

def build_pipeline(model):

    preprocessor = create_preprocessor()
    pipeline = make_pipeline(preprocessor, model)

    return pipeline

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model):
        
    pipeline = build_pipeline(model)
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return pipeline, rmse


def test_model_kaggle(pipeline, X_test2, model):
    
    y_pred = pipeline.predict(X_test2)

    results = pd.DataFrame(
        dict(
            Id=np.arange(y_pred.shape[0]),
            log_bike_count=y_pred,
        )
    )
    results.to_csv(f"//Users/solalzana/Desktop/X/Python for Data Science/Final Project/bike_counters/Submissions/submission_{model}.csv", index=False)

    return results, print("Submission file created, check data folder")

def tune_hyperparameters(pipeline, X_train, y_train):
    param_grid = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [10, 20, None],
        "model__min_samples_split": [2, 5, 10]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    return grid_search

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and compute RMSE."""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_pred, y_test))
    print(f"RMSE: {rmse}")
    print(y_test)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=np.array(X_test), y=y_pred, color="blue")
    sns.scatterplot(x=np.array(X_test), y=y_test, color="red")
    return rmse