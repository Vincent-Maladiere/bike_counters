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

import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

# Utility functions

def date_encoder(df, date_col="date"):
    """Encodes date-related features."""
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["weekday"] = df[date_col].dt.weekday
    df["hour"] = df[date_col].dt.hour
    # df.drop(columns=[date_col], inplace=True)
    return df

def cyclical_encoding(df, col, max_val):
    """Encodes a cyclical feature."""
    df[col + "_sin"] = np.sin(2 * np.pi * df[col] / max_val)
    df[col + "_cos"] = np.cos(2 * np.pi * df[col] / max_val)
    # df.drop(columns=[col], inplace=True)
    return df

def prepare_data(df, external_df):
    """
    Prepares the dataset by merging external data and encoding features. 
    Does everything done and explained in the EDA notebook to clean data 
    and feature engineer so it can be done on the test set too.
    """

    df['original_index'] = df.index

    # Handle missing values in the external dataset
    external_df.drop(columns=external_df.columns[(external_df.isnull().sum() / len(external_df)) > 0.1], inplace=True)
    external_df.fillna(external_df.median(numeric_only=True), inplace=True)

    # Ensure datetime compatibility
    df["date"] = df["date"].astype('datetime64[us]')
    external_df["date"] = external_df["date"].astype('datetime64[us]')

    # Merge datasets
    df = pd.merge_asof(df.sort_values("date"), external_df.sort_values("date"), on="date")

    # drop columns with zero or one unique value
    df.drop(columns=df.columns[df.nunique() == 0], inplace=True)
    df.drop(columns=df.columns[df.nunique() == 1], inplace=True)

    # Add quarantine dates
    df["quarantine1"] = ((df["date"] >= "2020-10-30") & (df["date"] <= "2020-12-14")).astype(int)
    df["quarantine2"] = ((df["date"] >= "2020-04-03") & (df["date"] <= "2020-05-02")).astype(int)

    # Add holiday information
    school_holidays = SchoolHolidayDates()
    jours_feries = JoursFeries()

    df['holidays'] = df.apply(
        lambda row: 1 if school_holidays.is_holiday_for_zone(row['date'].date(), 'C') else 0, axis=1
    )

    df['bank_holiday'] = df.apply(
        lambda row: 1 if jours_feries.is_bank_holiday(row['date'].date()) else 0, axis=1
    )

    # Drop highly correlated columns
    df.drop(columns=["pres", "raf10", "rafper", "td", "w2"], inplace=True)

    df.drop(columns=["counter_id", "site_id", "counter_installation_date", "counter_technical_id", "coordinates"], axis=1, inplace=True)
    df = date_encoder(df)
    df = df.sort_values(by='original_index').drop(columns=['original_index'])

    # df = df.reset_index(drop=True)
    # df = cyclical_encoding(df, "month", 12)
    # df = cyclical_encoding(df, "weekday", 7)
    # df = cyclical_encoding(df, "hour", 24)
    
    return df    

def encoder(X):
    
    def _encode_categorical_features(X):
        X = X.copy()
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded_features = encoder.fit_transform(X[["counter_name", "site_name"]])
        encoded_df = pd.DataFrame(
            encoded_features,
            columns=encoder.get_feature_names_out(["counter_name", "site_name"]),
            index=X.index
        )
        # Drop original columns and add encoded features
        X = X.drop(columns=["counter_name", "site_name"], errors="ignore")
        X = pd.concat([X, encoded_df], axis=1)

def build_pipeline(X_train, y_train, model):
    """Builds a preprocessing and modeling pipeline."""
    num_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(handle_unknown="ignore")

    # Extract column names
    num_cols = ["tend", "cod_tend", "dd", "ff", "t", "u", "vv", "ww", "w1", "n", 
                "nbas", "tend24", "etat_sol", "ht_neige", "rr1", "rr3", "rr6", 
                "rr12", "rr24", "latitude", "longitude"]
    cat_cols = ["counter_name", "site_name", "month", "day", "weekday", "hour"]

    # Ensure the columns exist in the DataFrame
    num_cols = [col for col in num_cols if col in X_train.columns]
    cat_cols = [col for col in cat_cols if col in X_train.columns]
    print("Numerical columns:", num_cols)
    print("Categorical columns:", cat_cols)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols)
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    return pipeline

def train_model(pipeline, model, X_train, y_train, X_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    submission = pd.DataFrame(y_pred, columns=["log_bike_count"])
    submission.rename(columns={"index": "Id"}).rename(columns={"index": "Id"}).to_csv(f"//Users/solalzana/Desktop/X/Python for Data Science/Final Project/bike_counters/Submissions/submission_{model}.csv", index=False)
    
    return pipeline, print("Submission file created, check data folder")

def tune_rf(pipeline, X_train, y_train):
    """Tunes hyperparameters using GridSearchCV."""
    param_grid = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [10, 20, None],
        "model__min_samples_split": [2, 5, 10]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    return grid_search

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and computes RMSE."""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_pred, y_test))
    print(f"RMSE: {rmse}")
    return rmse

def main():
    # Load data
    data = pd.read_parquet("data/train.parquet")
    external_data = pd.read_parquet("data/external_data.parquet")

    # Prepare data
    data = prepare_data(data, external_data)

    # Split data
    X = data.drop(columns=["log_bike_count", "bike_count"])
    y = data["log_bike_count"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build pipeline
    model = RandomForestRegressor()
    pipeline = build_pipeline(X_train, y_train, model)

    # Train model
    pipeline, _ = train_model(pipeline, model, X_train, y_train, X_test)

    # Evaluate model
    evaluate_model(pipeline, X_test, y_test)

df_train = pd.read_parquet("/Users/solalzana/Desktop/X/Python for Data Science/Final Project/bike_counters/data/train.parquet")
df_test = pd.read_parquet("/Users/solalzana/Desktop/X/Python for Data Science/Final Project/bike_counters/data/final_test.parquet")
df_ext = pd.read_csv("/Users/solalzana/Desktop/X/Python for Data Science/Final Project/bike_counters/data/external_data.csv")

# Prepare data
df_train_cleaned = prepare_data(df_train, df_ext)
df_test_cleaned = prepare_data(df_test, df_ext)

# train/test split
X_train = df_train_cleaned.drop(columns=["log_bike_count", "bike_count"])
X_test = df_test_cleaned
y_train = df_train_cleaned['log_bike_count']

# Build pipeline
import xgboost as xgb

model = xgb.XGBRegressor(objective='reg:squarederror')
pipeline_xgb = build_pipeline(X_train, y_train, model)
y_pred = train_model(pipeline_xgb, "xgb", X_train, y_train, X_test)