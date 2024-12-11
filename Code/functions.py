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
    """Encodes a cyclical feature."""
    df[col + "_sin"] = np.sin(2 * np.pi * df[col] / max_val)
    df[col + "_cos"] = np.cos(2 * np.pi * df[col] / max_val)
    # df.drop(columns=[col], inplace=True)
    return df

def prepare_data(df_orig, external_df):
    df_orig['index'] = np.arange(df_orig.shape[0])
    """
    Prepares the dataset by merging external data and encoding features. 
    Does everything done and explained in the EDA notebook to clean data 
    and feature engineer so it can be done on the test set too.
    """

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

    # # Add quarantine dates
    # df["quarantine1"] = ((df["date"] >= "2020-10-30") & (df["date"] <= "2020-12-14")).astype(int)
    # df["quarantine2"] = ((df["date"] >= "2020-04-03") & (df["date"] <= "2020-05-02")).astype(int)

    # # Add holiday information
    # school_holidays = SchoolHolidayDates()
    # jours_feries = JoursFeries()

    # df['holidays'] = df.apply(
    #     lambda row: 1 if school_holidays.is_holiday_for_zone(row['date'].date(), 'C') else 0, axis=1
    # )

    # df['bank_holiday'] = df.apply(
    #     lambda row: 1 if jours_feries.is_bank_holiday(row['date'].date()) else 0, axis=1
    # )

    # Drop highly correlated columns
    df.drop(columns=["pres", "raf10", "rafper", "td", "w2"], axis=1, inplace=True)

    df.drop(columns=["counter_id", "site_id", "counter_installation_date", "counter_technical_id", "coordinates"], axis=1, inplace=True)
    df = date_encoder(df)
    
    df = df.sort_values("index")
    # df = df.drop(columns=["index"])
    # df = cyclical_encoding(df, "month", 12)
    # df = cyclical_encoding(df, "weekday", 7)
    # df = cyclical_encoding(df, "hour", 24)
    # df = cyclical_encoding(df, "day", 31)
    # df.drop(columns=["date"], inplace=True)

    return df

def encoder(X):
    
    # 1. numerical features
    num_cols = ["tend", "cod_tend", "dd", "ff", "t", "u", "vv", "ww", "w1", "n", 
                "nbas", "tend24", "etat_sol", "ht_neige", "rr1", "rr3", "rr6", 
                "rr12", "rr24", "latitude", "longitude"]

    num_cols = [col for col in num_cols if col in X.columns]
    num_transformer = StandardScaler()
    X[num_cols] = num_transformer.fit_transform(X[num_cols])

    # 2. categorical features
    X = date_encoder(X)
    encoderr = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_features = encoderr.fit_transform(X[["counter_name", "site_name"]])
    encoded_df = pd.DataFrame(
        encoded_features,
        columns=encoderr.get_feature_names_out(["counter_name", "site_name"]),
        index=X.index
    )
    # Drop original columns and add encoded features
    X = X.drop(columns=["counter_name", "site_name"], errors="ignore")
    X = pd.concat([X, encoded_df], axis=1)

    # 3. date features
    date_encoder2 = FunctionTransformer(cyclical_encoding)
    X = date_encoder2.fit_transform(X, "month", 12)
    X = date_encoder2.fit_transform(X, "weekday", 7)
    X = date_encoder2.fit_transform(X, "hour", 24)
    X = date_encoder2.fit_transform(X, "day", 31)

    # X = cyclical_encoding(X, "month", 12)
    # X = cyclical_encoding(X, "weekday", 7)
    # X = cyclical_encoding(X, "hour", 24)
    # X = cyclical_encoding(X, "day", 31)

    X.reset_index(drop=True, inplace=True)
    X.drop(columns=["date"], inplace=True)
    pd.set_option('display.max_columns', None)
    print(X.columns)

    return X

def fit_encoder(X_train):
    X_train = date_encoder(X_train)

    global year_encoder, month_encoder, weekday_encoder, category_encoder, numerical_encoder
    year_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(X_train[["year"]])
    month_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(X_train[["month"]])
    weekday_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(X_train[["weekday"]])
    category_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(X_train[['counter_name', 'site_name']])
    numerical_encoder = StandardScaler().fit(X_train[[ 'cod_tend', 't', 'u', 'etat_sol']])


def encoder(X):
    X = X.copy()
    X = date_encoder(X)

    X = cyclical_encoding(X, 'hour', 23)
    X = cyclical_encoding(X, 'day', 31)

    years_encoded = year_encoder.transform(X[["year"]])
    months_encoded = month_encoder.transform(X[["month"]])
    weekdays_encoded = weekday_encoder.transform(X[["weekday"]])
    categories_encoded = category_encoder.transform(X[['counter_name', 'site_name']])
    numerical_encoded = numerical_encoder.transform(X[[ 'cod_tend', 't', 'u', 'etat_sol']])
    
    years_df = pd.DataFrame(years_encoded, columns=["2020","2021"])
    months_df = pd.DataFrame(months_encoded, columns=["janv","fev","mars","avril","mai","juin","juillet","aout","sept","octobre","novem","decembre"])
    weekdays_df = pd.DataFrame(weekdays_encoded, columns=[f"weekday_{i}" for i in range(weekdays_encoded.shape[1])])
    categories_df = pd.DataFrame(categories_encoded, columns=[f"cat_{i}" for i in range(categories_encoded.shape[1])])
    numercial_df = pd.DataFrame(numerical_encoded, columns = ([ 'cod_tend', 't', 'u', 'etat_sol']))
    
    X.reset_index(drop=True, inplace=True)

    # Concatenate all features
    X = pd.concat([X, years_df, months_df, weekdays_df, categories_df,numercial_df], axis=1)
    X.drop(columns=['year', 'date', 'month', 'weekday', 'day','hour', 'counter_name', 'site_name','cod_tend', 't', 'u', 'etat_sol'], inplace = True)
    
    return X

def build_pipeline(X_train, y_train, model):

    pipeline = make_pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    return pipeline

def train_model(X_train, y_train, model):

    transformer = FunctionTransformer(encoder)
    pipeline = make_pipeline(transformer, model)
    pipeline.fit(X_train, y_train)

    return pipeline


def test_model_kaggle(pipeline, X_test, model):
    
    y_pred = pipeline.predict(X_test)

    results = pd.DataFrame(
        dict(
            Id=np.arange(y_pred.shape[0]),
            log_bike_count=y_pred,
        )
    )
    results.to_csv(f"//Users/solalzana/Desktop/X/Python for Data Science/Final Project/bike_counters/Submissions/submission_{model}.csv", index=False)

    return results, print("Submission file created, check data folder")

def tune_hyperparameters(pipeline, X_train, y_train):
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
    print(y_test)
    plt.figure(figsize=(10, 6))
    # sns.scatterplot(x=X_test, y=y_pred, color="blue")
    # sns.scatterplot(x=X_test, y=y_test, color="red")
    return rmse