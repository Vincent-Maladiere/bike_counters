from pathlib import Path
import numpy as np
import pandas as pd
from astral.sun import sun
from astral.geocoder import LocationInfo
import pytz
import os

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge



def weather_cleaning(weather):
    """
    Cleans the weather dataset by handling missing values and dropping unnecessary columns.

    Steps:
    - Impute missing values using mode, median, or mean as appropriate.
    - Fill precipitation and snow-related columns with 0.
    - Drop columns with more than 1000 missing values.

    Args:
        weather (pd.DataFrame): The raw weather dataset.

    Returns:
        pd.DataFrame: The cleaned weather dataset.
    """
    # important to use a copy
    weather = weather.copy()

    mode_columns = ['w1', 'w2', 'n', 'cl', 'cm', 'ch', 'etat_sol', 'nnuage1', 'ctype1']
    for col in mode_columns:
        weather[col].fillna(weather[col].mode()[0], inplace=True)

    # Impute using mean or median for numerical variables
    weather['hnuage1'].fillna(int(weather['hnuage1'].mean()), inplace=True)
    weather['hbas'].fillna(int(weather['hbas'].mean()), inplace=True)
    weather['nbas'].fillna(int(weather['nbas'].mean()), inplace=True)
    weather['tend24'].fillna(weather['tend24'].median(), inplace=True)
    weather['raf10'].fillna(weather['raf10'].median(), inplace=True)

    # Fill precipitation and snow-related columns with 0
    zero_fill_columns = ['ht_neige', 'ssfrai', 'perssfrai', 'rr1', 'rr3', 'rr6', 'rr12', 'rr24']
    weather[zero_fill_columns] = weather[zero_fill_columns].fillna(0)

    # Drop columns with more than 1000 missing values
    columns_to_drop = weather.columns[weather.isnull().sum() > 1000]
    weather.drop(columns=columns_to_drop, inplace=True)

    return weather



def calculate_sunrise_sunset_astral(df):
    """
    Calculate if the sun is up for each timestamp in the dataframe, adding a column to the df.
    Assumes a 'date' column with complete timestamps and 'latitude', 'longitude' columns.
    """
    def is_sun_up(row):
        location = LocationInfo(
            name="Custom",
            region="Custom",
            timezone="Europe/Paris",
            latitude=row['latitude'],
            longitude=row['longitude']
        )
        # Parse the date from the row and localize it to Paris timezone
        date = pd.Timestamp(row['date']).tz_localize('Europe/Paris')
        # Get sunrise and sunset times
        s = sun(location.observer, date)
        sunrise = s['sunrise'].astimezone(pytz.timezone(location.timezone))
        sunset = s['sunset'].astimezone(pytz.timezone(location.timezone))
        # Check if the timestamp is within sunrise and sunset
        return sunrise <= date <= sunset

    # Apply the function and create the is_sun_up column
    df['is_sun_up'] = df.apply(is_sun_up, axis=1)
    return df



def train_test_split_temporal(X, y, delta_threshold="30 days"):
    """
    Split the data into training and validation sets based on a temporal cutoff.
    Args:
        X (pd.DataFrame): Features with a `date` column.
        y (pd.Series): Target variable.
        delta_threshold (str): Time delta defining the validation cutoff.
    Returns:
        Tuple: X_train, y_train, X_valid, y_valid
    """
    cutoff_date = X["date"].max() - pd.Timedelta(delta_threshold)
    mask = (X["date"] <= cutoff_date)
    X_train, X_valid = X.loc[mask], X.loc[~mask]
    y_train, y_valid = y[mask], y[~mask]
    return X_train, y_train, X_valid, y_valid


def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])


def _merge_external_data(X, external_data_path="external_data/external_data.csv", merge_columns=None, additional_functions=None):
    """
    Merges the initial dataset X with an external dataset, aligning by the closest timestamp.
    
    Args:
        X (pd.DataFrame): The initial dataset, must have a 'date' column.
        external_data_path (str): Path to the external dataset CSV file.
        merge_columns (list or None): Columns from the external dataset to merge.
            If None, all columns except 'date' will be merged.
    
    Returns:
        pd.DataFrame: Enriched dataset with external data.
    """
    # Load the external dataset
    file_path = os.path.join(os.getcwd(), external_data_path)
    external_data = pd.read_csv(file_path, parse_dates=["date"])
    # Function to clean this dataset. But it may be a different one for different datasets
    external_data = weather_cleaning(external_data)
    # Ensure 'date' columns are in datetime format
    X = X.copy()
    external_data = external_data.copy()
    X['date'] = pd.to_datetime(X['date'])
    external_data['date'] = pd.to_datetime(external_data['date'])

    # Default to merging all columns except 'date' if not specified
    if merge_columns is None:
        merge_columns = [col for col in external_data.columns if col != "date"]

    # Add a temporary index to restore original order later
    X["orig_index"] = np.arange(X.shape[0])

    # Perform the as-of merge
    enriched_data = pd.merge_asof(
        X.sort_values("date"),
        external_data[["date"] + merge_columns].sort_values("date"),
        on="date",
        direction="nearest"
    )

    # Restore the original order and clean up temporary columns
    enriched_data = enriched_data.sort_values("orig_index").drop(columns=["orig_index"])

    return enriched_data


def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ["year", "month", "day", "weekday", "hour"]

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name", "site_name"]

    preprocessor = ColumnTransformer(
        [
            ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
            ("cat", categorical_encoder, categorical_cols),
        ]
    )
    regressor = Ridge()

    pipe = make_pipeline(
        FunctionTransformer(_merge_external_data, validate=False),
        date_encoder,
        preprocessor,
        regressor,
    )

    return pipe