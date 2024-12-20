import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import ydata_profiling
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb


data = pd.read_parquet("data/train.parquet")
data = data.drop(columns=['coordinates','counter_id','site_id','counter_technical_id','bike_count'])

weather = pd.read_csv('external_data/external_data.csv')

threshold = 0.7  # 90% threshold

# Calculate the percentage of NaN values for each column
nan_percentage = weather.isnull().mean()

# Identify columns where NaN percentage is greater than the threshold
columns_to_drop = nan_percentage[weather.isnull().mean() > threshold].index

# Print the columns that will be dropped
print(f"Columns dropped (NaN > 70%): {list(columns_to_drop)}")

# Drop those columns from the dataset
weather = weather.drop(columns=columns_to_drop)

# Find columns with null values and their counts
null_columns = weather.isnull().sum()[weather.isnull().sum() > 0].sort_values()

weather = weather.drop(columns=['nnuage1','ctype1','hnuage1','nnuage2','hnuage2','tend24','perssfrai','pmer','tend','cod_tend','n','hbas','cl','ch','cm','ctype2','numer_sta'])

#Here we interpolate all the numerical values:
columns_to_interpolate = ['nbas', 'rr3', 'rr1', 'raf10', 'rr6', 'rr12', 'rr24', 'ht_neige']

# Perform linear interpolation for each column
for col in columns_to_interpolate:
    weather[col] = weather[col].interpolate(method='linear', limit_direction='both')

# Forward and backward fill the categorical variables
weather['w1'] = weather['w1'].fillna(method='ffill').fillna(method='bfill')
weather['w2'] = weather['w2'].fillna(method='ffill').fillna(method='bfill')
weather['etat_sol'] = weather['etat_sol'].fillna(method='ffill').fillna(method='bfill')

# Assume 0 snowfall for missing values
weather['ssfrai'] = weather['ssfrai'].fillna(0)  

# Explicitly ensure both datasets have 'date' columns in datetime64[ns] format
data['date'] = pd.to_datetime(data['date']).astype('datetime64[ns]')
weather['date'] = pd.to_datetime(weather['date']).astype('datetime64[ns]')

# Sort both datasets by 'date'
bike_counts = data.sort_values('date')
weather = weather.sort_values('date')

# Perform the merge_asof with direction='nearest' and tolerance of 3 hours
merged_data = pd.merge_asof(
    bike_counts,
    weather,
    on='date',
    direction='nearest',
    tolerance=pd.Timedelta('3h')
)

# Interpolate the weather data to fill in NaN values
numeric_weather_columns = weather.columns.difference(['timestamp'])  # Identify numeric columns
for col in numeric_weather_columns:
    merged_data[col] = merged_data[col].interpolate(method='linear')  
    
# Define a dictionary with old column names as keys and new column names as values
rename_dict = {
    'dd' : 'mean_wind_direction',
    'ff' : 'mean_wind_speed',
    't' : 'temperature',
    'td' : 'dew_point',
    'u' : 'humidity_perc',
    'vv' : 'horizontal_visibility',
    'ww' : 'present_weather', #potentially categorical - check 
    'w1' : 'past_time',
    'w2' : 'past_time_2', #potentially to take out
    'nbas' : 'cloudiness_lower_level',
    'pres' : 'atmospheric_pressure',
    'raf10' : 'max_wind_gust_speed',
    'rafper' : 'max_wind_gust_over_period',
    'etat_sol' : 'state_of_ground',
    'ht_neige': 'depth_of_snow_ice_or_other',
    'ssfrai': 'height_of_snow',
    'rr1': 'rain_1',
    'rr3': 'rain_3',
    'rr6': 'rain_6',
    'rr9': 'rain_9'
}

# Rename the columns in the DataFrame
merged_data = merged_data.rename(columns=rename_dict)

merged_data = merged_data.drop(columns=['max_wind_gust_over_period','max_wind_gust_speed','past_time_2',])

# Feature engineering: create interaction terms
merged_data['temperature_rain'] = merged_data['temperature'] * merged_data['rain_1']
merged_data['temperature_humidity'] = merged_data['temperature'] * merged_data['humidity_perc']
merged_data['temperature_wind_speed'] = merged_data['temperature'] * merged_data['mean_wind_speed']
merged_data['rain_visibility'] = merged_data['rain_1'] * merged_data['horizontal_visibility']
merged_data['wind_humidity'] = merged_data['mean_wind_speed'] * merged_data['humidity_perc']


# 1. Day of the Week: Weekday or Weekend
# Assuming you have a 'date' column in datetime format
merged_data['day_of_week'] = pd.to_datetime(merged_data['date']).dt.dayofweek  # 0=Monday, 6=Sunday
merged_data['is_weekend'] = merged_data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)



def encode_time_bins(data, date_column, output_column='time_bin'):
    bins = [0, 6, 12, 17, 21, 24]  # Define time bins
    labels = ['Night', 'Morning', 'Afternoon', 'Evening', 'Late Night']  # Define bin labels

    # Compute time bins without adding the intermediate 'hour' column
    data[output_column] = pd.cut(
        pd.to_datetime(data[date_column]).dt.hour, 
        bins=bins, 
        labels=labels, 
        right=False
    )
    return data


# Apply the function to your dataset
merged_data = encode_time_bins(merged_data, date_column='date', output_column='time_bin')


# 3. Temperature Effects: Include temperature squared
merged_data['temperature_squared'] = merged_data['temperature'] ** 2

# 4. Wind Chill Factor
# Simplified formula for wind chill: T_wc = 13.12 + 0.6215T - 11.37v^0.16 + 0.3965Tv^0.16
# where T is temperature in °C and v is wind speed in km/h. Convert m/s to km/h by multiplying by 3.6
merged_data['wind_speed_kmh'] = merged_data['mean_wind_speed'] * 3.6
merged_data['wind_chill'] = (
    13.12 +
    0.6215 * merged_data['temperature'] -
    11.37 * (merged_data['wind_speed_kmh'] ** 0.16) +
    0.3965 * merged_data['temperature'] * (merged_data['wind_speed_kmh'] ** 0.16)
)

#create a season function

def encode_season(date_series):
    return pd.to_datetime(date_series).dt.month.apply(
        lambda x: 1 if x in [9, 10, 11] else
                  2 if x in [12, 1, 2] else
                  3 if x in [3, 4, 5] else
                  4
    )

# Apply the function to create the 'season' column
merged_data['season'] = encode_season(merged_data['date'])

# Add all new features to the dataset
print("New features added to the dataset:")
merged_data[['temperature_rain', 'temperature_humidity', 'temperature_wind_speed', 
                   'rain_visibility', 'wind_humidity', 'is_weekend', 'time_bin', 'temperature_squared', 'wind_chill', 'season']].head()

#define the function to help you with the first date encoding part:

def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the date columns
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["weekday"] = X["date"].dt.weekday
    X["hour"] = X["date"].dt.hour

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])

# we keep this function here for now, but then we can move it to utils

_target_column_name = "log_bike_count"

def get_train_data(data):
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name], axis=1)
    return X_df, y_array

X, y = get_train_data(merged_data)

def train_test_split_temporal(X, y, delta_threshold="30 days"):
    
    cutoff_date = X["date"].max() - pd.Timedelta(delta_threshold)
    mask = (X["date"] <= cutoff_date)
    X_train, X_valid = X.loc[mask], X.loc[~mask]
    y_train, y_valid = y[mask], y[~mask]

    return X_train, y_train, X_valid, y_valid



X_train, y_train, X_valid, y_valid = train_test_split_temporal(X, y)




# -----------------------  MODEL ---------------------------------------------------------------


# Step 1: Create the date feature transformer
date_encoder = FunctionTransformer(_encode_dates, validate=False)

# Step 2: Define categorical columns and OneHotEncoder
categorical_columns = ['counter_name', 'site_name', 'time_bin']
categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# Step 3: Define preprocessor with ColumnTransformer,
preprocessor = ColumnTransformer(
    transformers=[
        ("date", OneHotEncoder(handle_unknown="ignore"), ["year", "month", "day", "day_of_week", "hour"]),  # OneHotEncode the date features
        ("cat", categorical_encoder, categorical_columns),  # Encode the categorical columns
    ],
    remainder="drop"  # Drop all other columns not specified
)


import xgboost as xgb

# Replace Ridge with XGBoost in the pipeline
xgb_regressor = xgb.XGBRegressor(
    n_estimators=500,          # Number of boosting rounds
    learning_rate=0.05,        # Shrinks the contribution of each tree
    max_depth=7,               # Maximum tree depth
    subsample=0.8,             # Fraction of samples for each boosting round
    colsample_bytree=0.8,      # Fraction of features for each tree
    random_state=42            # For reproducibility
)

pipe = Pipeline(steps=[
    ("date_encoder", date_encoder),  # Extract date features
    ("preprocessor", preprocessor),  # Apply preprocessing
    ("regressor", xgb_regressor)     # Train XGBoost
])

pipe.fit(X_train, y_train)
