import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost as xgb
from xgboost import plot_importance, XGBRegressor
from skrub import TableVectorizer
import matplotlib.pyplot as plt
import holidays

# ---------------- DEFINE THE FUNCTIONS --------------------------------

def weather_cleaning(weather):
    weather = weather.drop_duplicates()
    weather = weather[['date', 'ff', 't', 'ssfrai', 'etat_sol', 'ht_neige', 'rr1', 'rr3', 'rr6']]
    weather = weather.interpolate(method='linear', limit_direction='both')

    return weather


def extract_date_features(data, date_column='date'):
    data['hour'] = data[date_column].dt.hour
    data['weekday'] = data[date_column].dt.dayofweek
    data['month'] = data[date_column].dt.month
    data['year'] = data[date_column].dt.year
    data['weekend_day'] = data['weekday'].apply(lambda x: 1 if x in [5, 6] else 0)
    data['season'] = data['month'].apply(
        lambda x: 'spring' if x in [3, 4, 5]
        else 'winter' if x in [12, 1, 2]
        else 'summer' if x in [6, 7, 8]
        else 'autumn'
    )
    france_holidays = holidays.France(years=data['year'].unique())
    data['holidays'] = data[date_column].apply(lambda d: 1 if d in france_holidays else 0)
    return data

def add_rush_hour(data):
    data['is_rush_hour'] = data['hour'].apply(lambda x: 1 if (7 <= x <= 9 or 17 <= x <= 19) else 0)
    return data

def merge_external_data(data, weather):
    data['date'] = pd.to_datetime(data['date']).astype('datetime64[ns]')
    weather['date'] = pd.to_datetime(weather['date']).astype('datetime64[ns]')
    data["orig_index"] = np.arange(data.shape[0])
    merged_df = pd.merge_asof(
        data.sort_values("date"),
        weather.sort_values("date"),
        on="date"
    )
    merged_df = merged_df.sort_values("orig_index")
    del merged_df["orig_index"]
    return merged_df

def strikes (data):
    greves = pd.read_csv("mouvements-sociaux-depuis-2002.csv", sep=';')
    greves = greves[(greves['Date'] > '2020-09-01')]
    # only keep rows with date before 19 october 2021
    greves = greves[(greves['Date'] < '2021-10-19')]
    data['strike'] = data['date'].isin(greves['Date']).astype(int)
    return data
