from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

def prepare_data(df, weather_df, covid_df, is_train=None):
    df = df.copy()

    df['date'] = pd.to_datetime(df['date']).dt.floor('h')
    weather_df['date'] = pd.to_datetime(weather_df['date']).dt.floor('h')
    covid_df['date'] = pd.to_datetime(covid_df['date']).dt.floor('h')
    df = pd.merge(df, weather_df, on='date', how='left')
    df = pd.merge(df, covid_df, on='date', how='left')

    holidays = pd.read_csv('external_data/Holidays/Propre_jours_feries.csv')
    holidays_sco = pd.read_csv('external_data/Holidays/Propre_vacances_scolaires.csv')

    holidays['date'] = pd.to_datetime(holidays['date'])
    df['is_holiday'] = df['date'].dt.date.isin(holidays['date'].dt.date)
    holidays_sco['date'] = pd.to_datetime(holidays_sco['date'])
    df['is_holiday_sco'] = df['date'].dt.date.isin(holidays_sco['date'].dt.date)

    df['hour'] = df['date'].dt.hour
    df['month'] = df['date'].dt.month
    df['weekday'] = df['date'].dt.day_name()
    df['is_weekend'] = df['date'].dt.weekday >= 5

    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)

    df['is_working_day'] = (~df['is_weekend']) & (~df['is_holiday'])
    morning_rush = (df['hour'].between(6,8)) & df['is_working_day']
    evening_rush = (df['hour'].between(15,17)) & df['is_working_day']
    df['is_rush_hour'] = (morning_rush | evening_rush)

    cols_to_drop = ['counter_id', 'site_id', 'site_name', 'counter_installation_date',
                    'coordinates', 'counter_technical_id', 'hour', 'month', 'date']
    if 'bike_count' in df.columns:
        cols_to_drop.append('bike_count')

    df = df.drop(columns=cols_to_drop)

    if is_train:
        X = df.drop(columns=['log_bike_count'])
        y = df['log_bike_count']
        return X, y
    return df

if __name__ == "__main__":
    # Load data
    train_df = pd.read_parquet(Path("data/train.parquet"))
    test_df = pd.read_parquet(Path("data/final_test.parquet"))
    weather_df = pd.read_csv('external_data/Weather/Propre_meteo.csv')
    covid_df = pd.read_csv('external_data/Covid/Propre_nbr_Covid.csv')

    # Prepare features
    X_train, y_train = prepare_data(train_df, weather_df, covid_df, is_train=True)
    X_test = prepare_data(test_df, weather_df, covid_df, is_train=False)

    ## Best parameters from optuna to keep them
    params = {
        'iterations': 500,
        'learning_rate': 0.12,
        'depth': 12,
        'bagging_temperature': 2e-8,
        'l2_leaf_reg': 2e-3,
        'border_count': 84,
        'od_type': 'IncToDec',
        'od_wait': 24
    }

    # Train ensemble
    predictions_list = []
    for seed in [42, 123, 456]:
        model = CatBoostRegressor(**params, random_seed=seed, cat_features=['counter_name', 'weekday'])
        model.fit(X_train, y_train, verbose=False)
        predictions_list.append(model.predict(X_test))

    # Average predictions
    predictions = np.mean(predictions_list, axis=0)

    pd.DataFrame({
        'Id': range(len(predictions)),
        'log_bike_count': predictions
    }).to_csv('submission2.csv', index=False)