from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_engineered_features(df):
    """Create optimized feature set based on correlation analysis"""
    df = df.copy()

    # Define columns to drop
    cols_to_drop = ['counter_id', 'site_id', 'site_name',
                    'counter_installation_date', 'coordinates',
                    'counter_technical_id']

    # Only drop bike_count if it exists (in training data)
    if 'bike_count' in df.columns:
        cols_to_drop.append('bike_count')

    #Import holidays data
    holidays = pd.read_csv('external_data/Holidays/Propre_jours_feries.csv')
    holidays_sco = pd.read_csv('external_data/Holidays/Propre_vacances_scolaires.csv')

    #Define holidays columns and holidays_sco
    holidays['date'] = pd.to_datetime(holidays['date'])
    df['is_holiday'] = df['date'].dt.date.isin(holidays['date'].dt.date)
    holidays_sco['date'] = pd.to_datetime(holidays_sco['date'])
    df['is_holiday_sco'] = df['date'].dt.date.isin(holidays_sco['date'].dt.date)

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['month'] = df['date'].dt.month
    df['weekday'] = df['date'].dt.day_name()
    df['is_weekend'] = df['date'].dt.weekday >= 5

    # Keep only hour cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)

    # Create working day feature
    df['is_working_day'] = (~df['is_weekend']) & (~df['is_holiday'])
    morning_rush = (df['hour'].between(6, 8)) & df['is_working_day']
    evening_rush = (df['hour'].between(15, 17)) & df['is_working_day']
    df['is_rush_hour'] = (morning_rush | evening_rush).astype(bool)

    # Drop columns
    df = df.drop(columns=['hour', 'month'])
    df = df.drop(columns=cols_to_drop)
    df = df.drop(columns=['date'])  # Drop date after feature creation

    # Define feature groups
    numeric_features = ['latitude', 'longitude', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
    binary_features = ['is_weekend', 'is_holiday', 'is_holiday_sco', 'is_working_day', 'is_rush_hour']
    categorical_features = ['counter_name', 'weekday']

    reordered_columns = numeric_features + binary_features + categorical_features
    if 'log_bike_count' in df.columns:
        reordered_columns += ['log_bike_count']

    df = df.reindex(columns=reordered_columns)

    return df, numeric_features, binary_features, categorical_features

def prepare_data(df, is_train=None):
    # Detect if training data by presence of log_bike_count
    if is_train is None:
        is_train = 'log_bike_count' in df.columns

    transformed_df, num_feat, bin_feat, cat_feat = create_engineered_features(df)

    if is_train:
        X = transformed_df.drop(columns=['log_bike_count'])
        y = transformed_df['log_bike_count']
        return X, y, num_feat, bin_feat, cat_feat
    else:
        return transformed_df, num_feat, bin_feat, cat_feat

if __name__ == "__main__":
    train_df = pd.read_parquet(Path("data/train.parquet"))
    test_df = pd.read_parquet(Path("data/final_test.parquet"))

    X_train, y_train, num_feat, bin_feat, cat_feat = prepare_data(train_df, is_train=True)
    X_test = prepare_data(test_df, is_train=False)[0]

    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', Pipeline([
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(degree=2, include_bias=False))
            ]), num_feat),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_feat),
            ('bin', 'passthrough', bin_feat)
        ])),
        ('regressor', Ridge(alpha=1.0))
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    pd.DataFrame({
        'Id': range(len(predictions)),
        'log_bike_count': predictions
    }).to_csv('submission_ridge.csv', index=False)