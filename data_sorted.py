import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import FunctionTransformer

def load_and_clean_external_data(filepath):
    """Load and clean external data, handling missing values and date formatting."""
    external_data = pd.read_csv(filepath)
    external_data_cleaned = external_data.dropna(axis=1, how='all')

    columns_of_interest = ["date", "etat_sol", "dd", "ff", "t", "u", "vv", "n", "ht_neige", "rr3"]
    external_data_sorted = external_data_cleaned[columns_of_interest].copy()
    external_data_sorted["date"] = pd.to_datetime(external_data_sorted['date'])

    # Convert temperature from Kelvin to Celsius
    external_data_sorted.loc[:, "t"] = external_data_sorted["t"] - 273.15

    return external_data_sorted

def add_covid_restrictions(df):
    """Add COVID-related restriction periods as features."""
    covid_periods = {
        'Lockdown': [
            ('2020-10-30', '2020-12-15'),
            ('2021-04-03', '2021-05-04')
        ],
        'soft-curfew': [
            ('2020-10-17', '2020-10-30'),
            ('2020-12-15', '2021-01-16'),
            ('2021-05-19', '2021-06-21')
        ],
        'hard-curfew': [
            ('2021-01-16', '2021-04-03'),
            ('2021-05-04', '2021-05-19')
        ]
    }

    for restriction_type, periods in covid_periods.items():
        df[restriction_type] = 0
        for start_date, end_date in periods:
            mask = (df['date'] >= start_date) & (df['date'] < end_date)
            df.loc[mask, restriction_type] = 1

    return df

def expand_hourly_data(df):
    """Create missing hourly data points by copying existing rows."""
    def create_missing_hours(row):
        return [
            {**row.to_dict(), 'date': row['date'] - pd.Timedelta(hours=h)}
            for h in [2, 1]
        ]

    new_rows = []
    for _, row in df.iterrows():
        new_rows.extend(create_missing_hours(row))

    expanded_df = pd.concat([
        df,
        pd.DataFrame(new_rows)
    ], ignore_index=True)

    return expanded_df.sort_values(by='date').reset_index(drop=True)

def add_temporal_features(df):
    """Add various temporal features that might be useful for prediction."""
    df = df.copy()

    # Basic time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['hour'] = df['date'].dt.hour

    # Additional useful time features
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    holidays = pd.to_datetime(['2020-11-01', '2020-11-11', '2020-12-25', '2021-01-01', '2021-04-05',
                               '2021-05-01', '2021-05-13', '2021-05-24', '2021-07-14', '2021-08-15'])
    df['is_holiday'] = df['date'].isin(holidays).astype(int)
    df['season'] = df['month'].map(lambda m: (m%12 + 3)//3)

    return df

def main():
    # Load external data
    external_data = load_and_clean_external_data("external_data/external_data.csv")

    # Add COVID restrictions
    external_data = add_covid_restrictions(external_data)

    # Expand to hourly data
    external_data = expand_hourly_data(external_data)

    # Load training data
    train_data = pd.read_parquet("data/train.parquet")

    # Merge datasets
    final_data = pd.merge(train_data, external_data, on='date', how='inner')

    # Add temporal features
    final_data = add_temporal_features(final_data)

    # Sort by date and counter
    final_data = final_data.sort_values(['date', 'counter_name'])

    return final_data

if __name__ == "__main__":
    processed_data = main()
    processed_data.drop(columns=['counter_name', 'site_id', 'site_name', 'bike_count', 'date', 'counter_installation_date', 'coordinates', 'counter_technical_id', 'log_bike_count', 'season'], inplace=True)
    print("Data shape:", processed_data.shape)
    print("\nSample of processed data:")
    print(processed_data.dtypes)
    processed_data.to_csv("processed_data.csv", index=False)