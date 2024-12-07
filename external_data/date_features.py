import pandas as pd
from astral.sun import sun
from astral.geocoder import LocationInfo
from jours_feries_france import JoursFeries
import pytz
import numpy as np



# Dividing a day into 4 relevant sections
def assign_time_interval(hour):
    if 5 <= hour < 9:
        return 'morning'
    elif 9 <= hour < 15:
        return 'working_hours'
    elif 15 <= hour < 20:
        return 'peak_hours'
    else:
        return 'calm'

def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour
    X['is_weekend'] = X['weekday'].apply(lambda x: 1 if x >= 5 else 0)
    
    X['season'] = X['month'] % 12 // 3 # Winter=0, Spring=1, Summer=2, Fall=3
    X['time_interval'] = X['hour'].apply(assign_time_interval)

    # Cyclical encoding
    X['hour_sin'] = np.sin(2 * np.pi * X['hour']/24)
    X['hour_cos'] = np.cos(2 * np.pi * X['hour']/24)
    X['day_sin'] = np.sin(2 * np.pi * X['weekday']/7)
    X['day_cos'] = np.cos(2 * np.pi * X['weekday']/7)

    # One-hot encoding time_interval
    X = pd.get_dummies(X, columns=['time_interval'], prefix='time')
    
    # One-hot encoding for day_of_week and season
    #X = pd.get_dummies(X, columns=['weekday', 'season'], prefix=['day', 'season'])

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])




# Long function but it is well optimized (80 s --> 10 s runtime !!)
def calculate_sunrise_sunset_astral(df):
    """
    Calculate if the sun is up for each timestamp in the dataframe, adding a column to the df.
    Assumes a 'date' column with complete timestamps and 'latitude', 'longitude' columns.
    """
    # Convert 'date' to datetime, coerce errors to NaT
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    
    # Drop rows where 'date' is NaT
    df = df.dropna(subset=['date'])
    
    # Convert 'date' to Europe/Paris timezone
    df['date'] = df['date'].dt.tz_convert('Europe/Paris')
    
    # Extract 'date_only' (date without time)
    df['date_only'] = df['date'].dt.date
    
    # Drop any rows where 'date_only' is NaT, just in case
    df = df.dropna(subset=['date_only'])
    
    # Get unique combinations
    unique_locations = df[['date_only', 'latitude', 'longitude']].drop_duplicates()
    
    def compute_sunrise_sunset(row):
        if pd.isnull(row['date_only']):
            return pd.Series({'sunrise': pd.NaT, 'sunset': pd.NaT})
        location = LocationInfo(
            name="Custom",
            region="Custom",
            timezone="Europe/Paris",
            latitude=row['latitude'],
            longitude=row['longitude']
        )
        date_naive = row['date_only']
        tz = pytz.timezone(location.timezone)
        try:
            s = sun(location.observer, date=date_naive, tzinfo=tz)
            sunrise = s['sunrise']
            sunset = s['sunset']
        except Exception as e:
            # Handle exceptions from the sun function
            sunrise = pd.NaT
            sunset = pd.NaT
        return pd.Series({'sunrise': sunrise, 'sunset': sunset})
    
    # Apply the function to unique combinations
    unique_locations[['sunrise', 'sunset']] = unique_locations.apply(
        compute_sunrise_sunset, axis=1
    )
    
    # Merge back to original DataFrame
    df = df.merge(unique_locations, on=['date_only', 'latitude', 'longitude'], how='left')
    
    # Drop rows with missing sunrise/sunset times
    df = df.dropna(subset=['sunrise', 'sunset'])
    
    # Check if the timestamp is between sunrise and sunset
    df['is_sun_up'] = (df['date'] >= df['sunrise']) & (df['date'] <= df['sunset'])
    
    # Drop temporary columns if desired
    df = df.drop(columns=['date_only', 'sunrise', 'sunset'])
    
    return df


# Import the bank holidays in France for 2020 and 2021
holidays_2020 = JoursFeries.for_year(2020)
holidays_2021 = JoursFeries.for_year(2021)

# Create lists of dates from each dictionary
dates_2020 = list(holidays_2020.values())
dates_2021 = list(holidays_2021.values())

# Create DataFrame with all dates
all_dates = dates_2020 + dates_2021
bank_holidays_df = pd.DataFrame(all_dates, columns=["date"])
bank_holidays_df["date"] = pd.to_datetime(bank_holidays_df["date"])

# Add a new column "is_bank_holiday" to the data dataframe
def is_holidays(df):
    df["is_bank_holiday"] = df["date"].dt.date.isin(bank_holidays_df["date"].dt.date).astype(int)
    
    return df