import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

external_data = pd.read_csv(Path("external_data") / 'external_data.csv')
data = pd.read_parquet(Path("data") / "train.parquet")

external_data_cleaned = external_data.dropna(axis=1, how='all').sort_values(by=['counter_name', 'date'])


columns_of_interest = ["date", "etat_sol", "dd", "ff", "t", "u", "vv", "n", "ht_neige", "rr3"]

external_data_sorted = external_data_cleaned[columns_of_interest].copy()
external_data_sorted["date"] = pd.to_datetime(external_data_sorted['date'])

# Set temperature to celsius
external_data_sorted.loc[: ,"t"] = external_data_sorted["t"] - 273.15

# Set Lockdown dates
start_date_L1 = '2020-10-30'
end_date_L1 = '2020-12-15'

start_date_L2 = '2021-04-03'
end_date_L2 = '2021-05-04'

external_data_sorted.loc[:, 'Lockdown'] = (
    (external_data_sorted['date'] >= start_date_L1) & (external_data_sorted['date'] < end_date_L1) |
    (external_data_sorted['date'] >= start_date_L2) & (external_data_sorted['date'] < end_date_L2)
).astype(int)

# Set "soft-curfew" dates
start_date_SC1 = '2020-10-17'
end_date_SC1 = '2020-10-30'

start_date_SC2 = '2020-12-15'
end_date_SC2 = '2021-01-16'

start_date_SC3 = '2021-05-19'
end_date_SC3 = '2021-06-21'

external_data_sorted.loc[:, 'soft-curfew'] = (
    (external_data_sorted['date'] >= start_date_SC1) & (external_data_sorted['date'] < end_date_SC1) |
    (external_data_sorted['date'] >= start_date_SC2) & (external_data_sorted['date'] < end_date_SC2) |
    (external_data_sorted['date'] >= start_date_SC3) & (external_data_sorted['date'] < end_date_SC3)
).astype(int)

# Set "hard-curfew" dates
start_date_HC1 = '2021-01-16'
end_date_HC1 = '2021-04-03'

start_date_HC2 = '2021-05-04'
end_date_HC2 = '2021-05-19'

external_data_sorted.loc[:, 'hard-curfew'] = (
    (external_data_sorted['date'] >= start_date_HC1) & (external_data_sorted['date'] < end_date_HC1) |
    (external_data_sorted['date'] >= start_date_HC2) & (external_data_sorted['date'] < end_date_HC2)
).astype(int)

print(external_data_sorted.head())


# Let's fill missing hours : we will create new lines, which will be copies of the existing lines
def create_missing_hours(row):
    new_rows = []

    # copy = existing line minus two hours
    new_row_2h = row.copy()
    new_row_2h['date'] = row['date'] - pd.Timedelta(hours=2)
    new_rows.append(new_row_2h)

    # copy = existing line minus one hour
    new_row_1h = row.copy()
    new_row_1h['date'] = row['date'] - pd.Timedelta(hours=1)
    new_rows.append(new_row_1h)

    return new_rows

# Appliquer la fonction à chaque ligne du DataFrame
new_rows = []
for index, row in external_data_sorted.iterrows():
    new_rows.extend(create_missing_hours(row))

# Convertir la liste de nouvelles lignes en DataFrame
new_data = pd.DataFrame(new_rows)

# Concaténer les nouvelles lignes avec le DataFrame original
external_data_expanded = pd.concat([external_data_sorted, new_data], ignore_index=True)

# Trier le DataFrame par counter_name puis par date
external_data_expanded = external_data_expanded.sort_values(by=['counter_name','date']).reset_index(drop=True)

data_train = pd.merge(data, external_data_expanded, on='date', how='inner')


def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["weekday"] = X["date"].dt.weekday
    X["hour"] = X["date"].dt.hour

    return X.drop(columns=["date"])

external_data_train = _encode_dates(data_train)
external_data_train.head()