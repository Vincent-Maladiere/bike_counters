# %% [markdown]
# # EDA and Feature Engineering

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import branca.colormap as cm
# Rename to eda_iriginal_dataset c'est pas le fichier des modèles cleui là

# %% [markdown]
# # I. EDA of Original Dataset

# %%
data = pd.read_parquet("/Users/solalzana/Desktop/X/Python for Data Science/Final Project/bike_counters/data/train.parquet")
data.head()

# %%
data.nunique(axis=0)

# %%
# Check for missing values
data.isnull().sum() # no missing values

# %%
data['log_bike_count'<0].sum()

# %%
# def date_encoder(X, col="date"):
#     X = X.copy()  # modify a copy of X
#     X[col] = pd.to_datetime(X[col])

#     # Encode the date information from the date column
#     X["year"] = X[col].dt.year
#     X["quarter"] = X[col].dt.quarter
#     X["month"] = X[col].dt.month
#     X["day"] = X[col].dt.day
#     X["weekday"] = X[col].dt.weekday + 1
#     X["hour"] = X[col].dt.hour

#     # Once we did the encoding, we can drop the original column
#     X = X.drop(col, axis=1)
#     return X

# date_encoder(data, col="date")
# # data.set_index(data["date"], inplace=True)

# # Also drop redundant columns
# data.drop(columns=["counter_id", "site_id", "counter_installation_date", "counter_technical_id", "coordinates"], axis=1, inplace=True)
# data

# %% [markdown]
# Now, let's plot the data to have a general idea of its distribution.

# %%
# target variable distribution
sns.histplot(data['log_bike_count'], kde=True) # using the log as bike_count is skewed
plt.title('Distribution of Log Bike Count')
plt.show()

# %%
# # Daily average log bike count
# data['log_bike_count'].resample('D').mean().plot(title='Daily Average Log Bike Count')
# plt.show()

# %% [markdown]
# Kind of normally distrbuted...

# %%
# Find the top 10 counters by total bike count
top_counters = data.groupby("counter_name", observed = True)["bike_count"].sum().nlargest(10).index

# Create a mask for these top counters
mask = data["counter_name"].isin(top_counters)

# Group by week and plot
data[mask].groupby(["counter_name", pd.Grouper(freq="1w", key="date")], observed = True)["bike_count"].sum().unstack(0).plot(figsize = (12,6), ylabel='log_bike_count')
plt.legend(title='Counter Name', bbox_to_anchor=(1, 1), loc="upper right")

# %% [markdown]
# At first sight, we can see a correlation between the usage and time of the year (more specifically less use during winter and more during summer). We can also see that some counters are consistently used more than others, maybe because of their location (more traffic in some zones of Paris, at the center maybe?), we will check that later.

# %%
(
    data.groupby(["site_name", "counter_name"])["bike_count"].sum()
    .sort_values(ascending=False)
    .head(10)
    .to_frame()
)

# %% [markdown]
# The most used counter (73 boulevard de Sébastopol) is very close to the center of Paris. Maybe there is more traffic at the center of Paris than at the extremities, let's check that.

# %% [markdown]
# # ÇA MARCHE PREQUE DISTANCE DU CENTRE!

# %%
# import folium
# from folium.plugins import MarkerCluster
# import branca.colormap as cm

# # Create a Folium map centered in Paris (we took online coordinates for 'center of Paris')
# m = folium.Map(location=[48.8566, 2.3522], zoom_start=12)

# # create colormap gradient base don bike count
# colormap = cm.LinearColormap(
#     colors=["green", "yellow", "red"],  # Gradient from green to red
#     vmin=data['bike_count'].min(),
#     vmax=data['bike_count'].max()
# )

# marker_cluster = MarkerCluster().add_to(m)

# # Add markers for each bike counter
# for latitude, longitude, bike_count, site_name in zip(data['latitude'], data['longitude'], data['bike_count'], data['site_name']):
#     folium.CircleMarker(
#         location=[latitude, longitude],
#         radius=10,
#         color=None,
#         fill=True,
#         fill_color=colormap(bike_count),
#         fill_opacity=0.8,
#         popup=folium.Popup(f"{site_name}<br>Bikes: {bike_count}", max_width=300)
#     ).add_to(marker_cluster)

# # colormap.caption = "Bike Counts"
# marker_cluster.save("bike_map.html")

# %%
data["date"].min(), data["date"].max()

# %% [markdown]
# We also notice that dates coincide with the covid era. Upon further research we see that there has been 2 quarantines so we implement them.

# %%
data["quarantine1"] = np.where((data['date'] >= '2020-10-30') & (data['date'] <= '2020-12-14'), 1, 0)
data["quarantine2"] = np.where((data['date'] >= '2020-04-03') & (data['date'] <= '2020-05-02'), 1, 0)

# %% [markdown]
# We are checking our weather/season hypothesis in the eda_external file where we explore the given dataset about meteorological conditions. We will merge both cleaned dataset for better understanding.

# %% [markdown]
# # II. ADDING ADDITIONAL DATASETS

# %% [markdown]
# ## 1. Given weather data

# %%
df_ext = pd.read_csv("/Users/solalzana/Desktop/X/Python for Data Science/Final Project/bike_counters/external_data/external_data.csv")
df_ext

# %% [markdown]
# #### Missing values

# %%
# Check for missing values
df_ext.isnull().sum()
# Better to check the proportion of missing values relative to the total nb of obs
df_ext.isnull().sum() / len(df_ext)

# %% [markdown]
# Many columns have a ton of missing values, for that let's forget all those who have more than 10% of their values missing as they are not very reliable.

# %%
df_ext.drop(columns=df_ext.columns[(df_ext.isnull().sum()/len(df_ext)) >= 0.1], inplace=True)

# %% [markdown]
# Now, we have to replace missing values from the remaining columlns as some models like Random Forest can't work with them. We will replace them with the median to avoid sensitivity to outliers as there are many outliers in this dataset.

# %%
for col in df_ext.columns:
    if col != 'date':  # Exclude the 'date' column
        df_ext[col] = df_ext[col].fillna(df_ext[col].median())

# %%
df_ext.isnull().sum()

# %%
df_ext.nunique(axis=0)

# %% [markdown]
# We see that there is numerous empty columns, and that `numer_sta`, `tminsol`, and `per` only have a unique value, so they are irrelevant. 

# %%
df_ext.drop(columns=df_ext.columns[df_ext.nunique()==0], inplace=True)
df_ext.drop(columns=["numer_sta", "per"], inplace=True)

# %%


# %% [markdown]
# #### Checking for multicolinearity

# %%
df_ext_wo_date = df_ext.drop(columns=['date'])
correlation_matrix = df_ext_wo_date.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.show

# %% [markdown]
# Filtering for variables that are highly correlated.

# %%
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix[correlation_matrix.abs()>=0.8], annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.grid(True)

# %% [markdown]
# `pmer` and `pres` have a correlation of 1: we can keep one of both without losing any variance.
# We can say the same for `raf10`and `rafper` and keep only windspeed `ff`. Following that logic we also drop `td` and `w2`.

# %%
df_ext.drop(columns=["pres", "raf10", "rafper", "td", "w2"], inplace=True)

# %% [markdown]
# #### Merging both cleaned datasets

# %%
# Convert both date columns to the same precision
data["date"] = data["date"].astype('datetime64[us]')
df_ext["date"] = df_ext["date"].astype('datetime64[us]')

# Merge both datasets
df_merged = pd.merge_asof(data.sort_values("date"), df_ext.sort_values("date"), on="date")
df_merged

# %% [markdown]
# ## 2. Additional Data: Vacances Scholaires et Jours Feriés

# %%
!pip install vacances-scolaires-france
!pip install jours_feries_france
# !pip install chantiers-a-paris

chantiers = pd.read_parquet("/Users/solalzana/Desktop/X/Python for Data Science/Final Project/bike_counters/data/chantiers-a-paris.parquet")

# %%
from vacances_scolaires_france import SchoolHolidayDates
from jours_feries_france import JoursFeries

school_holidays = SchoolHolidayDates()
jours_feries = JoursFeries()

df_merged['holidays'] = df_merged.apply(
    lambda row: 1 if school_holidays.is_holiday_for_zone(row["date"].date(), 'C') else 0, axis=1
)

df_merged['jour_ferie'] = df_merged.apply(
    lambda row: 1 if jours_feries.is_bank_holiday(row["date"].date()) else 0, axis=1
)

# %%
df_merged.drop(columns=["bike_count"], inplace=True)

# %%
train_cleaned = df_merged.to_parquet("/Users/solalzana/Desktop/X/Python for Data Science/Final Project/bike_counters/data/train_cleaned.parquet")

# %% [markdown]
# ## Checking relationship of variables with target

# %%
# col_list = df_merged.columns.tolist()

# for i in range(0, 40):
#     # Sélectionnez la colonne courante
#     x = df_merged.iloc[:, i]
    
#     # Sélectionnez la colonne "log_bike_count"
#     y = df_merged["log_bike_count"]
    
#     # Créez un DataFrame temporaire pour regrouper et calculer la moyenne
#     temp_df = pd.DataFrame({'x': x, 'y': y})
#     temp_df_grouped = temp_df.groupby('x').mean().reset_index()
    
#     # Créez un nuage de points avec les valeurs agrégées
#     plt.scatter(temp_df_grouped['x'], temp_df_grouped['y'])

#     # Ajoutez des labels et une légende
#     plt.figure(figsize=(4, 3))
#     plt.xlabel(col_list[i])
#     plt.ylabel('log_bike_count (Moyenne)')
#     plt.title(f'Relation entre {col_list[i]} et log_bike_count (Moyenne)')
 
#     # Affichez chaque sous-plot séparément
#     plt.show()

# %%
df_merged.shape[0]

# %%



