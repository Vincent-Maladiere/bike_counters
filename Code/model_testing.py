# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from vacances_scolaires_france import SchoolHolidayDates
from jours_feries_france import JoursFeries
from code2 import date_encoder, prepare_data, build_pipeline, tune_hyperparameters, evaluate_model, train_model, test_model_kaggle, fit_encoder, encoder

# %% [markdown]
# #### Load data

# %%
df_train = pd.read_parquet("/Users/solalzana/Desktop/X/Python for Data Science/Final Project/bike_counters/data/train.parquet")
df_test = pd.read_parquet("/Users/solalzana/Desktop/X/Python for Data Science/Final Project/bike_counters/data/final_test.parquet")
df_ext = pd.read_csv("/Users/solalzana/Desktop/X/Python for Data Science/Final Project/bike_counters/data/external_data.csv")

# %% [markdown]
# #### prepare data

# %%
df_train_cleaned = prepare_data(df_train, df_ext)
df_test_cleaned = prepare_data(df_test, df_ext)

# %% [markdown]
# #### train/test split

# %%
X_train = df_train_cleaned.drop(columns=["log_bike_count", "bike_count"])
X_test = df_test_cleaned
y_train = df_train_cleaned['log_bike_count']

# %%
# X_train, X_test, y_train, y_test = train_test_split(df_train_cleaned.drop(columns=["log_bike_count", "bike_count"]), df_train_cleaned['log_bike_count'], test_size=0.2, random_state=42)

# %%
# from sklearn.decomposition import PCA

# pca = PCA()
# pca.fit(df_ext)

# # Calculate the cumulative explained variance ratio
# cumulative_explained_variance = pca.explained_variance_ratio_.cumsum()

# # Plot the cumulative explained variance ratio
# plt.figure(figsize=(10, 6))
# plt.plot(
#     range(1, len(cumulative_explained_variance) + 1),
#     cumulative_explained_variance,
#     marker="o",
#     linestyle="--",
#     color="b",
# )
# plt.title("Explained Variance depending on the number of parameters")
# plt.xlabel("Number of Parameters")
# plt.ylabel(" Explained Variance")
# plt.grid(True)
# plt.show()

# %% [markdown]
# #### build pipeline

# %%
from sklearn.pipeline import make_pipeline
import xgboost as xgb

fit_encoder(X_train)
# X_train = encoder(X_train)
# X_test = encoder(X_test)

model = xgb.XGBRegressor(objective='reg:squarederror')
trained_model = train_model(X_train, y_train, model)

test_model_kaggle(trained_model, X_test, "xgb") # results is a df storing y_pred(s)
# check submission folder now
# X_test.drop(columns=['date'], inplace=True)
# evaluate_model(trained_model, X_test, y_test)

# %%
from catboost import CatBoostRegressor

model = CatBoostRegressor()
pipeline_cb = build_pipeline(X_train, y_train, model)
trained_model_cb = train_model(pipeline_cb, model, X_train, y_train)

# test_model_kaggle(pipeline_cb, X_test, "cb") # results is a df storing y_pred(s)
# # check submission folder now
test_model_kaggle(model, X_test, y_test)

# %%
# show all rows
X_train[X_train["quarantine1"]==1]

# %%
# lightgbm
!pip install lightgbm
from lightgbm import LGBMRegressor
import lightgbm as lgb

model = lgb.LGBMRegressor()
pipeline_lgb = build_pipeline(X_train, y_train, model)
trained_model_lgb = train_model(pipeline_lgb, model, X_train, y_train)

test_model_kaggle(pipeline_lgb, X_test, "lgb") # results is a df storing y_pred(s)
# check submission folder

# %% [markdown]
# ### RF

# %%
# random forest
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_jobs=-1)
pipeline_cb = build_pipeline(X_train, y_train, model)
trained_model_cb = train_model(pipeline_cb, model, X_train, y_train)

test_model_kaggle(pipeline_xgb, X_test, "rf") # results is a df storing y_pred(s)
# check submission folder now

# %% [markdown]
# ####  tune hyperPs

# %%
pipeline_best_model = tune_hyperparameters(pipeline_xgb, X_train, y_train).best_estimator_

# %%
fitted_best = pipeline_best_model.fit(X_train, y_train)
y_pred = test_model_kaggle(fitted_best, X_test, "xgb")

# %%
model = xgb.XGBRegressor(objective='reg:squarederror')
y_pred_best = train_model(pipeline_best_model, "xgb", X_train, y_train) # y predictions to be submitted to Kaggle
results = test_model_kaggle(pipeline_best_model, X_test, "xgb") # results is a df storing y_pred(s)

# %%
pipeline_xgb_best = build_pipeline(X_train, y_train, model)

# %% [markdown]
# #### evaluate models

# %%
evaluate_model(best_model, X_test, y_test) # returns rmse with the best parameters
# function when train/test splitting on the train set only, not the Kaggle test set


