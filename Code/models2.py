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
from code2 import date_encoder, prepare_data, build_pipeline, tune_hyperparameters, evaluate_model, train_model

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

# %% [markdown]
# #### build pipeline

# %%
import xgboost as xgb

model = xgb.XGBRegressor(objective='reg:squarederror')
pipeline_xgb = build_pipeline(X_train, y_train, model)
y_pred = train_model(pipeline_xgb, "xgb", X_train, y_train, X_test)

# %% [markdown]
# ####  tune hyperPs

# %%
pipeline_best_model = tune_hyperparameters(pipeline_xgb, X_train, y_train).best_estimator_

# %%
model = xgb.XGBRegressor(objective='reg:squarederror')
y_pred_best = train_model(pipeline_best_model, "xgb", X_train, y_train, X_test) # y predictions to be submitted to Kaggle

# %%
pipeline_xgb_best = build_pipeline(X_train, y_train, model)
train_model(pipeline_xgb, best_model, X_train, y_train, X_test)

# %% [markdown]
# #### evaluate models

# %%
evaluate_model(best_model, X_test, y_test) # returns rmse with the best parameters

# %%



