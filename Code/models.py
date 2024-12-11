# %% [markdown]
# # Model Testing and Hyperparameter Tuning

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

import import_ipynb

# %%
df_train = pd.read_parquet("/Users/solalzana/Desktop/X/Python for Data Science/Final Project/bike_counters/data/train.parquet")
df_test = pd.read_parquet("/Users/solalzana/Desktop/X/Python for Data Science/Final Project/bike_counters/data/final_test.parquet")

# %% [markdown]
# # I. Model testing

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from vacances_scolaires_france import SchoolHolidayDates
from jours_feries_france import JoursFeries

# def get_estimator():

#     # Preprocessing for numerical data
#     numerical_transformer = StandardScaler()

#     # Preprocessing for categorical data
#     categorical_transformer = OneHotEncoder(handle_unknown='ignore')

#     # Bundle preprocessing for numerical and categorical data
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', numerical_transformer, make_column_selector(dtype_include=np.number)),
#             ('cat', categorical_transformer, make_column_selector(dtype_include=object))
#         ])

#     # Define model
#     model = RandomForestRegressor(n_estimators=100, random_state=0)

#     # Create and evaluate the pipeline
#     pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                                ('model', model)
#                               ])
    
#     return pipeline 

def prepare_test_set(X):
    X = X.copy()
    X.drop(columns=["counter_name", "site_name", "site_id", "coordinates", "counter_technical_id"], axis=1, inplace=True)
    df_ext = pd.read_csv("/Users/solalzana/Desktop/X/Python for Data Science/Final Project/bike_counters/external_data/external_data.csv")
    # Drop columns with more than 10% of missing values
    df_ext.drop(columns=df_ext.columns[(df_ext.isnull().sum()/len(df_ext)) >= 0.1], inplace=True)
    # Replace remaining by median
    df_ext.fillna(df_ext.median(numeric_only=True), inplace=True)

    # Drop columns with 0 or 1 unique value
    df_ext.drop(columns=df_ext.columns[df_ext.nunique()==0], inplace=True)
    df_ext.drop(columns=df_ext.columns[df_ext.nunique()==1], inplace=True)

    # Drop highly correlated columns (correlation  > 0.9)
    df_ext.drop(columns=["pres", "raf10", "rafper", "td", "w2"], inplace=True)

    # Convert both date columns to the same precision
    X["date"] = X["date"].astype('datetime64[us]')
    df_ext["date"] = df_ext["date"].astype('datetime64[us]')

    # add quarantine dates
    X["quarantine1"] = np.where((X['date'] >= '2020-10-30') & (X['date'] <= '2020-12-14'), 1, 0)
    X["quarantine2"] = np.where((X['date'] >= '2020-04-03') & (X['date'] <= '2020-05-02'), 1, 0)
    # Merge both datasets
    X["orig_index"] = np.arange(X.shape[0])
    df_merged = pd.merge_asof(X.sort_values("date"), df_ext.sort_values("date"), on="date")
    # df_merged.sort_index(inplace=True)

    school_holidays = SchoolHolidayDates()
    jours_feries = JoursFeries()

    df_merged['holidays'] = df_merged.apply(
        lambda row: 1 if school_holidays.is_holiday_for_zone(row["date"].date(), 'C') else 0, axis=1
    )

    df_merged['jour_ferie'] = df_merged.apply(
        lambda row: 1 if jours_feries.is_bank_holiday(row["date"].date()) else 0, axis=1
    )

    return df_merged

# %%
def encoder(X):
    X = X.copy()  # modify a copy of X
    # Encode the date
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    #  drop the original date column
    X = X.drop(columns=["date"])

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),  # Scale numerical columns
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)  # Encode categorical columns
        ]
    )

    X_encoded = preprocessor.fit_transform(X)

    return X_encoded

# %%
X_train = prepare_test_set(df_train.drop(['log_bike_count', 'bike_count'],axis =1))
X_test = prepare_test_set(df_test)

y_train = df_train['log_bike_count']

# %%
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

def train_model(X_train, y_train, model):
    transformer = FunctionTransformer(encoder)
    pipeline = make_pipeline(transformer, model)
    pipeline.fit(X_train, y_train)
    
    return pipeline

# %%
from sklearn.linear_model import Ridge

trained_pipeline = train_model(X_train, y_train, Ridge())

y_test = trained_pipeline.predict(X_test)

# rmse = np.sqrt(mean_squared_error(y_train, y_test))

# print(f"The RMSE for a Ridge regressor is {rmse} ")

# %%
import xgboost as xgb

trained_pipeline = train_model(X_train, y_train, xgb.XGBRegressor(objective='reg:squarederror'))

y_test = trained_pipeline.predict(X_test)

# rmse = np.sqrt(mean_squared_error(y_train, y_test))

# print(f"The RMSE for a XGBoost Regressor is {rmse} ")

pd.DataFrame(y_test, columns=["log_bike_count"]).reset_index().rename(
    columns={"index": "Id"}
).to_csv("//Users/solalzana/Desktop/X/Python for Data Science/Final Project/bike_counters/Submissions/submission_xgb.csv", index=False)

# %%
from sklearn.ensemble import RandomForestRegressor

# model = RandomForestRegressor(n_jobs=-1, random_state=42, max_depth=20, n_estimators=20)

trained_pipeline = train_model(X_train, y_train, RandomForestRegressor(n_jobs=-1, max_depth=10, n_estimators=100))

y_pred = trained_pipeline.predict(X_test)

# rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# print(f"The RMSE for a Random Forest Regressor using the encoder function is {rmse} ")

pd.DataFrame(y_test, columns=["log_bike_count"]).reset_index().rename(
    columns={"index": "Id"}
).to_csv("//Users/solalzana/Desktop/X/Python for Data Science/Final Project/bike_counters/Submissions/submission_rf.csv", index=False)

# %%
from catboost import CatBoostRegressor

trained_pipeline = train_model(X_train, y_train, CatBoostRegressor())

y_test = trained_pipeline.predict(X_test)

# rmse = np.sqrt(mean_squared_error(y_train, y_test))

# print(f"The RMSE for a CatBoost regressor using the encoder function is {rmse} ")

pd.DataFrame(y_test, columns=["log_bike_count"]).reset_index().rename(
    columns={"index": "Id"}
).to_csv("//Users/solalzana/Desktop/X/Python for Data Science/Final Project/bike_counters/Submissions/submission_cb.csv", index=False)

# %% [markdown]
# Model with loest RMSE: ... . Now let's tune it.

# %% [markdown]
# # II. Hyperparameter Tuning

# %%
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor

# Define the hyperparameters you want to search over
param_grid = {
    'iterations': [1000],
    'learning_rate': [0.01],
    'depth': [6],
}

# Create a CatBoostRegressor instance
catboost_model = CatBoostRegressor()

# Create a pipeline with the encoder and grid search
transformer = FunctionTransformer(encoder)
pipeline = make_pipeline(transformer, GridSearchCV(catboost_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1))

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Get the best model from the grid search
best_model = pipeline.named_steps['gridsearchcv'].best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print the best hyperparameters and RMSE
print("Best hyperparameters:", pipeline.named_steps['gridsearchcv'].best_params_)
print("Root Mean Squared Error:", rmse)

# %%
pd.DataFrame(
    y_pred, columns=["log_bike_count"])
.reset_index()
.rename(columns={"index": "Id"})
.to_csv("/Users/solalzana/Downloads/pierre-bike_counters-main 2/predictions_Catboost_tuned.csv", index=False)


