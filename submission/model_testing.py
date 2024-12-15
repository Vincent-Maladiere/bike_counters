# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# %%
df_train = pd.read_parquet('/Users/guenounraphael/Desktop/Cyclist project/Code/data_train_merged.parquet')
df_train = df_train.set_index('index')
df_test = pd.read_parquet('/Users/guenounraphael/Desktop/Cyclist project/Code/data_test_merged.parquet')
df_test = df_test.set_index('index')

# %% [markdown]
# Based on our previus explonatory data anylis, wez remove the predictors that are too correlated or that are redundant. A

# %%
df_train = df_train.drop(columns=["counter_id", "site_id", "bike_count", "counter_installation_date", "counter_technical_id", "latitude", "longitude","coordinates"], axis=1)


df_train

# %%
df_test = df_test.drop(columns=["counter_id", "site_id", "counter_installation_date", "counter_technical_id", "latitude", "longitude","coordinates"], axis=1)


df_test

# %%
df_train['quarantine1'] = df_train['quarantine1'].astype(bool)
df_train['quarantine2'] = df_train['quarantine2'].astype(bool)
df_train['christmas'] = df_train['christmas'].astype(bool)

df_test['quarantine1'] = df_test['quarantine1'].astype(bool)
df_test['quarantine2'] = df_test['quarantine2'].astype(bool)
df_test['christmas'] = df_test['christmas'].astype(bool)

# %%
train_numerical_columns = df_train.select_dtypes(include=['float', 'int']).drop(columns = ["log_bike_count"])
train_categorical_columns = df_train.select_dtypes(include=['object', 'category'])
train_bool_columns = df_train.select_dtypes(include=['bool'])

# %%
train_numerical_columns.columns

# %%
def encoder_for_dates(X):
    X = X.copy() 
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["weekday"] = X["date"].dt.weekday
    X["hour"] = X["date"].dt.hour
    X["weekend"] = X["weekday"] > 4
    return X.drop(columns=["date"])

encoder_for_dates(df_train)
   

# %%
encoder_for_dates(df_train).nunique()

# %%
def cyclical_encoding(X, column, max_val):
    X = X.copy()
    X[f"{column}_sin"] = np.sin(2 * np.pi * X[column] / max_val)
    X[f"{column}_cos"] = np.cos(2 * np.pi * X[column] / max_val)
    return X

# %%
def fit_encoder(X_train):
    global year_encoder, month_encoder, weekday_encoder, category_encoder, numerical_encoder
    
    X_train = encoder_for_dates(X_train)
    
    year_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(X_train[["year"]])
    month_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(X_train[["month"]])
    category_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(X_train[['counter_name', 'site_name']])
    weekday_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(X_train[["weekday"]])
    
    numerical_encoder = StandardScaler().fit(X_train[["cod_tend", "t", "u", "raf10", "etat_sol"]])

# %% [markdown]
# Since year, months, weekdays, and category have a quite low cardinality, it seems reasonable to use on hot encoding here. Moreover, based on the cyclical pattern of hour and day of these predictors, we also proceeded with cyclical encoding for them

# %%
def encoder(X):
    X_encoded = encoder_for_dates(X)

    X_encoded = cyclical_encoding(X_encoded, "hour", 24)
    X_encoded = cyclical_encoding(X_encoded, "day", 31)
    
    years_encoded = year_encoder.transform(X_encoded[["year"]])
    months_encoded = month_encoder.transform(X_encoded[["month"]])
    weekdays_encoded = weekday_encoder.transform(X_encoded[["weekday"]])
    categories_encoded = category_encoder.transform(X_encoded[["counter_name", "site_name"]])
    numerical_encoded = numerical_encoder.transform(X_encoded[["cod_tend", "t", "u", "raf10", "etat_sol"]])

    years_df = pd.DataFrame(years_encoded, columns=["2020", "2021"])
    months_df = pd.DataFrame(months_encoded, columns=["janv", "fev", "mars", "avril", "mai", "juin", "juillet", "aout", "sept", "octobre", "novem", "decembre"])
    weekdays_df = pd.DataFrame(weekdays_encoded, columns=[f"weekday_{i}" for i in range(weekdays_encoded.shape[1])])
    categories_df = pd.DataFrame(categories_encoded, columns=[f"cat_{i}" for i in range(categories_encoded.shape[1])])
    numerical_df = pd.DataFrame(numerical_encoded, columns=["cod_tend", "t", "u", "raf10", "etat_sol"])

    
    X_encoded.reset_index(drop=True, inplace=True)
    X_encoded = pd.concat([X_encoded, years_df, months_df, weekdays_df, categories_df, numerical_df], axis=1)

    X_encoded.drop(columns=["year", "month", "weekday", "day", "hour", "counter_name", "site_name", "cod_tend", "t", "u", "raf10", "etat_sol"], inplace=True)
    
    return X_encoded

# %%

X_train = df_train.drop(columns=["log_bike_count"])
y_train = df_train["log_bike_count"]

X_test = df_test


# %%
fit_encoder(X_train)
X_train_encoded = encoder(X_train)
X_test_encoded = encoder(X_test)

X_test_encoded

# %%

import xgboost as xgb

# %%

models = {
    "RandomForest": RandomForestRegressor(n_estimators=  2 , random_state=42),
    "Ridge": Ridge(),
    "XGBoost": xgb.XGBRegressor(verbosity=0, random_state=42),
    }

# %%

def train_and_predict_submission(X_train, y_train, X_test, model, model_name):
    transformer = FunctionTransformer(encoder)
    pipeline = make_pipeline(transformer, model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    submission_df = pd.DataFrame({
        "Id": range(len(y_pred)), 
        "log_bike_count": y_pred 
    })
    submission_df.to_csv(f"submission_{model_name}.csv", index=False)
    print(f"{model_name}: Submission file saved as submission_{model_name}.csv")

for model_name, model in models.items():
    print(f"Training and predicting with {model_name}...")
    train_and_predict_submission(X_train, y_train, X_train, model, model_name)




# %% [markdown]
# Based on our submission on the Kaggle (before tunning), the best model (The one with the smallest RMSE is XGtboost with a RMSE = 0.6921)

# %% [markdown]
# We are now going to proceed with a grid search to find and tune the best hyperparameters.

# %% [markdown]
# ## GRID SEARCH

# %%
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb


# Réduire la taille de l'échantillon (50% des données)
X_train_sampled, _, y_train_sampled, _ = train_test_split(
    X_train_encoded, y_train, test_size=0.5, random_state=42
)

param_grid = {
    'max_depth': [6],
    'learning_rate': [0.2,0.3],
    'n_estimators': [100, 200],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

xgb_model = xgb.XGBRegressor(verbosity=0, random_state=42)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',  
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train_sampled, y_train_sampled)

print("Meilleurs paramètres :", grid_search.best_params_)
print("Meilleur score (RMSE) :", -grid_search.best_score_)


# %%

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
import xgboost as xgb

# Use the best parameters found from GridSearchCV
best_params = {
    'colsample_bytree': 1.0,
    'learning_rate': 0.3,
    'max_depth': 6,
    'n_estimators': 200,
    'subsample': 0.8
}
best_xgb_model = xgb.XGBRegressor(**best_params, verbosity=0, random_state=42)

transformer = FunctionTransformer(encoder)
pipeline = make_pipeline(transformer, best_xgb_model)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

submission_df = pd.DataFrame({
    "Id": range(len(y_pred)), 
    "log_bike_count": y_pred 
})
submission_df.to_csv("final_submission.csv", index=False)

print("XGBoost: Submission file saved as final_submission.csv")





