import optuna
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from model_catboost_meteo_covid import prepare_data

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 600),
        'learning_rate': trial.suggest_float('learning_rate', 0.10, 0.32),
        'depth': trial.suggest_int('depth', 7, 9),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 2e-8, 2e-5),
        'border_count': trial.suggest_int('border_count', 80, 100),
        'od_type': 'IncToDec',
        'od_wait': trial.suggest_int('od_wait', 20, 30),
    }

    scores = []
    tscv = TimeSeriesSplit(
        n_splits=3,  # 3 evaluation periods
        test_size=24*28,  # 4 weeks test size
        gap=24  # 1 day gap
    )

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_sample)):
        X_fold_train, X_fold_val = X_sample.iloc[train_idx], X_sample.iloc[val_idx]
        y_fold_train, y_fold_val = y_sample.iloc[train_idx], y_sample.iloc[val_idx]

        model = CatBoostRegressor(
            **params,
            random_seed=42,
            cat_features=['counter_name', 'weekday']
        )

        model.fit(
            X_fold_train, y_fold_train,
            eval_set=(X_fold_val, y_fold_val),
            early_stopping_rounds=50,
            verbose=False
        )

        pred = model.predict(X_fold_val)
        rmse = np.sqrt(np.mean((y_fold_val - pred) ** 2))
        scores.append(rmse)

        # Print info about each fold
        print(f"\nFold {fold + 1}:")
        print(f"Train size: {len(X_fold_train)}, Val size: {len(X_fold_val)}")
        print(f"RMSE: {rmse:.4f}")

    return np.mean(scores)

if __name__ == "__main__":
    # Load training data
    train_df = pd.read_parquet(Path("data/train.parquet"))
    weather_df = pd.read_csv('external_data/Weather/Propre_meteo.csv')
    covid_df = pd.read_csv('external_data/Covid/Propre_nbr_Covid.csv')

    # Prepare features
    X_train, y_train = prepare_data(train_df, weather_df, covid_df, is_train=True)

    # Sample data while maintaining temporal order
    N = 4  # adjust this to control sample size
    X_sample = X_train.iloc[::N].copy()
    y_sample = y_train.iloc[::N].copy()

    print(f"Original data size: {len(X_train)}")
    print(f"Sample data size: {len(X_sample)}")

    # Print TimeSeriesSplit details
    tscv = TimeSeriesSplit(n_splits=3, test_size=24*28, gap=24)
    print("\nTimeSeriesSplit configuration:")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_sample)):
        print(f"\nFold {fold + 1}")
        print(f"Training set: {len(train_idx)} samples")
        print(f"Validation set: {len(val_idx)} samples")
        print(f"Gap between sets: 24 hours")

    # Run optimization study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)

    print("\nBest parameters:", study.best_params)
    print("Best RMSE:", study.best_value)

    # Train final model on full dataset with best parameters
    final_model = CatBoostRegressor(
        **study.best_params,
        random_seed=42,
        cat_features=['counter_name', 'weekday']
    )
    final_model.fit(X_train, y_train, verbose=False)

    # Print feature importances
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 feature importances:")
    print(importances.head(10))

    # Print optimization history
    print("\nOptimization history:")
    best_scores = [study.trials[i].value for i in range(len(study.trials))]
    print("Trial\tRMSE")
    for i, score in enumerate(best_scores):
        print(f"{i+1}\t{score:.4f}")