import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def optimize_xgboost(X_train, X_test, y_train, y_test, n_trials=30):
    """Optuna ile XGBoost hiperparametre optimizasyonu."""
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_jobs": -1,
            "random_state": 42
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        preds = model.predict(X_test)
        return np.sqrt(mean_squared_error(y_test, preds))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
