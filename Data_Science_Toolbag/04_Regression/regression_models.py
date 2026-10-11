from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

def train_regressors(X_train, X_test, y_train, y_test):
    """Popüler regresyon modellerini eğitir."""
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(random_state=42, n_jobs=-1, objective="reg:squarederror"),
        "LightGBM": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
        "CatBoost": CatBoostRegressor(random_state=42, verbose=0, thread_count=-1)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name} eğitimi tamamlandı.")
    return models
