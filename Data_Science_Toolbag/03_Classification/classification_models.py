from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

def train_classifiers(X_train, X_test, y_train, y_test):
    """Popüler sınıflandırma modellerini eğitir."""
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(random_state=42, n_jobs=-1, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
        "CatBoost": CatBoostClassifier(random_state=42, verbose=0, thread_count=-1)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name} eğitimi tamamlandı.")
    return models
