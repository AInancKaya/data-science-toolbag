from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def build_stacking_model(X_train, y_train):
    """Güçlü modelleri (XGBoost, LightGBM) birleştirir (Stacking)."""
    base_models = [
        ("xgb", XGBClassifier(random_state=42, n_jobs=-1)),
        ("lgb", LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1))
    ]
    meta_model = LogisticRegression()
    stack_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model)
    stack_clf.fit(X_train, y_train)
    return stack_clf
