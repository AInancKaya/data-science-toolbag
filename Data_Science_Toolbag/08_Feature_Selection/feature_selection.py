from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier

def auto_feature_selection(X_train, y_train):
    """Gereksiz değişkenleri atarak modeli hızlandırır."""
    estimator = XGBClassifier(random_state=42, n_jobs=-1)
    selector = RFECV(estimator, step=1, cv=3, scoring="accuracy", n_jobs=-1)
    selector.fit(X_train, y_train)
    X_selected = X_train.loc[:, selector.support_]
    return selector, X_selected
