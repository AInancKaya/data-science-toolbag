import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def handle_missing_values(df, strategy="median"):
    """Eksik verileri doldurur."""
    df_clean = df.copy()
    num_cols = df_clean.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df_clean.select_dtypes(include=["object", "category"]).columns

    if len(num_cols) > 0:
        df_clean[num_cols] = SimpleImputer(strategy=strategy).fit_transform(df_clean[num_cols])
    if len(cat_cols) > 0:
        df_clean[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df_clean[cat_cols])
    return df_clean

def encode_features(df, target_col=None):
    """Kategorik değişkenleri dönüştürür."""
    df_encoded = df.copy()
    cat_cols = [col for col in df_encoded.columns if df_encoded[col].dtype == "object" and col != target_col]
    
    for col in cat_cols:
        if df_encoded[col].nunique() <= 2:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])
        else:
            df_encoded = pd.get_dummies(df_encoded, columns=[col], drop_first=True)
    return df_encoded
