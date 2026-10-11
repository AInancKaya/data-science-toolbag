import numpy as np
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report)

def print_regression_metrics(y_true, y_pred, model_name="Model"):
    """Regresyon problemleri için hata metrikleri."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rse = sum((y_true - y_pred)**2) / sum((y_true - np.mean(y_true))**2)
    
    print(f"--- {model_name} Regresyon Sonuçları ---")
    print(f"RMSE : {rmse:.4f} | MSE: {mse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f} | RSE: {rse:.4f}")
    return {"RMSE": rmse, "MSE": mse, "MAE": mae, "R2": r2, "RSE": rse}

def print_classification_metrics(y_true, y_pred, y_prob=None, model_name="Model"):
    """Sınıflandırma problemleri için metrikler."""
    print(f"--- {model_name} Sınıflandırma Sonuçları ---")
    print(classification_report(y_true, y_pred))
    if y_prob is not None:
        print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}")
