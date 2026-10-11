import shap

def explain_model(model, X_test):
    """SHAP kullanarak modelin tahmin nedenlerini açıklar (Kara Kutuyu açar)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    print("SHAP Summary Plot oluşturuluyor...")
    shap.summary_plot(shap_values, X_test)
    return explainer, shap_values
