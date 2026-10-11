from sklearn.feature_extraction.text import TfidfVectorizer
import re

def clean_text(text):
    """Metinleri temizler (özel karakterler, küçük harf vb.)"""
    text = re.sub(r"[^\w\s]", "", str(text).lower())
    return text

def vectorize_texts(train_texts, test_texts):
    """Metinleri makine öğrenmesi modelleri için sayılara (Vektöre) dönüştürür."""
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(train_texts)
    X_test_vec = vectorizer.transform(test_texts)
    return X_train_vec, X_test_vec, vectorizer
