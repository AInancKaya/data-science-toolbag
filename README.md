# Data Science Toolbag 🧰

Bu depo, veri bilimi, makine öğrenmesi ve derin öğrenme projelerinde kullanılmak üzere modüler olarak hazırlanmış kod şablonlarını (snippet/template) içerir. 

Apple Silicon (M2 vb.) mimarisinin donanım hızlandırma yeteneklerinden (`mps`, `n_jobs=-1`) tam verim alacak şekilde optimize edilmiştir.

Colab Kullanıcıları İçin Ekstra Hızlandırma İpucu:
Colab'ın ücretsiz sunduğu Nvidia GPU gücünden tam anlamıyla yararlanmak için, sınıflandırma ve regresyon şablonlarında modelleri eğitirken GPU parametrelerini açabilirsin. Ağaç tabanlı modeller Colab'da bu sayede uçuşa geçer:
XGBoost için: XGBClassifier(tree_method='gpu_hist', random_state=42)
CatBoost için: CatBoostClassifier(task_type='GPU', random_state=42)
LightGBM için: LGBMClassifier(device='gpu', random_state=42)

## 📂 İçerik
1. Veri Ön İşleme (Eksik veri doldurma, encoding)
2. Sınıflandırma ve Regresyon (XGBoost, LightGBM, CatBoost)
3. Dengesiz Veri Çözümleri (SMOTE)
4. Açıklanabilir Yapay Zeka (SHAP)
5. Hiperparametre Optimizasyonu (Optuna)
6. PyTorch ile Derin Öğrenme
7. Büyük Veri (Polars) & Canlıya Alma (FastAPI)

## 🚀 Nasıl Kullanılır?
Bu repo bir Python paketinden ziyade bir **şablon deposudur**. 
Yeni bir probleme/projeye başlarken, ihtiyacınız olan `.py` dosyasını buradan kopyalayıp aktif çalışma dizininize yapıştırın ve projenizde doğrudan import edin.

```python

### Peki Bu Dosyaları Pratikte Nasıl Kullanacaksın?

Bu klasör yapısını (isimleri `01_...`, `02_...` diye numaralı olan yapı) bir Python kütüphanesi gibi yüklemeye çalışmak yerine **fiziksel bir alet çantası** gibi kullanmalısın. Çünkü Python'da isimleri rakamla başlayan klasörlerden `import` komutu ile dosya çekmek zordur (Syntax Error verir).

**İzleyeceğin iş akışı (Workflow) şu şekilde olmalı:**

1. VS Code'da yepyeni bir proje açtın (örneğin: `_analizi`).
2. Hemen GitHub'daki veya bilgisayarındaki bu "Data Science Toolbag" klasörüne gidiyorsun.
3. Sana veri temizleme ve sınıflandırma modeli mi lazım? `preprocessing_utils.py`, `classification_models.py` ve `evaluation_metrics.py` dosyalarını kopyalayıp `_analizi` klasörünün içine yapıştırıyorsun.
4. Sonra kendi yazdığın ana kod dosyasında (örneğin `main.py` veya `notebook.ipynb`) şunları yazıyorsun:

```python
import pandas as pd
from preprocessing_utils import handle_missing_values, encode_features
from classification_models import train_classifiers

# 1. Veriyi çek
df = pd.read_csv("musteri_verisi.csv")

# 2. Hazır şablonunla veriyi temizle
df_clean = handle_missing_values(df)
df_hazir = encode_features(df_clean, target_col="churn")

# 3. Hazır şablonunla X ve y'yi ayırıp 4 farklı modeli aynı anda eğit
X = df_hazir.drop("churn", axis=1)
y = df_hazir["churn"]

# XGBoost, LightGBM, CatBoost vs hepsi arka planda eğitilip sonuçları ekrana basılacak
modeller = train_classifiers(X_train, X_test, y_train, y_test)