import cv2
import numpy as np

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Görüntüleri CNN (Evrişimli Sinir Ağları) modelleri için hazırlar."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Görüntü okunamadı.")
    
    # Renk uzayını düzelt ve boyutlandır
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, target_size)
    
    # Pikselleri 0-1 arasına normalize et
    img_normalized = img_resized / 255.0
    return img_normalized
