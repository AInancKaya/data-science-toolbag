from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI(title="Model API Servisi")

# Model yükleme örneği
# with open("model.pkl", "rb") as f:
#     model = pickle.load(f)

class Item(BaseModel):
    feature1: float
    feature2: float

@app.post("/predict")
def predict(item: Item):
    """API üzerinden gelen verilere tahmin üretir."""
    # data = pd.DataFrame([item.dict()])
    # prediction = model.predict(data)
    # return {"prediction": int(prediction[0])}
    return {"status": "Model API is running!"}
