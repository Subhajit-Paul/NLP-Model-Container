import joblib
from fastapi import FastAPI

model = joblib.load(open("model_14.sav", 'rb'))
feature = joblib.load(open("feature.pkl", 'rb'))

app = FastAPI()

@app.post("/req")
async def login(data: str):
    results = model.predict(feature.transform([data.lower()]))
    return {"predictions": f"{results[0]}"}
    return data




