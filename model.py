import joblib
from fastapi import FastAPI

model = joblib.load(open("model_14.sav", 'rb'))
feature = joblib.load(open("feature.pkl", 'rb'))

app = FastAPI()

label = {
    0:  "Admiration/Pride",
    1:  "Amusement",
    2:  "Love",
    3:  "Desire/Optimism",
    4:  "Caring",
    5:  "Gratitude/Relief",
    6:  "Approval/Realization",
    7:  "Surprise/Curiosity/Confusion",
    8:  "Fear/Nervousness",
    9:  "Remorse/Embarrassment",
    10: "Disappointment/Sadness/Grief",
    11: "Annoyance/Anger/Disgust",
    12: "Disapproval",
    13: "Exitement/Joy"
}

@app.post("/req")
async def login(data: str):
    results = model.predict(feature.transform([data.lower()]))
    return {"predictions": f"{label[results[0]]}"}
    return data




