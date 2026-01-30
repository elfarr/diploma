from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict

from inference_pack.inference import ModelInference

app = FastAPI(title="Transplant Outcome API", version="1.0.0")
inf = ModelInference("inference_pack")

class PredictRequest(BaseModel):
    features: Dict[str, Any]

@app.post("/predict")
def predict(req: PredictRequest):
    return inf.predict(req.features)

@app.get("/health")
def health():
    return {"ok": True}
