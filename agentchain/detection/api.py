from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Dict, List
from agentchain.detection.model import detector
from agentchain.common.pipeline import pipeline
import numpy as np

router = APIRouter()

class NetworkData(BaseModel):
    payload: Dict[str, Any]

class TrainingData(BaseModel):
    samples: List[Dict[str, Any]]

@router.post("/detect")
def detect_anomaly(data: NetworkData):
    # Assume payload is a dict of features; convert to 2D array
    features = np.array([list(data.payload.values())]).astype(float)
    if not detector.is_trained():
        return {"error": "Anomaly detector model is not trained."}
    pred, score = detector.predict(features)
    result = {
        "anomaly": bool(pred[0] == -1),
        "score": float(score[0]),
        "features": data.payload
    }
    
    # If anomaly detected, send to classification agent
    if result["anomaly"]:
        pipeline.send_detection_event(result)
        print(f"🚨 Anomaly detected! Sending to classification agent...")
    
    return result

@router.post("/train")
def train_detector(data: TrainingData):
    X = np.array([list(sample.values()) for sample in data.samples]).astype(float)
    detector.fit(X)
    return {"status": "trained", "num_samples": len(data.samples)} 