from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Dict
from agentchain.classification.model import classify_threat
from agentchain.common.pipeline import pipeline

router = APIRouter()

class ThreatData(BaseModel):
    payload: Dict[str, Any]

@router.post("/classify")
def classify_threat_endpoint(data: ThreatData):
    result = classify_threat(data.payload)
    
    # Send classification result to triage agent
    pipeline.send_classification_event(result)
    print(f"🔍 Threat classified! Sending to triage agent...")
    
    return result 