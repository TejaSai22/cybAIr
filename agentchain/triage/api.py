from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Dict
from agentchain.triage.model import triage_threat
from agentchain.common.pipeline import pipeline

router = APIRouter()

class ClassificationData(BaseModel):
    payload: Dict[str, Any]

@router.post("/triage")
def triage_threat_endpoint(data: ClassificationData):
    result = triage_threat(data.payload)
    
    # Send triage result to mitigation agent
    pipeline.send_triage_event(result)
    print(f"⚖️ Threat triaged! Sending to mitigation agent...")
    
    return result 