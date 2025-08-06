from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Dict
from agentchain.mitigation.model import mitigate_threat
from agentchain.common.pipeline import pipeline

router = APIRouter()

class TriageData(BaseModel):
    payload: Dict[str, Any]

@router.post("/mitigate")
def mitigate_threat_endpoint(data: TriageData):
    result = mitigate_threat(data.payload)
    
    # Send mitigation result to graph agent
    pipeline.send_mitigation_event(result)
    print(f"🛡️ Threat mitigated! Sending to graph agent...")
    
    return result 