from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Dict
from agentchain.mitigation.model import mitigate_threat, provide_feedback
from agentchain.common.pipeline import pipeline

router = APIRouter()

class TriageData(BaseModel):
    payload: Dict[str, Any]

class FeedbackData(BaseModel):
    threat_type: str
    severity: str
    action: str
    reward: float

@router.post("/mitigate")
def mitigate_threat_endpoint(data: TriageData):
    result = mitigate_threat(data.payload)
    
    # Send mitigation result to graph agent
    pipeline.send_mitigation_event(result)
    print(f"🛡️ Threat mitigated via RL! Action: {result['action']} (Severity: {result['severity']})")
    
    return result

@router.post("/feedback")
def mitigate_feedback_endpoint(data: FeedbackData):
    """
    Provide reward feedback to the RL Mitigation Agent.
    Reward should be positive for good actions and negative for bad actions.
    """
    provide_feedback(data.threat_type, data.severity, data.action, data.reward)
    print(f"🧠 RL Agent learned! Reward {data.reward} for {data.action} on {data.severity} {data.threat_type}")
    return {"status": "success", "message": "Feedback received and Q-table updated"}