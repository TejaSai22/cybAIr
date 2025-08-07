from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Dict
from agentchain.classification.model import classify_threat
from agentchain.common.pipeline import pipeline
import time
import random

router = APIRouter()

class ThreatData(BaseModel):
    payload: Dict[str, Any]

# Global statistics for tracking accuracy
classification_stats = {
    "total_classifications": 0,
    "successful_classifications": 0,
    "accuracy_percentage": 0.0,  # Start at 0% - only real results
    "last_updated": time.time(),
    "threat_types": {
        "DDoS Attack": 0,
        "SQL Injection": 0,
        "Phishing": 0,
        "Malware": 0,
        "Port Scan": 0,
        "Brute Force": 0,
        "Unknown": 0
    }
}

@router.post("/classify")
def classify_threat_endpoint(data: ThreatData):
    result = classify_threat(data.payload)
    
    # Update statistics
    classification_stats["total_classifications"] += 1
    
    # Simulate accuracy tracking (in real implementation, this would be based on feedback)
    if result.get("threat_type") == "classified":
        classification_stats["successful_classifications"] += 1
        
        # Update threat type counts
        llm_result = result.get("llm_result", "")
        for threat_type in classification_stats["threat_types"].keys():
            if threat_type.lower() in llm_result.lower():
                classification_stats["threat_types"][threat_type] += 1
                break
        else:
            classification_stats["threat_types"]["Unknown"] += 1
    
    # Calculate accuracy based on actual results
    if classification_stats["total_classifications"] > 0:
        classification_stats["accuracy_percentage"] = (classification_stats["successful_classifications"] / classification_stats["total_classifications"]) * 100
    
    classification_stats["last_updated"] = time.time()
    
    # Send classification result to triage agent
    pipeline.send_classification_event(result)
    print(f"🔍 Threat classified! Sending to triage agent...")
    
    return result

@router.get("/accuracy")
def get_classification_accuracy():
    """Get classification accuracy statistics"""
    return {
        "accuracy_percentage": round(classification_stats["accuracy_percentage"], 1),
        "total_classifications": classification_stats["total_classifications"],
        "successful_classifications": classification_stats["successful_classifications"],
        "threat_type_distribution": classification_stats["threat_types"],
        "last_updated": classification_stats["last_updated"],
        "status": "operational"
    } 