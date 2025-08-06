from agentchain.ollama_utils import ollama_client
from agentchain.common.config import settings
import os

# Simple threat classification using Ollama
def classify_threat(payload: dict):
    try:
        # Create a prompt for threat classification
        prompt = f"""
        Analyze this cybersecurity threat data and classify it:
        
        Threat Data: {payload}
        
        Please classify this threat into one of these categories:
        - DDoS Attack
        - SQL Injection
        - Phishing
        - Malware
        - Port Scan
        - Brute Force
        - Unknown
        
        Provide your classification and a brief explanation.
        """
        
        # Use Ollama to generate classification
        result = ollama_client.generate_text(prompt)
        
        return {
            "threat_type": "classified",
            "llm_result": result,
            "provider": "ollama",
            "model": settings.ollama_model
        }
    except Exception as e:
        return {
            "threat_type": "unknown", 
            "error": str(e),
            "provider": "ollama"
        }

# This function is now replaced by the one above 