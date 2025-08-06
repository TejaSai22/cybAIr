"""
Simple Ollama utilities for AgentChain
"""
import requests
import json
import logging
from typing import List, Dict, Any, Optional
from agentchain.common.config import settings

logger = logging.getLogger(__name__)

class OllamaClient:
    """Simple Ollama client for embeddings and text generation"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url
        self.model = model
        
    def generate_text(self, prompt: str) -> str:
        """Generate text using Ollama"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            logger.error(f"Failed to generate text: {e}")
            # Return a fallback response instead of error
            return "Threat classification: Unknown - Unable to connect to Ollama server"
    
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for text using Ollama"""
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            # Return a dummy embedding of appropriate size
            # Use a smaller size to avoid memory issues
            return [0.0] * 768  # Smaller size for compatibility

# Global Ollama client
ollama_client = OllamaClient(
    base_url=settings.ollama_base_url,
    model=settings.ollama_model
) 