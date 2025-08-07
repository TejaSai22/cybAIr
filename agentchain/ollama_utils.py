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
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to Ollama server at {self.base_url}")
            return "Threat classification: Unknown - Ollama server not available"
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            return "Threat classification: Unknown - Request timeout"
        except Exception as e:
            logger.error(f"Failed to generate text: {e}")
            return "Threat classification: Unknown - Unable to process request"
    
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
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to Ollama server at {self.base_url}")
            # Return a dummy embedding of appropriate size
            return [0.0] * 768  # Smaller size for compatibility
        except requests.exceptions.Timeout:
            logger.error("Ollama embeddings request timed out")
            return [0.0] * 768
        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            return [0.0] * 768

# Global Ollama client
ollama_client = OllamaClient(
    base_url=settings.ollama_base_url,
    model=settings.ollama_model
) 