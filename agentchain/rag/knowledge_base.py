"""
RAG Knowledge Base - Threat Intelligence Storage and Retrieval
Uses ChromaDB for vector storage
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import numpy as np

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from agentchain.common.config import settings
from agentchain.ollama_utils import ollama_client
import chromadb

logger = logging.getLogger(__name__)

class ThreatKnowledgeBase:
    """Comprehensive threat intelligence knowledge base with RAG capabilities"""
    
    def __init__(self, vector_store_type: str = "chroma"):
        """
        Initialize the knowledge base
        """
        self.vector_store_type = vector_store_type
        
        # Initialize embeddings based on provider
        if settings.llm_provider == "ollama":
            class OllamaEmbeddingWrapper:
                def __init__(self, client):
                    self.client = client
                
                def embed_query(self, text):
                    return self.client.get_embeddings(text)
                
                def embed_documents(self, texts):
                    return [self.client.get_embeddings(text) for text in texts]
            
            self.embeddings = OllamaEmbeddingWrapper(ollama_client)
        else:
            from langchain_community.embeddings import OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
            
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self._initialize_vector_store()
        
        self.threat_indicators = []
        self.cve_database = {}
        self.attack_patterns = {}
        
    def _initialize_vector_store(self):
        """Initialize ChromaDB vector store"""
        try:
            logger.info("Initializing ChromaDB vector store...")
            # Use local persistent directory in the project data dir
            persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")
            
            self.vector_store = Chroma(
                collection_name="agentchain-threats",
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )
            # Make sure it creates the directory structure
            self.vector_store.persist()
            logger.info(f"ChromaDB initialized at {persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.vector_store = None
    
    def add_threat_intelligence(self, threat_data: Dict[str, Any]):
        """Add threat intelligence to the knowledge base"""
        try:
            content = self._format_threat_content(threat_data)
            # Ensure metadata values are basic types (ChromaDB requirement)
            metadata = {
                "threat_type": str(threat_data.get("threat_type", "unknown")),
                "severity": str(threat_data.get("severity", "medium")),
                "timestamp": datetime.now().isoformat(),
                "source": str(threat_data.get("source", "manual")),
                "indicators": ", ".join(threat_data.get("indicators", [])),
                "mitigation": ", ".join(threat_data.get("mitigation", []))
            }
            
            document = Document(page_content=content, metadata=metadata)
            chunks = self.text_splitter.split_documents([document])
            
            if self.vector_store:
                self.vector_store.add_documents(chunks)
                self.vector_store.persist()
                logger.info(f"Added threat intelligence: {metadata['threat_type']}")
                return True
            else:
                logger.error("Vector store not initialized")
                return False
                
        except Exception as e:
            logger.error(f"Failed to add threat intelligence: {e}")
            return False
    
    def _format_threat_content(self, threat_data: Dict[str, Any]) -> str:
        """Format threat data into searchable content"""
        content_parts = []
        content_parts.append(f"Threat Type: {threat_data.get('threat_type', 'unknown')}")
        content_parts.append(f"Description: {threat_data.get('description', '')}")
        content_parts.append(f"Severity: {threat_data.get('severity', 'medium')}")
        
        indicators = threat_data.get('indicators', [])
        if indicators:
            content_parts.append(f"Indicators: {', '.join(indicators)}")
        
        patterns = threat_data.get('attack_patterns', [])
        if patterns:
            content_parts.append(f"Attack Patterns: {', '.join(patterns)}")
        
        mitigation = threat_data.get('mitigation', [])
        if mitigation:
            content_parts.append(f"Mitigation: {', '.join(mitigation)}")
        
        cve_info = threat_data.get('cve_info', {})
        if cve_info:
            content_parts.append(f"CVE: {cve_info.get('id', '')}")
            content_parts.append(f"CVE Description: {cve_info.get('description', '')}")
        
        return "\n".join(content_parts)
    
    def search_threats(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant threat intelligence"""
        try:
            if not self.vector_store:
                logger.error("Vector store not initialized")
                return []
            
            docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k)
            
            results = []
            for doc, score in docs_and_scores:
                result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": float(score)
                }
                results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"Failed to search threats: {e}")
            return []
    
    def get_threat_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        try:
            if not self.vector_store:
                return {
                    "total_documents": 0, 
                    "vector_store": "chroma_not_initialized",
                    "embedding_model": settings.llm_provider
                }
            
            # Chroma implementation
            collection = self.vector_store._collection
            total_docs = collection.count() if collection else 0
            
            return {
                "total_documents": total_docs,
                "vector_store": "chroma",
                "embedding_model": settings.llm_provider,
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}
    
    def add_cve_data(self, cve_data: Dict[str, Any]):
        """Add CVE information to the knowledge base"""
        cve_id = cve_data.get("id")
        if cve_id:
            self.cve_database[cve_id] = cve_data
            logger.info(f"Added CVE data: {cve_id}")
    
    def add_attack_pattern(self, pattern: Dict[str, Any]):
        """Add attack pattern to the knowledge base"""
        pattern_id = pattern.get("id")
        if pattern_id:
            self.attack_patterns[pattern_id] = pattern
            logger.info(f"Added attack pattern: {pattern_id}")
    
    def get_common_threats(self) -> List[Dict[str, Any]]:
        return [
            {"threat_type": "DDoS Attack", "count": 45, "severity": "high"},
            {"threat_type": "SQL Injection", "count": 32, "severity": "high"},
            {"threat_type": "Phishing", "count": 28, "severity": "medium"},
            {"threat_type": "Malware", "count": 23, "severity": "high"},
            {"threat_type": "Port Scan", "count": 19, "severity": "low"}
        ]
    
    def get_threat_trends(self) -> Dict[str, Any]:
        return {
            "trends": {
                "ddos_attacks": {"trend": "increasing", "change": "+15%"},
                "malware": {"trend": "stable", "change": "+2%"},
                "phishing": {"trend": "decreasing", "change": "-8%"}
            },
            "top_attack_vectors": [
                "Web Application Attacks",
                "Network-based Attacks", 
                "Social Engineering",
                "Malware Distribution"
            ]
        }

knowledge_base = ThreatKnowledgeBase(vector_store_type="chroma") 