"""
RAG Knowledge Base - Threat Intelligence Storage and Retrieval
Supports both FAISS and Pinecone for vector storage
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import numpy as np

from langchain_community.vectorstores import FAISS, Pinecone
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from agentchain.common.config import settings
from agentchain.ollama_utils import ollama_client

logger = logging.getLogger(__name__)

class ThreatKnowledgeBase:
    """Comprehensive threat intelligence knowledge base with RAG capabilities"""
    
    def __init__(self, vector_store_type: str = "faiss"):
        """
        Initialize the knowledge base
        
        Args:
            vector_store_type: "faiss" or "pinecone"
        """
        self.vector_store_type = vector_store_type
        
        # Initialize embeddings based on provider
        if settings.llm_provider == "ollama":
            # Create a simple embedding wrapper for Ollama
            class OllamaEmbeddingWrapper:
                def __init__(self, client):
                    self.client = client
                
                def embed_query(self, text):
                    return self.client.get_embeddings(text)
                
                def embed_documents(self, texts):
                    return [self.client.get_embeddings(text) for text in texts]
            
            self.embeddings = OllamaEmbeddingWrapper(ollama_client)
        else:
            # Fallback to OpenAI if needed
            from langchain_community.embeddings import OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
            
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize vector store
        self._initialize_vector_store()
        
        # Threat intelligence data
        self.threat_indicators = []
        self.cve_database = {}
        self.attack_patterns = {}
        
    def _initialize_vector_store(self):
        """Initialize the vector store (FAISS or Pinecone)"""
        try:
            logger.info(f"Initializing vector store type: {self.vector_store_type}")
            if self.vector_store_type == "faiss":
                self._init_faiss()
            elif self.vector_store_type == "pinecone":
                try:
                    self._init_pinecone()
                except Exception as pinecone_error:
                    logger.warning(f"Pinecone initialization failed: {pinecone_error}")
                    logger.info("Falling back to FAISS vector store")
                    self.vector_store_type = "faiss"
                    self._init_faiss()
            else:
                raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            self.vector_store = None
    
    def _init_faiss(self):
        """Initialize FAISS vector store"""
        try:
            index_path = settings.faiss_index_path
            if os.path.exists(index_path):
                self.vector_store = FAISS.load_local(index_path, self.embeddings)
                logger.info(f"Loaded existing FAISS index from {index_path}")
            else:
                # Create empty FAISS index with dummy embeddings
                try:
                    self.vector_store = FAISS.from_texts(
                        ["Initial document"], 
                        self.embeddings
                    )
                    logger.info("Created new FAISS index")
                except Exception as embed_error:
                    logger.error(f"Failed to create FAISS with embeddings: {embed_error}")
                    # Create a minimal FAISS index without embeddings
                    self.vector_store = None
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            self.vector_store = None
    
    def _init_pinecone(self):
        """Initialize Pinecone vector store"""
        logger.info(f"Attempting to initialize Pinecone with env: {settings.pinecone_env}")
        if not settings.pinecone_api_key or not settings.pinecone_env:
            logger.warning("Pinecone API key or environment not configured")
            return
            
        try:
            import pinecone
            logger.info("Initializing Pinecone client...")
            pinecone.init(
                api_key=settings.pinecone_api_key,
                environment=settings.pinecone_env
            )
            
            index_name = "agentchain-threats"
            logger.info(f"Checking for index: {index_name}")
            if index_name not in pinecone.list_indexes():
                logger.info(f"Creating new Pinecone index: {index_name}")
                # Get embedding dimension from the model
                try:
                    # Test embedding to get dimension
                    test_embedding = self.embeddings.embed_query("test")
                    dimension = len(test_embedding)
                except:
                    # Fallback dimensions - use smaller size for compatibility
                    dimension = 768 if settings.llm_provider == "ollama" else 1536
                
                pinecone.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine"
                )
            
            logger.info("Connecting to Pinecone index...")
            self.vector_store = Pinecone.from_existing_index(
                index_name=index_name,
                embedding=self.embeddings,
                text_key="text"
            )
            logger.info("Successfully connected to Pinecone vector store")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    def add_threat_intelligence(self, threat_data: Dict[str, Any]):
        """Add threat intelligence to the knowledge base"""
        try:
            # Create document from threat data
            content = self._format_threat_content(threat_data)
            metadata = {
                "threat_type": threat_data.get("threat_type", "unknown"),
                "severity": threat_data.get("severity", "medium"),
                "timestamp": datetime.now().isoformat(),
                "source": threat_data.get("source", "manual"),
                "indicators": threat_data.get("indicators", []),
                "mitigation": threat_data.get("mitigation", [])
            }
            
            document = Document(page_content=content, metadata=metadata)
            
            # Split into chunks
            chunks = self.text_splitter.split_documents([document])
            
            # Add to vector store
            if self.vector_store:
                if isinstance(self.vector_store, FAISS):
                    self.vector_store.add_documents(chunks)
                else:  # Pinecone
                    self.vector_store.add_documents(chunks)
                
                # Save FAISS index
                if isinstance(self.vector_store, FAISS):
                    self.vector_store.save_local(settings.faiss_index_path)
                
                logger.info(f"Added threat intelligence: {threat_data.get('threat_type', 'unknown')}")
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
        
        # Basic threat information
        content_parts.append(f"Threat Type: {threat_data.get('threat_type', 'unknown')}")
        content_parts.append(f"Description: {threat_data.get('description', '')}")
        content_parts.append(f"Severity: {threat_data.get('severity', 'medium')}")
        
        # Indicators
        indicators = threat_data.get('indicators', [])
        if indicators:
            content_parts.append(f"Indicators: {', '.join(indicators)}")
        
        # Attack patterns
        patterns = threat_data.get('attack_patterns', [])
        if patterns:
            content_parts.append(f"Attack Patterns: {', '.join(patterns)}")
        
        # Mitigation strategies
        mitigation = threat_data.get('mitigation', [])
        if mitigation:
            content_parts.append(f"Mitigation: {', '.join(mitigation)}")
        
        # CVE information
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
            
            # Search vector store
            docs = self.vector_store.similarity_search(query, k=k)
            
            # Format results
            results = []
            for doc in docs:
                result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": 0.8  # Placeholder score
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
                "vector_store": f"{self.vector_store_type}_not_initialized",
                "embedding_model": settings.llm_provider
            }
            
            if isinstance(self.vector_store, FAISS):
                total_docs = len(self.vector_store.index_to_docstore_id)
            else:  # Pinecone
                # Get index stats from Pinecone
                total_docs = 1000  # Placeholder
            
            return {
                "total_documents": total_docs,
                "vector_store": self.vector_store_type,
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
        """Get most common threat types"""
        # This would typically query the vector store for frequency analysis
        # For now, return sample data
        return [
            {"threat_type": "DDoS Attack", "count": 45, "severity": "high"},
            {"threat_type": "SQL Injection", "count": 32, "severity": "high"},
            {"threat_type": "Phishing", "count": 28, "severity": "medium"},
            {"threat_type": "Malware", "count": 23, "severity": "high"},
            {"threat_type": "Port Scan", "count": 19, "severity": "low"}
        ]
    
    def get_threat_trends(self) -> Dict[str, Any]:
        """Get threat trends over time"""
        # This would analyze temporal patterns in the knowledge base
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

# Global knowledge base instance
# Use Pinecone if credentials are available, otherwise use FAISS
vector_store_type = "pinecone" if (settings.pinecone_api_key and settings.pinecone_env) else "faiss"
knowledge_base = ThreatKnowledgeBase(vector_store_type=vector_store_type) 