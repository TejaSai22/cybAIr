"""
Advanced Threat Classification
Enhanced classification using ensemble methods and improved RAG
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
import json
import re
from datetime import datetime

# LangChain imports
try:
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAI
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available for advanced classification")

logger = logging.getLogger(__name__)

class AdvancedThreatClassifier:
    """Advanced threat classification using ensemble methods and improved RAG"""
    
    def __init__(self, model_type: str = "ensemble"):
        self.model_type = model_type
        self.models = {}
        self.vectorizers = {}
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained = False
        
        # LangChain components
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        
        # Model configurations
        self.model_configs = {
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            },
            "tfidf": {
                "max_features": 1000,
                "ngram_range": (1, 2),
                "stop_words": "english"
            }
        }
        
        # Threat categories
        self.threat_categories = [
            "brute_force", "port_scan", "ddos", "malware", "phishing",
            "sql_injection", "xss", "privilege_escalation", "data_exfiltration",
            "ransomware", "apt", "insider_threat", "unknown"
        ]
    
    def initialize_langchain(self, openai_api_key: str):
        """Initialize LangChain components"""
        if not LANGCHAIN_AVAILABLE:
            logger.error("LangChain not available")
            return False
        
        try:
            self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
            self.llm = OpenAI(api_key=openai_api_key, temperature=0.1)
            logger.info("LangChain components initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing LangChain: {e}")
            return False
    
    def extract_text_features(self, threat_data: Dict[str, Any]) -> str:
        """Extract text features from threat data"""
        text_parts = []
        
        # Extract relevant text fields
        if "packets" in threat_data:
            for packet in threat_data["packets"]:
                text_parts.extend([
                    f"src_ip_{packet.get('src_ip', 'unknown')}",
                    f"dst_ip_{packet.get('dst_ip', 'unknown')}",
                    f"src_port_{packet.get('src_port', 0)}",
                    f"dst_port_{packet.get('dst_port', 0)}",
                    f"protocol_{packet.get('protocol', 0)}",
                    f"suspicious_{packet.get('suspicious', False)}"
                ])
        
        if "event" in threat_data:
            event = threat_data["event"]
            text_parts.extend([
                f"log_type_{event.get('log_type', 'unknown')}",
                f"threat_indicators_{'_'.join(event.get('threat_indicators', []))}",
                f"severity_{event.get('severity', 'unknown')}",
                f"raw_line_{event.get('raw_line', '')}"
            ])
        
        if "threat_indicators" in threat_data:
            text_parts.append(f"indicators_{'_'.join(threat_data['threat_indicators'])}")
        
        if "source_ips" in threat_data:
            text_parts.append(f"ips_{'_'.join(threat_data['source_ips'])}")
        
        return " ".join(text_parts)
    
    def extract_numerical_features(self, threat_data: Dict[str, Any]) -> List[float]:
        """Extract numerical features from threat data"""
        features = []
        
        # Packet-based features
        if "packets" in threat_data:
            packets = threat_data["packets"]
            features.extend([
                len(packets),  # Number of packets
                sum(1 for p in packets if p.get("suspicious", False)),  # Suspicious packets
                np.mean([p.get("length", 0) for p in packets]) if packets else 0,  # Avg packet length
                np.std([p.get("length", 0) for p in packets]) if packets else 0,  # Packet length std
                len(set(p.get("src_ip") for p in packets if p.get("src_ip"))),  # Unique source IPs
                len(set(p.get("dst_ip") for p in packets if p.get("dst_ip"))),  # Unique destination IPs
                len(set(p.get("dst_port") for p in packets if p.get("dst_port"))),  # Unique destination ports
            ])
        else:
            features.extend([0] * 7)  # Default values
        
        # Event-based features
        if "event" in threat_data:
            event = threat_data["event"]
            features.extend([
                len(event.get("threat_indicators", [])),  # Number of threat indicators
                1 if event.get("severity") == "high" else 0,  # High severity flag
                1 if event.get("severity") == "medium" else 0,  # Medium severity flag
                len(event.get("source_ips", [])),  # Number of source IPs
            ])
        else:
            features.extend([0] * 4)
        
        # General features
        features.extend([
            len(threat_data.get("threat_indicators", [])),  # General threat indicators
            1 if threat_data.get("severity") == "high" else 0,  # General severity
        ])
        
        return features
    
    def create_training_data(self, threat_samples: List[Dict[str, Any]]) -> Tuple[List[str], List[List[float]], List[str]]:
        """Create training data from threat samples"""
        texts = []
        numerical_features = []
        labels = []
        
        for sample in threat_samples:
            # Extract features
            text = self.extract_text_features(sample)
            numerical = self.extract_numerical_features(sample)
            label = sample.get("threat_type", "unknown")
            
            texts.append(text)
            numerical_features.append(numerical)
            labels.append(label)
        
        return texts, numerical_features, labels
    
    def train(self, training_data: List[Dict[str, Any]]):
        """Train the advanced threat classifier"""
        logger.info(f"Training advanced threat classifier with {len(training_data)} samples")
        
        # Create training data
        texts, numerical_features, labels = self.create_training_data(training_data)
        
        if not texts or not numerical_features or not labels:
            logger.error("No valid training data")
            return False
        
        # Encode labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        self.label_encoders["threat_type"] = label_encoder
        
        # Train TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer(**self.model_configs["tfidf"])
        tfidf_features = tfidf_vectorizer.fit_transform(texts)
        self.vectorizers["tfidf"] = tfidf_vectorizer
        
        # Combine features
        numerical_array = np.array(numerical_features)
        combined_features = np.hstack([tfidf_features.toarray(), numerical_array])
        
        # Store feature names
        self.feature_names = (
            [f"tfidf_{i}" for i in range(tfidf_features.shape[1])] +
            [f"numerical_{i}" for i in range(numerical_array.shape[1])]
        )
        
        # Train models
        if self.model_type == "ensemble":
            self._train_ensemble_models(combined_features, encoded_labels)
        elif self.model_type == "random_forest":
            self._train_random_forest(combined_features, encoded_labels)
        else:
            logger.error(f"Unknown model type: {self.model_type}")
            return False
        
        self.is_trained = True
        logger.info("Advanced threat classifier training completed")
        return True
    
    def _train_ensemble_models(self, X: np.ndarray, y: np.ndarray):
        """Train ensemble of models"""
        # Random Forest
        rf = RandomForestClassifier(**self.model_configs["random_forest"])
        rf.fit(X, y)
        self.models["random_forest"] = rf
        
        # Add more models here if needed
        logger.info("Ensemble models trained successfully")
    
    def _train_random_forest(self, X: np.ndarray, y: np.ndarray):
        """Train Random Forest model"""
        rf = RandomForestClassifier(**self.model_configs["random_forest"])
        rf.fit(X, y)
        self.models["random_forest"] = rf
    
    def predict(self, threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict threat type using trained models"""
        if not self.is_trained:
            return {"threat_type": "unknown", "confidence": 0.0, "model_used": "none"}
        
        # Extract features
        text = self.extract_text_features(threat_data)
        numerical = self.extract_numerical_features(threat_data)
        
        # Transform features
        tfidf_features = self.vectorizers["tfidf"].transform([text])
        numerical_array = np.array([numerical])
        combined_features = np.hstack([tfidf_features.toarray(), numerical_array])
        
        # Get predictions from all models
        predictions = {}
        confidences = {}
        
        # Random Forest
        if "random_forest" in self.models:
            rf = self.models["random_forest"]
            rf_pred = rf.predict(combined_features)[0]
            rf_proba = rf.predict_proba(combined_features)[0]
            
            # Decode prediction
            threat_type = self.label_encoders["threat_type"].inverse_transform([rf_pred])[0]
            confidence = float(np.max(rf_proba))
            
            predictions["random_forest"] = threat_type
            confidences["random_forest"] = confidence
        
        # Ensemble decision
        if predictions:
            # Use the model with highest confidence
            best_model = max(confidences.keys(), key=lambda k: confidences[k])
            final_prediction = predictions[best_model]
            final_confidence = confidences[best_model]
        else:
            final_prediction = "unknown"
            final_confidence = 0.0
        
        # Enhanced classification with LLM if available
        llm_enhancement = {}
        if self.llm and self.vectorstore:
            llm_enhancement = self._enhance_with_llm(threat_data, final_prediction)
        
        return {
            "threat_type": final_prediction,
            "confidence": final_confidence,
            "model_used": best_model if predictions else "none",
            "model_predictions": predictions,
            "model_confidences": confidences,
            "llm_enhancement": llm_enhancement,
            "feature_count": len(self.feature_names)
        }
    
    def _enhance_with_llm(self, threat_data: Dict[str, Any], ml_prediction: str) -> Dict[str, Any]:
        """Enhance classification with LLM analysis"""
        try:
            # Create context from threat data
            context = self._create_llm_context(threat_data)
            
            # Create prompt
            prompt = f"""
            Analyze this cybersecurity threat and provide detailed classification:
            
            Threat Data: {context}
            ML Prediction: {ml_prediction}
            
            Please provide:
            1. Detailed threat classification
            2. Confidence level (0-1)
            3. Key indicators
            4. Recommended actions
            5. CVE references if applicable
            """
            
            # Get LLM response
            response = self.llm.invoke(prompt)
            
            return {
                "llm_analysis": response,
                "enhanced_classification": True,
                "context_used": context
            }
        except Exception as e:
            logger.error(f"Error in LLM enhancement: {e}")
            return {"error": str(e)}
    
    def _create_llm_context(self, threat_data: Dict[str, Any]) -> str:
        """Create context for LLM analysis"""
        context_parts = []
        
        if "packets" in threat_data:
            packets = threat_data["packets"]
            context_parts.append(f"Network packets: {len(packets)} packets analyzed")
            
            # Add suspicious packet info
            suspicious_count = sum(1 for p in packets if p.get("suspicious", False))
            if suspicious_count > 0:
                context_parts.append(f"Suspicious packets: {suspicious_count}")
        
        if "event" in threat_data:
            event = threat_data["event"]
            context_parts.append(f"Log event: {event.get('log_type', 'unknown')}")
            context_parts.append(f"Threat indicators: {', '.join(event.get('threat_indicators', []))}")
            context_parts.append(f"Severity: {event.get('severity', 'unknown')}")
        
        if "threat_indicators" in threat_data:
            context_parts.append(f"General indicators: {', '.join(threat_data['threat_indicators'])}")
        
        return "; ".join(context_parts)
    
    def build_knowledge_base(self, threat_documents: List[str]):
        """Build knowledge base for RAG"""
        if not LANGCHAIN_AVAILABLE or not self.embeddings:
            logger.error("LangChain not available for knowledge base")
            return False
        
        try:
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            texts = []
            for doc in threat_documents:
                texts.extend(text_splitter.split_text(doc))
            
            # Create vector store
            self.vectorstore = FAISS.from_texts(texts, self.embeddings)
            
            # Create QA chain
            prompt_template = """
            Use the following context to answer the question about cybersecurity threats.
            
            Context: {context}
            Question: {question}
            
            Answer based on the context provided:
            """
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(),
                chain_type_kwargs={"prompt": prompt}
            )
            
            logger.info(f"Knowledge base built with {len(texts)} text chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error building knowledge base: {e}")
            return False
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            logger.error("Cannot save untrained model")
            return False
        
        model_data = {
            "model_type": self.model_type,
            "models": self.models,
            "vectorizers": self.vectorizers,
            "label_encoders": self.label_encoders,
            "feature_names": self.feature_names,
            "model_configs": self.model_configs,
            "is_trained": self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
        return True
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        try:
            model_data = joblib.load(filepath)
            self.model_type = model_data["model_type"]
            self.models = model_data["models"]
            self.vectorizers = model_data["vectorizers"]
            self.label_encoders = model_data["label_encoders"]
            self.feature_names = model_data["feature_names"]
            self.model_configs = model_data["model_configs"]
            self.is_trained = model_data["is_trained"]
            
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model"""
        return {
            "model_type": self.model_type,
            "is_trained": self.is_trained,
            "feature_count": len(self.feature_names),
            "models_available": list(self.models.keys()),
            "vectorizers_available": list(self.vectorizers.keys()),
            "threat_categories": self.threat_categories,
            "langchain_available": LANGCHAIN_AVAILABLE
        } 