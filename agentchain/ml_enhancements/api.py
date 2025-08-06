"""
Enhanced ML Models API
FastAPI endpoints for advanced ML models
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Any, Optional
import json
import logging
import random
from datetime import datetime

from agentchain.ml_enhancements.advanced_detection import AdvancedAnomalyDetector
from agentchain.ml_enhancements.advanced_classification import AdvancedThreatClassifier
from agentchain.ml_enhancements.advanced_rl import AdvancedRLMitigation

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Enhanced ML Models"])

# Global model instances
advanced_detector = None
advanced_classifier = None
advanced_rl = None

@router.post("/detection/initialize")
async def initialize_advanced_detection(model_type: str = "ensemble"):
    """Initialize advanced anomaly detection model"""
    global advanced_detector
    
    try:
        advanced_detector = AdvancedAnomalyDetector(model_type=model_type)
        return {
            "status": "success",
            "message": f"Advanced detection model initialized with type: {model_type}",
            "model_info": advanced_detector.get_model_info()
        }
    except Exception as e:
        logger.error(f"Error initializing advanced detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detection/train")
async def train_advanced_detection(training_data: List[Dict[str, Any]], 
                                 labels: Optional[List[int]] = None):
    """Train advanced anomaly detection model"""
    global advanced_detector
    
    if not advanced_detector:
        raise HTTPException(status_code=400, detail="Advanced detector not initialized")
    
    try:
        success = advanced_detector.train(training_data, labels)
        if success:
            return {
                "status": "success",
                "message": f"Advanced detection model trained with {len(training_data)} samples",
                "model_info": advanced_detector.get_model_info()
            }
        else:
            raise HTTPException(status_code=500, detail="Training failed")
    except Exception as e:
        logger.error(f"Error training advanced detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detection/predict")
async def predict_advanced_detection(packets: List[Dict[str, Any]]):
    """Predict anomalies using advanced detection model"""
    global advanced_detector
    
    if not advanced_detector:
        raise HTTPException(status_code=400, detail="Advanced detector not initialized")
    
    try:
        result = advanced_detector.predict(packets)
        return {
            "status": "success",
            "prediction": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in advanced detection prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/classification/initialize")
async def initialize_advanced_classification(model_type: str = "ensemble", 
                                           openai_api_key: Optional[str] = None):
    """Initialize advanced threat classification model"""
    global advanced_classifier
    
    try:
        advanced_classifier = AdvancedThreatClassifier(model_type=model_type)
        
        # Initialize LangChain if API key provided
        if openai_api_key:
            advanced_classifier.initialize_langchain(openai_api_key)
        
        return {
            "status": "success",
            "message": f"Advanced classifier initialized with type: {model_type}",
            "model_info": advanced_classifier.get_model_info()
        }
    except Exception as e:
        logger.error(f"Error initializing advanced classification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/classification/train")
async def train_advanced_classification(training_data: List[Dict[str, Any]]):
    """Train advanced threat classification model"""
    global advanced_classifier
    
    if not advanced_classifier:
        raise HTTPException(status_code=400, detail="Advanced classifier not initialized")
    
    try:
        success = advanced_classifier.train(training_data)
        if success:
            return {
                "status": "success",
                "message": f"Advanced classifier trained with {len(training_data)} samples",
                "model_info": advanced_classifier.get_model_info()
            }
        else:
            raise HTTPException(status_code=500, detail="Training failed")
    except Exception as e:
        logger.error(f"Error training advanced classification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/classification/predict")
async def predict_advanced_classification(threat_data: Dict[str, Any]):
    """Predict threat type using advanced classification model"""
    global advanced_classifier
    
    if not advanced_classifier:
        raise HTTPException(status_code=400, detail="Advanced classifier not initialized")
    
    try:
        result = advanced_classifier.predict(threat_data)
        return {
            "status": "success",
            "prediction": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in advanced classification prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/classification/build-knowledge-base")
async def build_knowledge_base(threat_documents: List[str]):
    """Build knowledge base for RAG-enhanced classification"""
    global advanced_classifier
    
    if not advanced_classifier:
        raise HTTPException(status_code=400, detail="Advanced classifier not initialized")
    
    try:
        success = advanced_classifier.build_knowledge_base(threat_documents)
        if success:
            return {
                "status": "success",
                "message": f"Knowledge base built with {len(threat_documents)} documents"
            }
        else:
            raise HTTPException(status_code=500, detail="Knowledge base building failed")
    except Exception as e:
        logger.error(f"Error building knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rl/initialize")
async def initialize_advanced_rl(agent_type: str = "q_learning", state_size: int = 10):
    """Initialize advanced RL mitigation system"""
    global advanced_rl
    
    try:
        advanced_rl = AdvancedRLMitigation(agent_type=agent_type)
        success = advanced_rl.initialize(state_size)
        
        if success:
            return {
                "status": "success",
                "message": f"Advanced RL system initialized with {agent_type} agent",
                "model_info": advanced_rl.get_model_info()
            }
        else:
            raise HTTPException(status_code=500, detail="RL initialization failed")
    except Exception as e:
        logger.error(f"Error initializing advanced RL: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rl/train")
async def train_advanced_rl(episodes: int = 1000):
    """Train advanced RL mitigation agent"""
    global advanced_rl
    
    if not advanced_rl:
        raise HTTPException(status_code=400, detail="Advanced RL system not initialized")
    
    try:
        success = advanced_rl.train(episodes)
        if success:
            return {
                "status": "success",
                "message": f"Advanced RL agent trained for {episodes} episodes",
                "model_info": advanced_rl.get_model_info()
            }
        else:
            raise HTTPException(status_code=500, detail="RL training failed")
    except Exception as e:
        logger.error(f"Error training advanced RL: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rl/predict")
async def predict_advanced_rl(threat_data: Dict[str, Any]):
    """Predict optimal action using advanced RL agent"""
    global advanced_rl
    
    if not advanced_rl:
        raise HTTPException(status_code=400, detail="Advanced RL system not initialized")
    
    try:
        result = advanced_rl.predict_action(threat_data)
        return {
            "status": "success",
            "prediction": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in advanced RL prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/save")
async def save_enhanced_models(base_path: str = "models"):
    """Save all enhanced ML models"""
    import os
    
    try:
        results = {}
        
        # Save detection model
        if advanced_detector and advanced_detector.is_trained:
            os.makedirs(base_path, exist_ok=True)
            detection_path = f"{base_path}/advanced_detection_model.pkl"
            success = advanced_detector.save_model(detection_path)
            results["detection"] = {"saved": success, "path": detection_path}
        
        # Save classification model
        if advanced_classifier and advanced_classifier.is_trained:
            os.makedirs(base_path, exist_ok=True)
            classification_path = f"{base_path}/advanced_classification_model.pkl"
            success = advanced_classifier.save_model(classification_path)
            results["classification"] = {"saved": success, "path": classification_path}
        
        # Save RL model
        if advanced_rl and advanced_rl.is_trained:
            os.makedirs(base_path, exist_ok=True)
            rl_path = f"{base_path}/advanced_rl_model.pkl"
            success = advanced_rl.save_model(rl_path)
            results["rl"] = {"saved": success, "path": rl_path}
        
        return {
            "status": "success",
            "message": "Enhanced models saved",
            "results": results
        }
    except Exception as e:
        logger.error(f"Error saving enhanced models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/load")
async def load_enhanced_models(base_path: str = "models"):
    """Load all enhanced ML models"""
    import os
    
    try:
        results = {}
        
        # Load detection model
        detection_path = f"{base_path}/advanced_detection_model.pkl"
        if os.path.exists(detection_path):
            global advanced_detector
            advanced_detector = AdvancedAnomalyDetector()
            success = advanced_detector.load_model(detection_path)
            results["detection"] = {"loaded": success, "path": detection_path}
        
        # Load classification model
        classification_path = f"{base_path}/advanced_classification_model.pkl"
        if os.path.exists(classification_path):
            global advanced_classifier
            advanced_classifier = AdvancedThreatClassifier()
            success = advanced_classifier.load_model(classification_path)
            results["classification"] = {"loaded": success, "path": classification_path}
        
        # Load RL model
        rl_path = f"{base_path}/advanced_rl_model.pkl"
        if os.path.exists(rl_path):
            global advanced_rl
            advanced_rl = AdvancedRLMitigation()
            success = advanced_rl.load_model(rl_path)
            results["rl"] = {"loaded": success, "path": rl_path}
        
        return {
            "status": "success",
            "message": "Enhanced models loaded",
            "results": results
        }
    except Exception as e:
        logger.error(f"Error loading enhanced models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_enhanced_ml_status():
    """Get status of all enhanced ML models"""
    status = {
        "detection": {
            "initialized": advanced_detector is not None,
            "trained": advanced_detector.is_trained if advanced_detector else False,
            "model_info": advanced_detector.get_model_info() if advanced_detector else None
        },
        "classification": {
            "initialized": advanced_classifier is not None,
            "trained": advanced_classifier.is_trained if advanced_classifier else False,
            "model_info": advanced_classifier.get_model_info() if advanced_classifier else None
        },
        "rl": {
            "initialized": advanced_rl is not None,
            "trained": advanced_rl.is_trained if advanced_rl else False,
            "model_info": advanced_rl.get_model_info() if advanced_rl else None
        }
    }
    
    return {
        "status": "success",
        "enhanced_ml_status": status,
        "timestamp": datetime.now().isoformat()
    }

@router.post("/demo/train-all")
async def demo_train_all_models():
    """Demo: Train all enhanced models with sample data"""
    try:
        # Generate sample training data
        sample_detection_data = []
        sample_classification_data = []
        
        # Sample detection data (network packets)
        for i in range(100):
            sample_detection_data.append({
                "length": random.randint(64, 1500),
                "src_port": random.randint(1024, 65535),
                "dst_port": random.randint(1, 65535),
                "protocol": random.choice([6, 17, 1]),  # TCP, UDP, ICMP
                "ttl": random.randint(32, 255),
                "window": random.randint(1024, 65535),
                "payload_size": random.randint(0, 1000),
                "suspicious": random.choice([True, False]),
                "timestamp": datetime.now().isoformat()
            })
        
        # Sample classification data (threat events)
        threat_types = ["brute_force", "port_scan", "ddos", "malware", "phishing"]
        for i in range(50):
            sample_classification_data.append({
                "threat_type": random.choice(threat_types),
                "packets": sample_detection_data[i*2:(i+1)*2],
                "threat_indicators": random.sample(["port_scan", "brute_force", "ddos"], random.randint(1, 3)),
                "severity": random.choice(["low", "medium", "high"]),
                "anomaly_score": random.uniform(0.1, 0.9),
                "confidence": random.uniform(0.5, 0.95)
            })
        
        # Initialize and train models
        results = {}
        
        # Detection model
        global advanced_detector
        advanced_detector = AdvancedAnomalyDetector("ensemble")
        detection_success = advanced_detector.train(sample_detection_data)
        results["detection"] = {"trained": detection_success, "samples": len(sample_detection_data)}
        
        # Classification model
        global advanced_classifier
        advanced_classifier = AdvancedThreatClassifier("ensemble")
        classification_success = advanced_classifier.train(sample_classification_data)
        results["classification"] = {"trained": classification_success, "samples": len(sample_classification_data)}
        
        # RL model
        global advanced_rl
        advanced_rl = AdvancedRLMitigation("q_learning")
        rl_init_success = advanced_rl.initialize(10)
        if rl_init_success:
            rl_train_success = advanced_rl.train(100)  # Quick training for demo
            results["rl"] = {"trained": rl_train_success, "episodes": 100}
        else:
            results["rl"] = {"trained": False, "error": "Initialization failed"}
        
        return {
            "status": "success",
            "message": "Demo training completed",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in demo training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/demo/test-pipeline")
async def demo_test_enhanced_pipeline(threat_data: Dict[str, Any]):
    """Demo: Test the full enhanced ML pipeline"""
    try:
        results = {}
        
        # Test detection
        if advanced_detector and advanced_detector.is_trained:
            detection_result = advanced_detector.predict(threat_data.get("packets", []))
            results["detection"] = detection_result
        else:
            results["detection"] = {"error": "Detection model not trained"}
        
        # Test classification
        if advanced_classifier and advanced_classifier.is_trained:
            classification_result = advanced_classifier.predict(threat_data)
            results["classification"] = classification_result
        else:
            results["classification"] = {"error": "Classification model not trained"}
        
        # Test RL
        if advanced_rl and advanced_rl.is_trained:
            rl_result = advanced_rl.predict_action(threat_data)
            results["rl"] = rl_result
        else:
            results["rl"] = {"error": "RL model not trained"}
        
        return {
            "status": "success",
            "message": "Enhanced pipeline test completed",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in demo pipeline test: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 