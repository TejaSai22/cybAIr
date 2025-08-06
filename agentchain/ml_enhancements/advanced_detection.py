"""
Advanced Anomaly Detection Models
Enhanced ML models for better threat detection
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
from datetime import datetime, timedelta
import json

# For advanced models (optional)
try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import OneClassSVM
    from sklearn.cluster import DBSCAN
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False
    logging.warning("Advanced ML models not available. Install scikit-learn>=1.0.0")

logger = logging.getLogger(__name__)

class AdvancedAnomalyDetector:
    """Advanced anomaly detection using multiple ML models"""
    
    def __init__(self, model_type: str = "ensemble"):
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained = False
        
        # Model configurations
        self.model_configs = {
            "isolation_forest": {
                "contamination": 0.1,
                "n_estimators": 100,
                "random_state": 42
            },
            "one_class_svm": {
                "kernel": "rbf",
                "nu": 0.1,
                "gamma": "scale"
            },
            "dbscan": {
                "eps": 0.5,
                "min_samples": 5
            },
            "mlp": {
                "hidden_layer_sizes": (100, 50),
                "max_iter": 500,
                "random_state": 42
            }
        }
    
    def extract_advanced_features(self, packets: List[Dict[str, Any]]) -> np.ndarray:
        """Extract advanced features from network packets"""
        if not packets:
            return np.array([])
        
        features = []
        for packet in packets:
            # Basic features
            packet_features = [
                packet.get("length", 0),
                packet.get("src_port", 0),
                packet.get("dst_port", 0),
                packet.get("protocol", 0),
                packet.get("ttl", 0),
                packet.get("window", 0),
                packet.get("payload_size", 0),
                int(packet.get("suspicious", False))
            ]
            
            # Advanced features
            # Port entropy (measure of port diversity)
            port_entropy = self._calculate_port_entropy(packets)
            packet_features.append(port_entropy)
            
            # Protocol distribution
            protocol_dist = self._get_protocol_distribution(packets)
            packet_features.extend(protocol_dist)
            
            # Time-based features
            time_features = self._extract_time_features(packet)
            packet_features.extend(time_features)
            
            # Statistical features
            stat_features = self._calculate_statistical_features(packets)
            packet_features.extend(stat_features)
            
            features.append(packet_features)
        
        return np.array(features)
    
    def _calculate_port_entropy(self, packets: List[Dict[str, Any]]) -> float:
        """Calculate entropy of destination ports"""
        ports = [p.get("dst_port", 0) for p in packets if p.get("dst_port")]
        if not ports:
            return 0.0
        
        unique_ports, counts = np.unique(ports, return_counts=True)
        probabilities = counts / len(ports)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def _get_protocol_distribution(self, packets: List[Dict[str, Any]]) -> List[float]:
        """Get normalized protocol distribution"""
        protocols = [p.get("protocol", 0) for p in packets]
        unique_protocols = [6, 17, 1]  # TCP, UDP, ICMP
        
        distribution = []
        for protocol in unique_protocols:
            count = protocols.count(protocol)
            distribution.append(count / len(protocols) if protocols else 0.0)
        
        return distribution
    
    def _extract_time_features(self, packet: Dict[str, Any]) -> List[float]:
        """Extract time-based features"""
        try:
            timestamp = datetime.fromisoformat(packet.get("timestamp", ""))
            hour = timestamp.hour / 24.0  # Normalize to [0,1]
            minute = timestamp.minute / 60.0
            second = timestamp.second / 60.0
            
            # Day of week (0=Monday, 6=Sunday)
            day_of_week = timestamp.weekday() / 7.0
            
            return [hour, minute, second, day_of_week]
        except:
            return [0.0, 0.0, 0.0, 0.0]
    
    def _calculate_statistical_features(self, packets: List[Dict[str, Any]]) -> List[float]:
        """Calculate statistical features"""
        if not packets:
            return [0.0] * 5
        
        lengths = [p.get("length", 0) for p in packets]
        payload_sizes = [p.get("payload_size", 0) for p in packets]
        
        features = [
            np.mean(lengths) if lengths else 0.0,
            np.std(lengths) if lengths else 0.0,
            np.mean(payload_sizes) if payload_sizes else 0.0,
            np.std(payload_sizes) if payload_sizes else 0.0,
            len([p for p in packets if p.get("suspicious", False)]) / len(packets)
        ]
        
        return features
    
    def train(self, training_data: List[Dict[str, Any]], labels: Optional[List[int]] = None):
        """Train the anomaly detection models"""
        logger.info(f"Training advanced anomaly detector with {len(training_data)} samples")
        
        # Extract features
        X = self.extract_advanced_features(training_data)
        if len(X) == 0:
            logger.error("No features extracted from training data")
            return False
        
        # Store feature names for later use
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers["main"] = scaler
        
        if self.model_type == "ensemble":
            self._train_ensemble_models(X_scaled, labels)
        elif self.model_type == "isolation_forest":
            self._train_isolation_forest(X_scaled)
        elif self.model_type == "one_class_svm":
            self._train_one_class_svm(X_scaled)
        elif self.model_type == "dbscan":
            self._train_dbscan(X_scaled)
        elif self.model_type == "mlp" and labels is not None:
            self._train_mlp(X_scaled, labels)
        else:
            logger.error(f"Unknown model type: {self.model_type}")
            return False
        
        self.is_trained = True
        logger.info("Advanced anomaly detector training completed")
        return True
    
    def _train_ensemble_models(self, X: np.ndarray, labels: Optional[List[int]] = None):
        """Train ensemble of models"""
        # Isolation Forest
        iso_forest = IsolationForest(**self.model_configs["isolation_forest"])
        iso_forest.fit(X)
        self.models["isolation_forest"] = iso_forest
        
        # One-Class SVM
        if ADVANCED_MODELS_AVAILABLE:
            oc_svm = OneClassSVM(**self.model_configs["one_class_svm"])
            oc_svm.fit(X)
            self.models["one_class_svm"] = oc_svm
        
        # DBSCAN for clustering
        if ADVANCED_MODELS_AVAILABLE:
            dbscan = DBSCAN(**self.model_configs["dbscan"])
            dbscan.fit(X)
            self.models["dbscan"] = dbscan
        
        # Supervised model if labels provided
        if labels is not None and ADVANCED_MODELS_AVAILABLE:
            mlp = MLPClassifier(**self.model_configs["mlp"])
            mlp.fit(X, labels)
            self.models["mlp"] = mlp
    
    def _train_isolation_forest(self, X: np.ndarray):
        """Train Isolation Forest model"""
        iso_forest = IsolationForest(**self.model_configs["isolation_forest"])
        iso_forest.fit(X)
        self.models["isolation_forest"] = iso_forest
    
    def _train_one_class_svm(self, X: np.ndarray):
        """Train One-Class SVM model"""
        if not ADVANCED_MODELS_AVAILABLE:
            logger.error("One-Class SVM not available")
            return
        
        oc_svm = OneClassSVM(**self.model_configs["one_class_svm"])
        oc_svm.fit(X)
        self.models["one_class_svm"] = oc_svm
    
    def _train_dbscan(self, X: np.ndarray):
        """Train DBSCAN clustering model"""
        if not ADVANCED_MODELS_AVAILABLE:
            logger.error("DBSCAN not available")
            return
        
        dbscan = DBSCAN(**self.model_configs["dbscan"])
        dbscan.fit(X)
        self.models["dbscan"] = dbscan
    
    def _train_mlp(self, X: np.ndarray, labels: List[int]):
        """Train Multi-layer Perceptron"""
        if not ADVANCED_MODELS_AVAILABLE:
            logger.error("MLP not available")
            return
        
        mlp = MLPClassifier(**self.model_configs["mlp"])
        mlp.fit(X, labels)
        self.models["mlp"] = mlp
    
    def predict(self, packets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict anomalies using trained models"""
        if not self.is_trained:
            return {"anomaly_score": 0.5, "is_anomaly": False, "confidence": 0.0}
        
        # Extract features
        X = self.extract_advanced_features(packets)
        if len(X) == 0:
            return {"anomaly_score": 0.5, "is_anomaly": False, "confidence": 0.0}
        
        # Scale features
        X_scaled = self.scalers["main"].transform(X)
        
        # Get predictions from all models
        predictions = {}
        anomaly_scores = []
        
        # Isolation Forest
        if "isolation_forest" in self.models:
            iso_scores = self.models["isolation_forest"].score_samples(X_scaled)
            iso_predictions = self.models["isolation_forest"].predict(X_scaled)
            predictions["isolation_forest"] = {
                "scores": iso_scores.tolist(),
                "predictions": iso_predictions.tolist()
            }
            anomaly_scores.extend(iso_scores)
        
        # One-Class SVM
        if "one_class_svm" in self.models:
            svm_scores = self.models["one_class_svm"].score_samples(X_scaled)
            svm_predictions = self.models["one_class_svm"].predict(X_scaled)
            predictions["one_class_svm"] = {
                "scores": svm_scores.tolist(),
                "predictions": svm_predictions.tolist()
            }
            anomaly_scores.extend(svm_scores)
        
        # DBSCAN
        if "dbscan" in self.models:
            dbscan_labels = self.models["dbscan"].labels_
            # Convert DBSCAN labels to anomaly scores (-1 = anomaly)
            dbscan_scores = [-1.0 if label == -1 else 1.0 for label in dbscan_labels]
            predictions["dbscan"] = {
                "labels": dbscan_labels.tolist(),
                "scores": dbscan_scores
            }
            anomaly_scores.extend(dbscan_scores)
        
        # MLP (if available and trained)
        if "mlp" in self.models:
            mlp_proba = self.models["mlp"].predict_proba(X_scaled)
            mlp_predictions = self.models["mlp"].predict(X_scaled)
            predictions["mlp"] = {
                "probabilities": mlp_proba.tolist(),
                "predictions": mlp_predictions.tolist()
            }
            # Use probability of anomaly class
            anomaly_proba = mlp_proba[:, 1] if mlp_proba.shape[1] > 1 else mlp_proba[:, 0]
            anomaly_scores.extend(anomaly_proba)
        
        # Ensemble decision
        if anomaly_scores:
            avg_score = np.mean(anomaly_scores)
            # Normalize to [0, 1] range
            normalized_score = max(0.0, min(1.0, (avg_score + 1) / 2))
            is_anomaly = normalized_score > 0.7  # Threshold for anomaly detection
            confidence = abs(normalized_score - 0.5) * 2  # Confidence based on distance from 0.5
        else:
            normalized_score = 0.5
            is_anomaly = False
            confidence = 0.0
        
        return {
            "anomaly_score": float(normalized_score),
            "is_anomaly": bool(is_anomaly),
            "confidence": float(confidence),
            "model_predictions": predictions,
            "feature_count": len(self.feature_names),
            "models_used": list(self.models.keys())
        }
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            logger.error("Cannot save untrained model")
            return False
        
        model_data = {
            "model_type": self.model_type,
            "models": self.models,
            "scalers": self.scalers,
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
            self.scalers = model_data["scalers"]
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
            "feature_names": self.feature_names,
            "model_configs": self.model_configs
        } 