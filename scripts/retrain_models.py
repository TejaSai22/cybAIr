import sys
import os
import numpy as np
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentchain.detection.model import detector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RetrainingPipeline")

def generate_training_data(n_samples=1000):
    """
    Generates synthetic network traffic data for retraining.
    In a real system, this would query a database of historical logs.
    """
    # Normal traffic: mostly clustered around standard values
    # Features: [packet_size, inter_arrival_time, flag_count]
    normal_data = np.random.normal(loc=[500, 0.05, 2], scale=[100, 0.01, 1], size=(int(n_samples * 0.95), 3))
    
    # Anomalous traffic: outliers (e.g., massive packets, very fast bursts)
    anomalous_data = np.random.uniform(low=[1500, 0.001, 5], high=[5000, 0.01, 10], size=(int(n_samples * 0.05), 3))
    
    X = np.vstack([normal_data, anomalous_data])
    # Shuffle the data
    np.random.shuffle(X)
    return X

def retrain():
    logger.info("Starting automated ML retraining pipeline...")
    
    logger.info("Gathering historical network traffic logs...")
    X_train = generate_training_data(2000)
    logger.info(f"Gathered {len(X_train)} data points.")
    
    logger.info("Retraining Isolation Forest model...")
    detector.fit(X_train)
    
    logger.info("Model retrained and saved successfully!")
    
    # Evaluate a quick test
    test_normal = np.array([[500, 0.05, 2]])
    test_anomaly = np.array([[4000, 0.002, 8]])
    
    preds, scores = detector.predict(np.vstack([test_normal, test_anomaly]))
    logger.info(f"Test Predictions (1=Normal, -1=Anomaly): {preds}")
    logger.info(f"Test Anomaly Scores: {scores}")

if __name__ == "__main__":
    retrain()
