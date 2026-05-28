import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentchain.detection.model import detector
from agentchain.classification.model import classify_threat
from agentchain.triage.model import triage_threat
from agentchain.mitigation.model import mitigate_threat, rl_agent

def test_full_pipeline_e2e():
    """
    Simulates the full end-to-end flow of the AgentChain pipeline.
    """
    # 1. Detection Agent: Send an anomalous packet
    # (High packet size, very low inter-arrival time -> likely anomaly)
    import numpy as np
    malicious_packet = np.array([[5000, 0.001, 10]])
    
    # Force the detector to be trained for the test if it isn't
    if not detector.is_trained():
        dummy_data = np.random.normal(loc=[500, 0.05, 2], scale=[100, 0.01, 1], size=(100, 3))
        detector.fit(np.vstack([dummy_data, malicious_packet]))
        
    preds, scores = detector.predict(malicious_packet)
    # IsolationForest outputs -1 for anomalies
    assert preds[0] == -1, "Detector failed to identify the anomaly"
    
    # 2. Classification Agent: Mock Ollama to return SQL Injection
    with patch('agentchain.classification.model.ollama_client.generate_text') as mock_llm:
        mock_llm.return_value = "Based on the payload, this is a SQL Injection attack."
        
        classification_result = classify_threat({
            "packet_data": malicious_packet.tolist(),
            "anomaly_score": float(scores[0])
        })
        
        assert classification_result["threat_type"] == "classified"
        assert "SQL Injection" in classification_result["llm_result"]
        
        # Parse the mocked LLM result into our standard schema (simulating the app logic)
        identified_threat = "SQL Injection"
        
    # 3. Triage Agent: Assess severity
    triage_payload = {
        "threat_type": "SQL Injection",
        "details": "Detected high volume traffic characteristic of SQLi"
    }
    
    # We update the triage_threat rule in test just to make sure it handles SQLi (defaults to low if unknown)
    # The current triage_model.py doesn't explicitly have SQLi, so it falls to low/monitor.
    # Let's see what it returns natively:
    triage_result = triage_threat(triage_payload)
    severity = triage_result["severity"]
    
    # 4. Mitigation Agent: Q-Learning Action Selection
    mitigation_payload = {
        "threat_type": identified_threat,
        "severity": severity
    }
    
    # Set epsilon to 0 to ensure deterministic greedy action
    rl_agent.epsilon = 0.0
    
    # Artificially train the agent so 'block_ip' is definitely the highest Q-value for this state
    state_key = rl_agent._get_state_key(identified_threat, severity)
    rl_agent._ensure_state_exists(state_key)
    rl_agent.q_table[state_key]["block_ip"] = 10.0
    
    mitigation_result = mitigate_threat(mitigation_payload)
    
    # Assert the RL agent chose the optimal learned action
    assert mitigation_result["action"] == "block_ip", f"Expected block_ip but got {mitigation_result['action']}"
    assert mitigation_result["status"] == "mitigation_initiated"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
