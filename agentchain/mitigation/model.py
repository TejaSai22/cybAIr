"""
Mitigation Agent - RL-based threat response
"""
from typing import Dict, Any
import random
import os
from agentchain.mitigation.rl_agent import QLearningMitigationAgent

# Initialize the global agent instance
# We store the Q-table in the data directory if it exists, otherwise locally.
q_table_path = os.path.join(os.path.dirname(__file__), "q_table.json")
rl_agent = QLearningMitigationAgent(q_table_path=q_table_path)

def mitigate_threat(threat_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    RL-based threat mitigation using Custom Tabular Q-Learning.
    """
    threat_type = threat_data.get("threat_type", "unknown")
    severity = threat_data.get("severity", "medium")
    
    # Use the RL agent to choose the optimal action
    selected_action = rl_agent.choose_action(threat_type, severity)
    
    # We still mock confidence and estimated_time for the dashboard,
    # but the action itself is chosen via RL.
    return {
        "action": selected_action,
        "confidence": random.uniform(0.7, 0.95),
        "estimated_time": random.randint(30, 300),
        "status": "mitigation_initiated",
        "threat_type": threat_type,
        "severity": severity
    }

def provide_feedback(threat_type: str, severity: str, action: str, reward: float):
    """
    Allows the system to provide feedback (rewards) to the RL agent.
    """
    rl_agent.learn(threat_type, severity, action, reward)
 