import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentchain.mitigation.model import mitigate_threat, provide_feedback, q_table_path

def test_rl():
    # Clear existing Q-table for a clean test
    if os.path.exists(q_table_path):
        os.remove(q_table_path)
        
    threat = "SQL Injection"
    severity = "High"
    
    print("Training RL Agent on (SQL Injection, High)...")
    print("Optimal action is 'block_ip'")
    
    # Train for 2000 episodes to ensure convergence
    for i in range(2000):
        result = mitigate_threat({"threat_type": threat, "severity": severity})
        action = result["action"]
        
        if action == "block_ip":
            reward = 1.0
        elif action in ["isolate_host", "alert_admin"]:
            reward = 0.5
        else:
            reward = -1.0 
            
        provide_feedback(threat, severity, action, reward)
        
    print("\nTraining complete. Testing learned policy (greedy):")
    
    from agentchain.mitigation.model import rl_agent
    rl_agent.epsilon = 0.0
    
    result = mitigate_threat({"threat_type": threat, "severity": severity})
    action = result["action"]
    
    print(f"Agent chose: {action}")
    
    # Print Q-table
    print("Q-Table for state:")
    state_key = rl_agent._get_state_key(threat, severity)
    print(json.dumps(rl_agent.q_table.get(state_key, {}), indent=2))
    
    if action == "block_ip":
        print("PASS: RL Agent successfully learned the optimal policy!")
    else:
        print("FAIL: RL Agent failed to learn.")

if __name__ == "__main__":
    test_rl()
