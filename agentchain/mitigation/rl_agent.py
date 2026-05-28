import json
import os
import random
import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

class QLearningMitigationAgent:
    """
    A custom Tabular Q-Learning Agent for selecting threat mitigation actions.
    Maintains a Q-table that maps (threat_type, severity) states to action Q-values.
    """
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.1,
        q_table_path: str = "q_table.json"
    ):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table_path = q_table_path
        
        # State: string representation of (threat_type, severity)
        # Value: Dictionary mapping action -> Q-value
        self.q_table: Dict[str, Dict[str, float]] = {}
        
        # All possible actions our system can take
        self.available_actions = [
            "block_ip", 
            "isolate_host", 
            "alert_admin", 
            "rate_limit", 
            "monitor_closely", 
            "log_event", 
            "monitor"
        ]
        
        self.load()

    def _get_state_key(self, threat_type: str, severity: str) -> str:
        # Normalize and create a composite key
        return f"{str(threat_type).lower()}|{str(severity).lower()}"

    def _ensure_state_exists(self, state_key: str):
        if state_key not in self.q_table:
            # Initialize Q-values to 0.0 for all actions
            self.q_table[state_key] = {action: 0.0 for action in self.available_actions}

    def choose_action(self, threat_type: str, severity: str) -> str:
        """Selects an action using an epsilon-greedy policy."""
        state_key = self._get_state_key(threat_type, severity)
        self._ensure_state_exists(state_key)
        
        # Exploration
        if random.random() < self.epsilon:
            return random.choice(self.available_actions)
            
        # Exploitation (choose action with max Q-value)
        action_values = self.q_table[state_key]
        max_q = max(action_values.values())
        
        # Break ties randomly
        best_actions = [a for a, q in action_values.items() if q == max_q]
        return random.choice(best_actions)

    def learn(self, threat_type: str, severity: str, action: str, reward: float):
        """Updates the Q-value for the given state-action pair using the Bellman equation."""
        state_key = self._get_state_key(threat_type, severity)
        self._ensure_state_exists(state_key)
        
        if action not in self.available_actions:
            logger.warning(f"Attempted to learn about unknown action: {action}")
            return
            
        current_q = self.q_table[state_key][action]
        
        # For this simple formulation, the episode ends immediately after the mitigation action,
        # so there is no 'next_state' max_Q to add. It simplifies to:
        # Q(s,a) = Q(s,a) + alpha * (reward - Q(s,a))
        new_q = current_q + self.lr * (reward - current_q)
        
        self.q_table[state_key][action] = new_q
        self.save()

    def save(self):
        """Persists the Q-table to disk."""
        try:
            with open(self.q_table_path, 'w') as f:
                json.dump(self.q_table, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save Q-table: {e}")

    def load(self):
        """Loads the Q-table from disk if it exists."""
        if os.path.exists(self.q_table_path):
            try:
                with open(self.q_table_path, 'r') as f:
                    loaded_data = json.load(f)
                    # Validate structure
                    if isinstance(loaded_data, dict):
                        self.q_table = loaded_data
            except Exception as e:
                logger.error(f"Failed to load Q-table: {e}")
