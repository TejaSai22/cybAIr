"""
Advanced Reinforcement Learning for Threat Response
Enhanced RL models for optimal mitigation actions
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import random
import logging
from datetime import datetime
import json
import pickle

# Optional RL libraries
try:
    import gym
    from gym import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    logging.warning("Gym not available for advanced RL")

logger = logging.getLogger(__name__)

class ThreatEnvironment:
    """Custom environment for threat response RL"""
    
    def __init__(self, state_size: int = 10, action_size: int = 5):
        self.state_size = state_size
        self.action_size = action_size
        
        # Define action space
        self.actions = [
            "block_ip", "isolate_host", "alert_admin", "rate_limit", "monitor"
        ]
        
        # Define state space
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(state_size,), dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(action_size)
        
        # Environment state
        self.current_state = None
        self.threat_history = []
        self.response_history = []
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_state = np.random.random(self.state_size)
        self.threat_history = []
        self.response_history = []
        return self.current_state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take action and return next state, reward, done, info"""
        if action >= self.action_size:
            raise ValueError(f"Invalid action: {action}")
        
        # Get action name
        action_name = self.actions[action]
        
        # Simulate threat response
        reward = self._calculate_reward(action_name)
        
        # Update state
        self.current_state = self._update_state(action_name)
        
        # Check if episode is done
        done = self._is_episode_done()
        
        # Additional info
        info = {
            "action_taken": action_name,
            "reward": reward,
            "threat_level": self._get_threat_level()
        }
        
        return self.current_state, reward, done, info
    
    def _calculate_reward(self, action: str) -> float:
        """Calculate reward for taking an action"""
        base_reward = 0.0
        
        # Reward based on action effectiveness
        action_effectiveness = {
            "block_ip": 0.8,
            "isolate_host": 0.9,
            "alert_admin": 0.6,
            "rate_limit": 0.7,
            "monitor": 0.4
        }
        
        base_reward += action_effectiveness.get(action, 0.5)
        
        # Penalty for over-reaction
        if action in ["isolate_host", "block_ip"] and self._get_threat_level() < 0.7:
            base_reward -= 0.3
        
        # Bonus for appropriate response
        if self._get_threat_level() > 0.8 and action in ["isolate_host", "block_ip"]:
            base_reward += 0.2
        
        return base_reward
    
    def _update_state(self, action: str) -> np.ndarray:
        """Update environment state based on action"""
        # Simulate state transition
        new_state = self.current_state.copy()
        
        # Reduce threat level based on action effectiveness
        threat_reduction = {
            "block_ip": 0.3,
            "isolate_host": 0.4,
            "alert_admin": 0.1,
            "rate_limit": 0.2,
            "monitor": 0.05
        }
        
        reduction = threat_reduction.get(action, 0.1)
        new_state[0] = max(0.0, new_state[0] - reduction)  # Threat level
        
        # Add some randomness
        new_state += np.random.normal(0, 0.05, self.state_size)
        new_state = np.clip(new_state, 0, 1)
        
        return new_state
    
    def _get_threat_level(self) -> float:
        """Get current threat level"""
        return self.current_state[0] if self.current_state is not None else 0.0
    
    def _is_episode_done(self) -> bool:
        """Check if episode should end"""
        # End if threat level is very low or very high
        threat_level = self._get_threat_level()
        return threat_level < 0.1 or threat_level > 0.95

class QLearningAgent:
    """Q-Learning agent for threat response"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.1, 
                 discount_factor: float = 0.95, epsilon: float = 0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Q-table (state discretization for continuous state space)
        self.state_bins = 10
        self.q_table = {}
        
        # Training history
        self.training_history = []
        
    def discretize_state(self, state: np.ndarray) -> str:
        """Discretize continuous state for Q-table lookup"""
        discrete_state = []
        for i, value in enumerate(state):
            bin_index = int(value * self.state_bins)
            bin_index = min(bin_index, self.state_bins - 1)
            discrete_state.append(bin_index)
        return str(discrete_state)
    
    def get_action(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy"""
        discrete_state = self.discretize_state(state)
        
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # Exploit learned Q-values
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_size)
        
        return np.argmax(self.q_table[discrete_state])
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool):
        """Update Q-values using Q-learning update rule"""
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        # Initialize Q-values if not present
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_size)
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros(self.action_size)
        
        # Q-learning update
        current_q = self.q_table[discrete_state][action]
        max_next_q = np.max(self.q_table[discrete_next_state]) if not done else 0
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[discrete_state][action] = new_q
        
        # Record training step
        self.training_history.append({
            "state": state.tolist(),
            "action": action,
            "reward": reward,
            "next_state": next_state.tolist(),
            "q_value": new_q
        })
    
    def save_model(self, filepath: str):
        """Save the trained Q-learning model"""
        model_data = {
            "q_table": self.q_table,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "state_bins": self.state_bins,
            "training_history": self.training_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Q-learning model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained Q-learning model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_table = model_data["q_table"]
            self.state_size = model_data["state_size"]
            self.action_size = model_data["action_size"]
            self.learning_rate = model_data["learning_rate"]
            self.discount_factor = model_data["discount_factor"]
            self.epsilon = model_data["epsilon"]
            self.state_bins = model_data["state_bins"]
            self.training_history = model_data["training_history"]
            
            logger.info(f"Q-learning model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading Q-learning model: {e}")
            return False

class PolicyGradientAgent:
    """Policy Gradient agent for threat response"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Policy parameters (simple linear policy)
        self.weights = np.random.randn(state_size, action_size) * 0.01
        self.bias = np.zeros(action_size)
        
        # Training history
        self.training_history = []
        
    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """Get action probabilities using softmax policy"""
        logits = np.dot(state, self.weights) + self.bias
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        return exp_logits / np.sum(exp_logits)
    
    def get_action(self, state: np.ndarray) -> int:
        """Sample action from policy"""
        probs = self.get_action_probs(state)
        return np.random.choice(self.action_size, p=probs)
    
    def update(self, states: List[np.ndarray], actions: List[int], 
               rewards: List[float]):
        """Update policy using policy gradient"""
        if not states:
            return
        
        # Convert to numpy arrays
        states_array = np.array(states)
        actions_array = np.array(actions)
        rewards_array = np.array(rewards)
        
        # Normalize rewards
        rewards_array = (rewards_array - np.mean(rewards_array)) / (np.std(rewards_array) + 1e-8)
        
        # Compute gradients
        for state, action, reward in zip(states_array, actions_array, rewards_array):
            probs = self.get_action_probs(state)
            
            # Policy gradient update
            for a in range(self.action_size):
                if a == action:
                    self.weights[:, a] += self.learning_rate * reward * state
                    self.bias[a] += self.learning_rate * reward
                else:
                    self.weights[:, a] -= self.learning_rate * reward * state * probs[a]
                    self.bias[a] -= self.learning_rate * reward * probs[a]
        
        # Record training step
        self.training_history.append({
            "episode_reward": np.sum(rewards),
            "avg_reward": np.mean(rewards),
            "policy_entropy": -np.sum(probs * np.log(probs + 1e-8))
        })
    
    def save_model(self, filepath: str):
        """Save the trained policy gradient model"""
        model_data = {
            "weights": self.weights,
            "bias": self.bias,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "learning_rate": self.learning_rate,
            "training_history": self.training_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Policy gradient model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained policy gradient model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.weights = model_data["weights"]
            self.bias = model_data["bias"]
            self.state_size = model_data["state_size"]
            self.action_size = model_data["action_size"]
            self.learning_rate = model_data["learning_rate"]
            self.training_history = model_data["training_history"]
            
            logger.info(f"Policy gradient model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading policy gradient model: {e}")
            return False

class AdvancedRLMitigation:
    """Advanced RL-based threat mitigation system"""
    
    def __init__(self, agent_type: str = "q_learning"):
        self.agent_type = agent_type
        self.environment = None
        self.agent = None
        self.is_trained = False
        
        # Available actions
        self.actions = [
            "block_ip", "isolate_host", "alert_admin", "rate_limit", "monitor"
        ]
        
        # Training parameters
        self.episodes = 1000
        self.max_steps = 100
        
    def initialize(self, state_size: int = 10):
        """Initialize the RL environment and agent"""
        if not GYM_AVAILABLE:
            logger.error("Gym not available for RL initialization")
            return False
        
        try:
            # Create environment
            self.environment = ThreatEnvironment(state_size, len(self.actions))
            
            # Create agent
            if self.agent_type == "q_learning":
                self.agent = QLearningAgent(state_size, len(self.actions))
            elif self.agent_type == "policy_gradient":
                self.agent = PolicyGradientAgent(state_size, len(self.actions))
            else:
                logger.error(f"Unknown agent type: {self.agent_type}")
                return False
            
            logger.info(f"RL system initialized with {self.agent_type} agent")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing RL system: {e}")
            return False
    
    def train(self, episodes: int = None):
        """Train the RL agent"""
        if not self.environment or not self.agent:
            logger.error("RL system not initialized")
            return False
        
        episodes = episodes or self.episodes
        logger.info(f"Training RL agent for {episodes} episodes")
        
        training_rewards = []
        
        for episode in range(episodes):
            state = self.environment.reset()
            episode_reward = 0
            episode_states = []
            episode_actions = []
            episode_rewards = []
            
            for step in range(self.max_steps):
                # Choose action
                action = self.agent.get_action(state)
                
                # Take action
                next_state, reward, done, info = self.environment.step(action)
                
                # Store experience
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                
                # Update agent
                if self.agent_type == "q_learning":
                    self.agent.update(state, action, reward, next_state, done)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Update policy gradient agent
            if self.agent_type == "policy_gradient":
                self.agent.update(episode_states, episode_actions, episode_rewards)
            
            training_rewards.append(episode_reward)
            
            if episode % 100 == 0:
                avg_reward = np.mean(training_rewards[-100:])
                logger.info(f"Episode {episode}, Average Reward: {avg_reward:.3f}")
        
        self.is_trained = True
        logger.info("RL training completed")
        return True
    
    def predict_action(self, threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict optimal action for threat mitigation"""
        if not self.is_trained or not self.agent:
            return {
                "action": "monitor",
                "confidence": 0.5,
                "reason": "RL agent not trained"
            }
        
        # Convert threat data to state
        state = self._threat_data_to_state(threat_data)
        
        # Get action from agent
        action_idx = self.agent.get_action(state)
        action = self.actions[action_idx]
        
        # Get confidence (for Q-learning, use Q-value; for policy gradient, use probability)
        if self.agent_type == "q_learning":
            discrete_state = self.agent.discretize_state(state)
            if discrete_state in self.agent.q_table:
                q_values = self.agent.q_table[discrete_state]
                confidence = float(np.max(q_values))
            else:
                confidence = 0.5
        else:
            probs = self.agent.get_action_probs(state)
            confidence = float(probs[action_idx])
        
        return {
            "action": action,
            "confidence": confidence,
            "agent_type": self.agent_type,
            "state_features": state.tolist(),
            "all_action_probs": self._get_all_action_probs(state)
        }
    
    def _threat_data_to_state(self, threat_data: Dict[str, Any]) -> np.ndarray:
        """Convert threat data to RL state representation"""
        state = np.zeros(self.environment.state_size)
        
        # Threat level (0-1)
        severity = threat_data.get("severity", "medium")
        if severity == "high":
            state[0] = 0.9
        elif severity == "medium":
            state[0] = 0.5
        else:
            state[0] = 0.2
        
        # Number of threat indicators
        indicators = threat_data.get("threat_indicators", [])
        state[1] = min(1.0, len(indicators) / 5.0)
        
        # Anomaly score
        anomaly_score = threat_data.get("anomaly_score", 0.5)
        state[2] = anomaly_score
        
        # Confidence from classification
        confidence = threat_data.get("confidence", 0.5)
        state[3] = confidence
        
        # Time-based features
        hour = datetime.now().hour
        state[4] = hour / 24.0
        
        # Add some randomness for exploration
        state[5:] = np.random.random(self.environment.state_size - 5)
        
        return state
    
    def _get_all_action_probs(self, state: np.ndarray) -> Dict[str, float]:
        """Get probabilities for all actions"""
        if self.agent_type == "q_learning":
            discrete_state = self.agent.discretize_state(state)
            if discrete_state in self.agent.q_table:
                q_values = self.agent.q_table[discrete_state]
                # Convert Q-values to probabilities using softmax
                exp_q = np.exp(q_values - np.max(q_values))
                probs = exp_q / np.sum(exp_q)
            else:
                probs = np.ones(len(self.actions)) / len(self.actions)
        else:
            probs = self.agent.get_action_probs(state)
        
        return {action: float(prob) for action, prob in zip(self.actions, probs)}
    
    def save_model(self, filepath: str):
        """Save the trained RL model"""
        if not self.is_trained:
            logger.error("Cannot save untrained model")
            return False
        
        model_data = {
            "agent_type": self.agent_type,
            "is_trained": self.is_trained,
            "actions": self.actions,
            "environment_state_size": self.environment.state_size if self.environment else None
        }
        
        # Save agent-specific data
        if self.agent:
            agent_filepath = filepath.replace('.pkl', '_agent.pkl')
            self.agent.save_model(agent_filepath)
            model_data["agent_filepath"] = agent_filepath
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"RL model saved to {filepath}")
        return True
    
    def load_model(self, filepath: str):
        """Load a trained RL model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.agent_type = model_data["agent_type"]
            self.is_trained = model_data["is_trained"]
            self.actions = model_data["actions"]
            
            # Load agent
            if "agent_filepath" in model_data:
                self.agent = None
                if self.agent_type == "q_learning":
                    self.agent = QLearningAgent(1, 1)  # Dummy initialization
                elif self.agent_type == "policy_gradient":
                    self.agent = PolicyGradientAgent(1, 1)  # Dummy initialization
                
                if self.agent:
                    self.agent.load_model(model_data["agent_filepath"])
            
            # Recreate environment
            state_size = model_data.get("environment_state_size", 10)
            self.environment = ThreatEnvironment(state_size, len(self.actions))
            
            logger.info(f"RL model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading RL model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the RL model"""
        return {
            "agent_type": self.agent_type,
            "is_trained": self.is_trained,
            "actions": self.actions,
            "environment_available": self.environment is not None,
            "agent_available": self.agent is not None,
            "gym_available": GYM_AVAILABLE
        } 