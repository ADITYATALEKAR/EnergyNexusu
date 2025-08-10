# src/optimization/algorithms/drl/enhanced_ddpg_agent.py
"""
TRANSFORMATION EXAMPLE: Original agent.py → Enhanced DDPG Agent
================================================================

This shows exactly how we transform the original EnergySystem_DRL code
into our professional-grade EnergyNexus system.

BEFORE (Original): Basic DDPG with limited features
AFTER (Enhanced): Professional-grade multi-objective agent

Key improvements:
1. Prioritized Experience Replay (learns from important experiences first)
2. Multi-objective reward function (cost + sustainability + stability)  
3. Real-world energy constraints (ramp rates, minimum loads)
4. Advanced exploration strategies (Ornstein-Uhlenbeck noise)
5. Professional monitoring and logging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Set up professional logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EnhancedDRLConfig:
    """
    ENHANCEMENT 1: Professional configuration management
    
    Original: Hard-coded parameters scattered throughout code
    Enhanced: Centralized, documented configuration
    """
    # Network architecture
    state_dim: int = 8          # Enhanced: [solar, wind, demand, hour, weekday, gas_prev, battery_soc, price]
    action_dim: int = 2         # Enhanced: [gas_output, battery_action] 
    actor_hidden1: int = 400
    actor_hidden2: int = 300
    critic_hidden1: int = 400
    critic_hidden2: int = 300
    
    # Learning parameters
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    batch_size: int = 256
    memory_size: int = 1000000
    gamma: float = 0.99
    tau: float = 0.001
    
    # Energy system constraints (NEW - real-world limits)
    gas_min_load: float = 0.3       # 30% minimum load
    gas_max_load: float = 1.0       # 100% maximum load  
    gas_ramp_rate: float = 0.1      # 10% ramp rate per step
    battery_capacity: float = 500.0  # kWh capacity
    battery_efficiency: float = 0.9  # Round-trip efficiency
    
    # Multi-objective weights (NEW - balances multiple goals)
    cost_weight: float = 0.4         # 40% focus on cost
    sustainability_weight: float = 0.3  # 30% focus on renewables
    stability_weight: float = 0.2    # 20% focus on grid stability
    safety_weight: float = 0.1       # 10% focus on operational safety

class PrioritizedReplayBuffer:
    """
    ENHANCEMENT 2: Prioritized Experience Replay
    
    Original: Simple random sampling from replay buffer
    Enhanced: Prioritizes important experiences for faster learning
    
    Why this matters for energy systems:
    - Critical events (outages, price spikes) are rare but crucial to learn from
    - Faster convergence to optimal policies
    - Better handling of imbalanced scenarios
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        """Add experience with maximum priority (so it gets sampled at least once)"""
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)
    
    def sample(self, batch_size: int, beta: float = 0.4):
        """
        Sample experiences based on their priority
        
        Returns:
            batch: Sampled experiences
            indices: Indices for updating priorities
            weights: Importance sampling weights
        """
        if len(self.buffer) == 0:
            return None, None, None
        
        # Calculate sampling probabilities based on priorities
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Get experiences
        batch = [self.buffer[i] for i in indices]
        
        # Calculate importance sampling weights (corrects for bias)
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        return batch, indices, weights
    
    def update_priorities(self, indices, priorities):
        """Update priorities based on TD errors"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Small epsilon to avoid zero priority

class EnergyConstraints:
    """
    ENHANCEMENT 3: Real-world energy system constraints
    
    Original: No realistic operational constraints
    Enhanced: Physics-based constraints that real power plants must follow
    """
    
    def __init__(self, config: EnhancedDRLConfig):
        self.config = config
        
    def apply_gas_constraints(self, raw_gas_action: float, previous_gas_output: float) -> float:
        """
        Apply gas turbine operational constraints
        
        Real gas turbines have:
        - Minimum stable load (can't run below 30% of capacity)
        - Ramp rate limits (can't change output instantly)
        - Start-up and shut-down procedures
        """
        # Convert action from [-1,1] to [min_load, max_load]
        desired_output = self.config.gas_min_load + (raw_gas_action + 1) * 0.5 * (
            self.config.gas_max_load - self.config.gas_min_load
        )
        
        # Apply ramp rate constraints
        max_increase = previous_gas_output + self.config.gas_ramp_rate
        max_decrease = previous_gas_output - self.config.gas_ramp_rate
        
        # Constrain the output
        constrained_output = np.clip(desired_output, max_decrease, max_increase)
        constrained_output = np.clip(constrained_output, 
                                   self.config.gas_min_load, 
                                   self.config.gas_max_load)
        
        return constrained_output
    
    def apply_battery_constraints(self, raw_battery_action: float, current_soc: float) -> float:
        """
        Apply battery operational constraints
        
        Real batteries have:
        - State of charge limits
        - Power limits
        - Efficiency losses
        """
        # Prevent overcharging or deep discharge
        if current_soc > 0.95 and raw_battery_action > 0:  # Nearly full, can't charge
            return 0.0
        elif current_soc < 0.05 and raw_battery_action < 0:  # Nearly empty, can't discharge
            return 0.0
        else:
            return raw_battery_action

class MultiObjectiveReward:
    """
    ENHANCEMENT 4: Multi-objective reward function
    
    Original: Simple cost-only reward
    Enhanced: Balances cost, sustainability, stability, and safety
    
    This is what makes our system commercially valuable!
    """
    
    def __init__(self, config: EnhancedDRLConfig):
        self.config = config
        
    def calculate_reward(self, system_state: Dict) -> Tuple[float, Dict]:
        """
        Calculate multi-objective reward that balances:
        1. Economic efficiency (minimize costs)
        2. Environmental sustainability (maximize renewables)
        3. Grid stability (minimize fluctuations)
        4. Operational safety (avoid constraint violations)
        """
        
        # 1. Cost efficiency reward (40% weight)
        cost_reward = -system_state['total_cost'] / 1000  # Normalize to reasonable scale
        
        # 2. Sustainability reward (30% weight) 
        renewable_percentage = system_state['renewable_generation'] / system_state['total_generation']
        sustainability_reward = renewable_percentage * 100  # 0-100 scale
        
        # 3. Stability reward (20% weight)
        generation_change = abs(system_state['current_generation'] - system_state['previous_generation'])
        stability_reward = -generation_change * 10  # Penalize large changes
        
        # 4. Safety reward (10% weight)
        violations = system_state.get('constraint_violations', 0)
        safety_reward = 50 if violations == 0 else -violations * 100