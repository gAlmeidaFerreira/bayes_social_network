from dataclasses import dataclass, field
import numpy as np
from typing import Dict, Any

@dataclass
class SimulationConfig:
    # Network Topology Parameters
    N: int = 100                            # Total number of nodes
    m: int = 3                              # Number of connections of new node on a scale free network
    #TODO: #1 Introduce a gamma distribution for the prior parameters to generate different priors for each agent
    
    # Cognitive and memory parameters
    lambda_decay: float = 0.95              # Memory decay factor
    w_ind: float = 1.0                      # Count weight added for incorporated individual experience
    w_soc: float = 0.5                      # Count weight added for incorporated social experiece
    
    # Dissonance (sigmoid function)
    beta_ind: float = 2.0                   # Rigidity/slope for individual dissonance
    tau_ind: float = 1.5                    # tolerance thresshold for individual experience
    beta_soc: float = 1.5                   # Rigidity/slope for social dissonance
    tau_soc: float = 1.0                    # tolerance thresshold for social interactions
    #TODO: #2 Introduce randomly assigned beta and tau parameters for opinion acceptance
    
    # Global simulation settings
    max_steps: int = 10000                  # Max number of timestamps
    K: int = 2                              # Number of discrete opinions
    record_alphas_every: int = 200          # Interval to record couting parameters for dirichlet distribution of each agent
    opinions_reliability:np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5], dtype=float)) # reliability of each opinion
    seed: int = 42                          # Random seed for reproducibility
    
    def __post_init__(self):
        """Validates configuration bounds immediately after initialization."""
        if not (0 < self.lambda_decay <= 1.0):
            raise ValueError(f"lambda_decay (λ) must be in (0, 1], got {self.lambda_decay}")
        if self.m >= self.N:
            raise ValueError(f"m ({self.m}) must be strictly less than N ({self.N}) for scale-free generation.")
        if self.beta_ind <= 0 or self.beta_soc <= 0:
            raise ValueError("Beta parameters must be strictly positive to maintain a proper sigmoid slope.")
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the config into a dictionary for clean logging or exporting to JSON/CSV."""
        return self.__dict__