import numpy as np
import pandas as pd
from typing import Dict, List, Any
from analysis.metrics import calculate_global_entropy, calculate_polarization_index
from configs.simulation_config import SimulationConfig

class StateTracker:
    """
    Logs fast scalar telemetry at every simulation step and archives raw alpha 
    concentration parameter snapshots periodically for deferred continuous analysis.
    """
    def __init__(self, config: SimulationConfig):
        self.K = config.K
        self.record_alphas_every = config.record_alphas_every
        
        # Continuous time-series logs
        self.history: Dict[str, List[Any]] = {
            "step": [],
            "global_entropy": [],
            "polarization": [],
            "ind_rejection_rate": [],
            "soc_rejection_rate": []
        }
        # Explicitly macro channels for opinions vectors
        for k in range(self.K):
            self.history[f"mean_p_{k}"] = []
            
        # heavy-data deferred repository: maps { step_idx: alpha_matrix_snapshot}
        self.alpha_snapshot: Dict[int, np.ndarray] = {}
        
    def record_step(self, step_idx: int, agents: list, ind_rejections: List[bool], soc_rejections: List[bool]):
        """Vectorizes agent states and updates history logs."""
        # Probability expectations from all agents
        p_matrix = np.vstack([agents.p for agent in agents])
        
        # Process fast track telemetry
        self.history["step"].append(step_idx)
        self.history["global_entropy"].append(calculate_global_entropy(p_matrix))
        self.history["polarization"].append(calculate_polarization_index(p_matrix))
        self.history["ind_rejection_rate"].append(float(np.mean(ind_rejections)) if ind_rejections else 0.0)
        self.history["soc_rejection_rate"].append(float(np.mean(soc_rejections)) if soc_rejections else 0.0)
        
        mean_dist = np.mean(p_matrix, axis=0)
        for k in range(self.K):
            self.history[f"mean_p_{k}"].append(mean_dist[k])
        
        # Process slow track
        if step_idx % self.record_alphas_every == 0:
            alpha_matrix = np.vstack([agent.alphas for agent in agents])
            self.alpha_snapshots[step_idx] = alpha_matrix.copy()
    
    def get_dataframe(self):
        """Compiles the time-series history into a standard Pandas DataFrame."""
        return pd.DataFrame(self.history)
                    
        