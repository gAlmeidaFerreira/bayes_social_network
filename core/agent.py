from configs.simulation_config import SimulationConfig
from utils.math_utils import sigmoid
from typing import Literal
import numpy as np

class Agent:
    """ 
    Agents that will interact inside the network, having their own opinion that will cgange given individual
    experiences or ideas received from neighbors
    """
    def __init__(self, config:SimulationConfig, initial_alphas: np.ndarray = None):
        self.K = config.K                           # Number of Possible opinions
        
        if initial_alphas is not None:
            self.alphas = initial_alphas.copy()
        else:
            self.alphas = np.ones(self.K)
        
        self.p = np.zeros(self.K)                   # Probability of assuming each opinion on the set
        self.beta_ind = config.beta_ind
        self.beta_soc = config.beta_soc
        self.tau_ind = config.tau_ind
        self.tau_soc = config.tau_soc
        self.lambda_decay = config.lambda_decay
        self.w_ind = config.w_ind
        self.w_soc = config.w_soc
        
        self.get_expectation()
        
    def get_expectation(self):
        """Calculates and updates the normalized expected belief vector p from alphas."""
        total_mass = np.sum(self.alphas)
        if total_mass > 0:
            self.p = self.alphas / total_mass
        else:
            self.p = np.ones(self.K) / self.K
        return self.p
    
    def realize_opinion(self):
        """Returns a discrete opinion index by sampling a Categorical distribution parameterized by p."""
        self.get_expectation()
        opinion = np.random.choice(np.arange(self.K), p = self.p)
        return opinion
    
    def calculate_dissonance(self, x):
        """Quantifies individual cognitive friction or 'surprise' using negative log-likelihood."""
        self.get_expectation()
        prob = max(self.p[x], 1e-12)
        dissonance = -np.log(prob)
        return dissonance
    
    def attempt_incorporation(self, x, source_type: Literal["individual", "social"]):
        """
        Evaluates an incoming experience or peer opinion index x against internal dissonance boundaries.
        Applies memory decay unconditionally or conditionally based on acceptance rules.
        """
        if source_type == "individual":
            beta = self.beta_ind
            tau = self.tau_ind
            w = self.w_ind
        elif source_type == "social":
            beta = self.beta_soc
            tau = self.tau_soc
            w = self.w_soc
        else:
            raise ValueError("source_type must be 'individual' or 'social'")
        
        dissonance = self.calculate_dissonance(x)
        p_inc = sigmoid(dissonance, beta=beta, tau=tau)
        u = np.random.uniform()
        if u < p_inc:
            # ACCEPTS: applies memory decay and update the prior alpha
            self.alphas = self.lambda_decay * self.alphas
            self.alphas[x] += w
            self.get_expectation()
            return True
        else:
           # REJECTS: Apply memory decay but does not incorporate opinion
           self.alphas = self.lambda_decay * self.alphas
           self.get_expectation()
           return False
        
            
        
        
    
        