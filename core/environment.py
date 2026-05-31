import numpy as np
from configs.simulation_config import SimulationConfig
from network.topology import NetworkTopology
from core.agent import Agent

class Environment:
    def __init__(self, config:SimulationConfig):
        self.config = config
        self.N = config.N
        self.K = config.K
        self.opinions_reliability = config.opinions_reliability
        self.network = NetworkTopology(self.config)
        self.agents = None
    
    def reset(self):
        """Initializes graph structure and instantiates agents with uniform priors."""
        # Initialize network
        self.network.create_network()
        
        # Create a matrix of uniform priors for every agent
        initial_alphas = np.ones((self.N, self.K))
        
        # Populate agents list
        self.agents = [
            Agent(self.config, initial_alphas=initial_alphas[i])
            for i in range(self.N)
        ]
    
    def step(self):
        """Executes a single dual-timescale iteration of individual and social learning."""
        # Phase 1: Global individual experiences
        # Every agent receives an independent external environmental signal
        individual_experience = np.random.choice(
            np.arange(self.K), 
            size=self.N, 
            p=self.opinions_reliability)
        
        for i in range(self.N):
            self.agents[i].attempt_incorporation(individual_experience[i], source_type="individual")
            
        # Phase 2: Asynchronous social influence
        # Select one node to update the opinion uniformely at random
        selected_index = int(np.random.randint(0, self.N))
        # Get an array of influencers
        neighbors = self.network.get_neighbors(node_idx=selected_index)
        
        # If agent has no neighbor this loop is skipped
        for i in range(neighbors.shape[0]):
            # Getting index of influencers
            neighbor_idx = neighbors[i]
            
            # Opinion realization of neighbot
            neighbor_opinion = self.agents[neighbor_idx].realize_opinion()
            
            # The selected agent evaluates the incroporation of that opinion
            self.agents[selected_index].attempt_incorporation(neighbor_opinion, source_type="social")