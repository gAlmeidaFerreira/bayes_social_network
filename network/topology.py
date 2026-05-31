import networkx as nx
import numpy as np
from configs.simulation_config import SimulationConfig

class NetworkTopology:
    """
    Handles the structural creation and tracking of the network topology.
    Generates directed scale-free network structures for the simulation.
    """
    def __init__(self, config: SimulationConfig):
        self.N = config.N                  # Number of nodes in the complete network
        self.m = config.m                  # Number of connections of new nodes implemented on the network
        self.graph = None                  # NetworkX Graph Object container
        self.adj_matrix = None             # NumPy adjacency matrix container
        
    def create_network(self):
        """
        Generates a true directed scale-free network using preferential attachment.
        Returns a NetworkX DiGraph object.
        """
        # Initilize undirected scale-free network
        undirected_G = nx.barabasi_albert_graph(n = self.N, m = self.m, seed=None)
        
        # Instantiate a directed graph
        G = nx.DiGraph()
        G.add_nodes_from(undirected_G.nodes())
        
        # Orient edges from newer nodes to older nodes (preferential source tracking)
        for u, v in undirected_G.edges():
            if u > v:
                G.add_edge(u, v) # u newer then v, so u follows v
            else:
                G.add_edge(v, u) # v newer then u, so v follows u
                
        self.graph = G
        self.adj_matrix = nx.to_numpy_array(G)
        
        return self.graph
    
    def get_neighbors(self, node_idx: int) -> np.ndarray:
        """
        Returns a list of node indices that INFLUENCE the target node_idx.
        Since edges point from Follower -> Influencer, the nodes that influence 
        node_idx are its SUCCESSORS (the nodes it has outgoing edges towards).
        """
        if self.graph is None:
            raise ValueError("Network has not been initialized. Call create_network() first.")
        
        # successsors(v) returns all nodes u where a directed edge exists from v to u (all nodes that influence v)
        return np.array(list(self.graph.successors(node_idx)), dtype=np.int32)