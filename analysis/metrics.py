import numpy as np

def calculate_mean_network_distribution(p_matrix: np.ndarray) -> np.ndarray:
    """Computes the population-wide average probability vector M_t."""
    return np.mean(p_matrix, axis=0)

def calculate_global_entropy(p_matrix: np.ndarray) -> float:
    """
    Computes the network-wide mean Shannon entropy from agent probability vectors.
    Measures individual uncertainty levels across the system.
    """
    eps = 1e-12
    p_safe = np.clip(p_matrix, eps, 1.0)
    agent_entropies = -np.sum(p_safe * np.log(p_safe), axis=1)
    return float(np.mean(agent_entropies))

def calculate_polarization_index(p_matrix: np.ndarray) -> float:
    """
    Measures the variance of probability vectors across the network population.
    High value indicates that agents have split into opposing categorical beliefs.
    """
    mean_distribution = calculate_mean_network_distribution(p_matrix)
    squared_deviations = (p_matrix - mean_distribution) ** 2
    return float(np.mean(squared_deviations))

def calculate_dirichlet_population_divergence(j_matrix: np.ndarray) -> float:
    """
    Extracts the population-wide continuous scalar indicator from a 
    precomputed Jeffreys Divergence matrix. Used for deferred notebook analysis.
    """
    N = j_matrix.shape[0]
    if N <= 1:
        return 0.0
    
    # Enxtract entries from lower triangle
    row_indices, col_indices = np.tril_indices(N, k=-1)
    unique_pairwise_distances = j_matrix[row_indices, col_indices]
    return float(np.mean(unique_pairwise_distances))