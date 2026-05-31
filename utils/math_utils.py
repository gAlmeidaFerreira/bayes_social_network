import numpy as np
from scipy.special import digamma, gammaln

def sigmoid(x, beta, tau):
    return 1 / (1 + np.exp(-beta * (x - tau)))

def compute_jeffreys_divergence_matrix(alpha_matrix: np.ndarray) -> np.ndarray:
    """
    Computes the full N x N symmetric Jeffreys Divergence matrix across all agents
    
    Args:
        alpha_matrix: NumPy array of shape (N, K) containing the Dirichlet 
                     concentration parameters (alphas) for N agents across K opinions.
    Returns:
        np.ndarray: A symmetric float matrix of shape (N, N) where entry [i, j] 
                    is the Jeffreys Divergence between agent i and agent j.
    """
    N, K = alpha_matrix.shape
    
    # Compute row sums of concentration parameters: shape (N, 1)
    sum_alpha = np.sum(alpha_matrix, axis=1, keepdims=True)
    
    # Precompute log_gamma terms for equations
    ln_gamma_sum = gammaln(sum_alpha)           # Shape: (N, 1)
    ln_gamma_alpha = gammaln(alpha_matrix)      # Shape: (N, K)
    
    # Precomputr digamma term differences: shape (N, K)
    digamma_diff = digamma(alpha_matrix) - digamma(sum_alpha)
    
    # Use broadcasting to expand matrices to 3D for pairwise combinations:
    # Dimension mapping: (i, j, k) where i = agent_i, j = agent_j, k = opinion_k
    alpha_i = alpha_matrix[:, np.newaxis, :]       # Shape: (N, 1, K)
    alpha_j = alpha_matrix[np.newaxis, :, :]       # Shape: (1, N, K)
    
    ln_gamma_alpha_i = ln_gamma_alpha[:, np.newaxis, :]  # Shape: (N, 1, K)
    ln_gamma_alpha_j = ln_gamma_alpha[np.newaxis, :, :]  # Shape: (1, N, K)
    
    digamma_diff_i = digamma_diff[:, np.newaxis, :]      # Shape: (N, 1, K)
    digamma_diff_j = digamma_diff[np.newaxis, :, :]      # Shape: (1, N, K)
    
    ln_gamma_sum_i = ln_gamma_sum[:, np.newaxis]         # Shape: (N, 1, 1)
    ln_gamma_sum_j = ln_gamma_sum[np.newaxis, :]         # Shape: (1, N, 1)
    
    # Calculate the asymmetrical KL divergence matrix
    kl_ij = ((ln_gamma_sum_i - ln_gamma_sum_j).squeeze(-1)
             - np.sum(ln_gamma_alpha_i - ln_gamma_alpha_j, axis=2)
             + np.sum((alpha_i - alpha_j) * digamma_diff_i, axis=2))
    
    # Symmetrize by adding the transpose
    j_matrix = kl_ij + kl_ij.T
    np.fill_diagonal(j_matrix, 0.0)
    
    return j_matrix