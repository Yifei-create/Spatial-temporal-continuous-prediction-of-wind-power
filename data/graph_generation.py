import numpy as np
import networkx as nx

def generate_adjacency_matrix(coords, sigma_km=30.0, top_k=16, self_loop=False):
    """
    Generate weighted adjacency matrix
    
    coords: (N, 2) array, each row is [x, y] coordinates
    sigma_km: Standard deviation of Gaussian kernel (km)
    top_k: Keep top-k nearest neighbors for each node
    self_loop: Whether to keep self-loops
    
    Returns: (N, N) adjacency matrix
    """
    from util.distance_utils import compute_distance_matrix, distance_to_weight
    
    N = coords.shape[0]
    
    # Compute distance matrix
    dist_matrix = compute_distance_matrix(coords)
    
    # Convert to weight matrix
    weight_matrix = distance_to_weight(dist_matrix, sigma_km=sigma_km)
    
    # Top-K sparsification
    if top_k is not None and top_k > 0:
        adj = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            # Find top-k nearest nodes (excluding itself)
            distances = dist_matrix[i].copy()
            distances[i] = np.inf  # Exclude itself
            nearest_indices = np.argsort(distances)[:top_k]
            adj[i, nearest_indices] = weight_matrix[i, nearest_indices]
    else:
        adj = weight_matrix
    
    # Symmetrize (take maximum)
    adj = np.maximum(adj, adj.T)
    
    # Self-loop
    if not self_loop:
        np.fill_diagonal(adj, 0.0)
    else:
        np.fill_diagonal(adj, 1.0)

    return adj.astype(np.float32)

def adjacency_to_networkx(adj):
    """Convert adjacency matrix to NetworkX graph"""
    return nx.from_numpy_array(adj)
