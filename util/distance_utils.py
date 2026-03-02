import numpy as np
from math import radians, cos, sin, asin, sqrt

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two lat/lon points (unit: km)
    Using Haversine formula
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth radius in km
    
    return c * r

def compute_distance_matrix(coords):
    """
    Compute distance matrix
    coords: (N, 2) array, each row is [x, y] or [lat, lon]
    Returns: (N, N) distance matrix (unit: km)
    """
    N = coords.shape[0]
    dist_matrix = np.zeros((N, N), dtype=np.float32)
    
    for i in range(N):
        for j in range(i+1, N):
            # Assume coords are [x, y] in meters, convert to km
            # If lat/lon, use haversine_distance
            dist = np.sqrt((coords[i, 0] - coords[j, 0])**2 + 
                          (coords[i, 1] - coords[j, 1])**2) / 1000.0
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    return dist_matrix

def distance_to_weight(dist_matrix, sigma_km=30.0):
    """
    Convert distance matrix to weight matrix
    Using Gaussian kernel: w_ij = exp(-d_ij^2 / (2 * sigma^2))
    
    dist_matrix: (N, N) distance matrix
    sigma_km: Standard deviation of Gaussian kernel (unit: km)
    Returns: (N, N) weight matrix, range (0, 1]
    """
    weight_matrix = np.exp(-dist_matrix**2 / (2 * sigma_km**2))
    return weight_matrix.astype(np.float32)
