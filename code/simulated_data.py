import numpy as np

def simulate_data(n_samples, n_features, n_tasks, nnz, snr):
    """Generates a simulated row-sparse weight 
       matrix for multi-task lasso

    Args:
        n_samples (int): number of samples
        n_features (int): number of features
        n_tasks (int): [description]
        nnz (int): number of non-zero coefficients
        snr (float): signal-to-noise ratio

    Returns:
        X (np.ndarray): design matrix (n_samples, n_features)
        Y (np.ndarray): (n_samples, n_tasks)
        W (np.ndarray): (n_features, n_tasks) 
    """

    return