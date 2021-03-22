import numpy as np

def simulate_data(n_samples=100, n_features=10000, n_tasks=100, nnz=10, snr=3,
                  mean_X=.0, std_X=1.5, mean_W=.0, std_W=.5):
    """Generates a simulated dataset and a row-sparse weight 
       matrix for multi-task lasso.

       As a reminder (in neuroscience inverse problem):
        n_samples: number of sensors on the scalp
        n_features: number of discretized areas in the brain 
        n_tasks: time series length of the brain recording

    Args:
        n_samples (int): number of samples
        n_features (int): number of features
        n_tasks (int): number of tasks
        nnz (int): number of non-zero coefficients
        snr (float): signal-to-noise ratio

    Returns:
        X (np.ndarray): design matrix (n_samples, n_features)
        Y (np.ndarray): (n_samples, n_tasks)
        W (np.ndarray): (n_features, n_tasks) 
    """

    if nnz > n_features:
        raise ValueError("Number of non-zero coefficients can't be greater than the number of features")

    X = np.random.normal(loc=mean_X, scale=std_X, size=(n_samples, n_features))

    dense_W = np.random.normal(loc=mean_W, scale=std_W, size=(n_features, n_tasks))
    sparse_mat = np.zeros_like(dense_W)

    for i in range(n_tasks):
        col = np.array([1] * nnz + [0] * (n_features - nnz)).T
        np.random.shuffle(col)
        sparse_mat[:, i] = col
    
    W = dense_W * sparse_mat
    Y = X @ W

    # Adding a pre-defined SNR to a signal: https://sites.ualberta.ca/~msacchi/SNR_Def.pdf
    noise = np.random.normal(size=(n_samples, n_tasks))
    alpha = np.sqrt((np.linalg.norm(Y, axis=0) ** 2) / (snr * np.linalg.norm(noise, axis=0) ** 2))
    Y += alpha * noise

    return X, Y, W