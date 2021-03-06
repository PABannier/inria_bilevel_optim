from mtl.simulated_data import simulate_data
import numpy as np


def test_nnz():
    """Tests that the weight matrix is row-sparse
    with the expected number of non-zero coefficients"""
    nnz = 40
    _, _, W, _ = simulate_data(nnz=nnz)
    non_zeros = np.count_nonzero(W, axis=0)

    np.testing.assert_equal(non_zeros, [nnz] * W.shape[1])
