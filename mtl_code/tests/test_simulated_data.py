from mtl_code.simulated_data import simulate_data
import numpy as np


def test_nnz():
    """Tests that the weight matrix is row-sparse
    with the expected number of non-zero coefficients"""
    nnz = 40
    _, _, W = simulate_data(nnz=nnz)
    non_zeros = np.count_nonzero(W, axis=0)

    np.testing.assert_equal(non_zeros, [nnz] * W.shape[1])


def test_noise():
    """Tests that enough noise has been generated"""
    X, Y, W = simulate_data()
    assert not np.allclose(X @ W, Y, atol=1e-3)
