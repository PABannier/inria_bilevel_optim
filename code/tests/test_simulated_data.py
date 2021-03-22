import unittest 
from simulated_data import simulate_data
import numpy as np


class TestSyntheticData(unittest.TestCase):
    def test_nnz():
        nnz = 40
        _, _, W = simulate_data(nnz=nnz)
        true_nnz = np.count_nonzero(W, axis=0)

        np.testing.assert_equal(true_nnz, [40] * W.shape[1])