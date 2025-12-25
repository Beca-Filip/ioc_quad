"""
Unit tests for the math_utils module.

This module contains tests for mathematical utility functions including
positive semi-definite matrix generation.
"""

import unittest
import numpy as np
from ioc_quad.math_utils import random_psd, matrix_sqrt


class TestMathUtils(unittest.TestCase):
    """Test case for mathematical utility functions in ioc_quad.math_utils."""

    def test_random_psd(self):
        """
        Test the random_psd function for generating positive semi-definite matrices.

        This test verifies that:
        - Generated matrices have the correct shape (n x n)
        - Matrices are symmetric (A = A^T)
        - All eigenvalues fall within the specified lambda_range
        """
        n_matrix_samples = 5
        n_lambda = 5
        lambda_values = np.logspace(-2, 2, n_lambda)
        for n in range(2, 100):
            for i_lambda in range(n_lambda-1):
                for j_lambda in range(i_lambda+1, n_lambda):

                    lambda_low = lambda_values[i_lambda]
                    lambda_high = lambda_values[j_lambda]
                    lambda_range = [lambda_low, lambda_high]

                    psd_matrices = random_psd(n=n, n_samples=n_matrix_samples, lambda_range=lambda_range)

                    for i in range(n_matrix_samples):
                        psd_matrix = psd_matrices[i]
                        # Test shape
                        self.assertEqual(
                            psd_matrix.shape,
                            (n, n),
                            f"Matrix {i} has incorrect shape: {psd_matrix.shape}"
                        )

                        # Test symmetry
                        self.assertTrue(
                            np.allclose(psd_matrix, psd_matrix.T, atol=1e-12),
                            f"Matrix {i} is not symmetric"
                        )

                        # Test eigenvalues are within range
                        eig_vals = np.linalg.eigvals(psd_matrix)
                        eig_vals = eig_vals.flatten()
                        for j, eig_val in enumerate(eig_vals):
                            self.assertGreaterEqual(
                                eig_val,
                                lambda_range[0],
                                f"Eigenvalue {j} of matrix {i} is outside of the lambda_range interval [{lambda_range[0]}, {lambda_range[1]}] -- below lower bound: {eig_val} < {lambda_range[0]}"
                            )
                            self.assertLessEqual(
                                eig_val,
                                lambda_range[1],
                                f"Eigenvalue {j} of matrix {i} is outside of the lambda_range interval [{lambda_range[0]}, {lambda_range[1]}] -- above upper bound: {eig_val} > {lambda_range[1]}"
                            )

    
    def test_matrix_sqrt(self):
        """Test the matrix_sqrt function for computing matrix square roots."""

        n_matrix_samples = 5
        n_lambda = 5
        lambda_values = np.logspace(-2, 2, n_lambda)

        for n in range(2, 100):
            for i_lambda in range(n_lambda-1):
                for j_lambda in range(i_lambda+1, n_lambda):

                    lambda_low = lambda_values[i_lambda]
                    lambda_high = lambda_values[j_lambda]
                    lambda_range = [lambda_low, lambda_high]

                    psd_matrices = random_psd(n=n, n_samples=n_matrix_samples, lambda_range=lambda_range)

                    for i in range(n_matrix_samples):
                        psd_matrix = psd_matrices[i]                       
                        sqrt_psd_matrix = matrix_sqrt(psd_matrix)

                        # Test shape
                        self.assertEqual(
                            sqrt_psd_matrix.shape,
                            (n, n),
                            f"Matrix {i} has incorrect shape: {sqrt_psd_matrix.shape}"
                        )
                        
                        diff_matrix = sqrt_psd_matrix.T @ sqrt_psd_matrix - psd_matrix

                        # Test algebraic soundness
                        self.assertTrue(
                            np.allclose(diff_matrix, 0, atol=1e-12),
                            f"The square of a matrix' square-root is not equal to the matrix itself by: {np.max(diff_matrix.flatten())}."
                        )


def main():
    """Run the test suite."""
    unittest.main()


if __name__ == "__main__":
    main()
