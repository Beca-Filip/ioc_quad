"""Mathematical utility functions for IOC quadratic optimization."""

import numpy as np
from typing import List


def simplex_grid(m, r):
    """
    Generate a uniform grid of points on the (m-1)-dimensional probability simplex.

    Parameters
    ----------
    m : int
        Dimension of the simplex (number of objectives). The simplex will be
        (m-1)-dimensional embedded in R^m.
    r : int
        Resolution parameter controlling grid density. The grid will have
        (r-1) divisions along each dimension, resulting in approximately
        C(m+r-2, m-1) points.

    Returns
    -------
    list of ndarray
        List of points on the simplex. Each point is an (m, 1) array with
        non-negative entries that sum to 1.

    Notes
    -----
    The simplex is defined as: {w in R^m : w_i >= 0, sum(w_i) = 1}.
    For r=2, returns only the vertices of the simplex.
    For r=3, adds midpoints of edges.
    Higher r values create denser grids.

    Examples
    --------
    >>> grid = simplex_grid(3, 3)  # 3D simplex with resolution 3
    >>> len(grid)
    6
    """
    d = 0
    j = np.zeros((m, 1))
    sigma = 0
    grid = []
    grid = simplex_grid_rec(d, m, r, j, sigma, grid)
    return grid


def simplex_grid_rec(d, m, r, j, sigma, grid):
    """
    Recursive helper function for generating simplex grid points.

    Parameters
    ----------
    d : int
        Current dimension being processed (0 to m-1).
    m : int
        Total number of dimensions (simplex dimension).
    r : int
        Resolution parameter.
    j : ndarray
        Current point being constructed, shape (m, 1).
    sigma : int
        Sum of coordinates assigned so far.
    grid : list
        Accumulator for grid points.

    Returns
    -------
    list of ndarray
        Updated grid with new points added.

    Notes
    -----
    This function recursively generates integer lattice points that satisfy
    the constraint sum(j_i) = r-1, then normalizes by (r-1) to get simplex
    points. The algorithm uses a combinatorial enumeration strategy.
    """
    if d >= m - 1:
        j[d] = (r-1) - sigma
        grid.append(j / (r-1))
        return grid

    for i in range(r - sigma):
        j[d] = i
        sigmai = sigma + i
        grid = simplex_grid_rec(d+1, m, r, j, sigmai, grid)
    return grid


def matrix_sqrt(M):
    """
    Compute the matrix square root using SVD.

    Parameters
    ----------
    M : ndarray
        Input matrix.

    Returns
    -------
    ndarray
        Matrix square root of M.
    """
    U, S, Vh = np.linalg.svd(M)
    sqrtM = U * np.diag(np.sqrt(S)) * Vh
    return sqrtM


def random_psd(n, n_samples, lambda_range=[0.5, 2]):
    """
    Generate random symmetric positive-definite matrices.

    Parameters
    ----------
    n : int
        Dimension of each matrix (n x n).
    n_samples : int
        Number of matrices to generate.
    lambda_range : list of float, optional
        Range [min, max] for eigenvalue sampling. Default is [0.5, 2].

    Returns
    -------
    list of ndarray
        List of n_samples PSD matrices, each of shape (n, n).

    Notes
    -----
    Each matrix is constructed as Q = V Lambda V^T where V is an orthonormal
    basis from QR decomposition and Lambda has eigenvalues drawn uniformly
    from lambda_range.
    """
    # Sample n_samples stacks of n Gaussian n-vectors (shape: n_samples x n x n)
    v = np.random.randn(n_samples, n, n)
    # Orthonormal basis per stack via QR; discard R
    V, _ = np.linalg.qr(v)
    V_t = np.matrix_transpose(V)
    # Random eigenvalues in lambda_range for each matrix (shape: n_samples x 1 x n)
    lam = np.diff(lambda_range, n=1, axis=0) * np.random.rand(n_samples, 1, n) + lambda_range[0]
    # Scale basis vectors by eigenvalues, then reconstruct Q = V Lambda V^T
    V_lam = V * np.tile(lam, (1, n, 1))
    Q = np.matmul(V_lam, V_t)
    return list(Q)


def random_shell(n, n_samples, rho_range=[1, 2]):
    """
    Sample points uniformly from an n-dimensional spherical shell.

    Generates points with uniformly random directions and uniformly random
    radii within the specified range. The direction is sampled uniformly on
    the unit sphere using normalized Gaussian vectors.

    Parameters
    ----------
    n : int
        Dimension of the space.
    n_samples : int
        Number of points to generate.
    rho_range : list of float, optional
        Range [min, max] for point radii. Default is [1, 2].

    Returns
    -------
    ndarray
        Array of sampled points with shape (n_samples, n, 1).

    Notes
    -----
    This samples uniformly in direction and radius, which differs from
    volume-uniform sampling on a spherical shell (which would weight by r^(n-1)).
    For generating random optimal points in quadratic optimization, this
    uniform directional sampling is typically preferred.

    Examples
    --------
    >>> points = random_shell(3, 100, rho_range=[1.0, 2.0])
    >>> points.shape
    (100, 3, 1)
    """
    # Sample n_samples Gaussian n-vectors then normalize each to unit length
    x_star = np.random.randn(n_samples, n, 1)
    x_star /= np.sqrt(np.sum(x_star**2, axis=1, keepdims=True))
    # Assign a random radius in rho_range to each vector
    rho = np.diff(rho_range, n=1, axis=0) * np.random.rand(n_samples, 1, 1) + rho_range[0]
    x_star *= rho
    return x_star


def random_quadfun(n, n_samples, rho_range=[1, 2], lambda_range=[0.5, 1.5]):
    """
    Sample quadratic functions with PSD Hessian matrices and random optima.

    Parameters
    ----------
    n : int
        Dimension of the input space.
    n_samples : int
        Number of quadratic functions to generate.
    rho_range : list of float, optional
        Range [min, max] for optimal point radius. Default is [1, 2].
    lambda_range : list of float, optional
        Range [min, max] for eigenvalue sampling. Default is [0.5, 1.5].

    Returns
    -------
    q_list : list of ndarray
        List of n_samples PSD matrices Q, each of shape (n, n).
    p_list : list of ndarray
        List of n_samples linear coefficients p, each of shape (n, 1).

    Notes
    -----
    Each function has the form f(x) = 0.5 x^T Q x + p^T x where Q is PSD
    and p = -Q x* with x* being a random point sampled from a spherical shell
    with radii in rho_range. The optimal point x* is sampled uniformly in
    direction and radius using random_shell().
    """
    # Sample random optimal points from spherical shell
    x_star = random_shell(n=n, n_samples=n_samples, rho_range=rho_range)
    # Random PSD matrices and corresponding p vectors
    q_list = random_psd(n, n_samples, lambda_range)
    p_list = [-q_list[i] @ x_star[i] for i in range(len(q_list))]
    return q_list, p_list


def fact(n):
    """
    Compute factorial of n.

    Parameters
    ----------
    n : int
        Non-negative integer.

    Returns
    -------
    int
        Factorial of n.
    """
    if n > 1:
        return fact(n-1) * n
    return 1


def choose(m, n):
    """
    Compute binomial coefficient C(m, n).

    Parameters
    ----------
    m : int
        Number of items.
    n : int
        Number of selections.

    Returns
    -------
    float
        Binomial coefficient.

    Raises
    ------
    ValueError
        If m is not greater than n.
    """
    if m > n:
        return fact(m) / (fact(m-n) * fact(n))
    else:
        raise ValueError("m must be > n.")