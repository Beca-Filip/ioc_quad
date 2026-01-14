"""Plotting utility functions for IOC quadratic optimization."""

import numpy as np
import matplotlib.pyplot as plt
from .math_utils import matrix_sqrt


def plot_ellipse(M, c, ax=None, numpts=50, **kwargs):
    """
    Plot a 2D ellipse defined by x^T M x = 1 centered at c.

    Parameters
    ----------
    M : ndarray
        2x2 positive definite matrix defining the ellipse.
    c : ndarray
        2x1 center point.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object. If provided, plots on this axes.
    numpts : int, optional
        Number of points for drawing the ellipse. Default is 50.
    **kwargs
        Additional plotting arguments passed to plt.plot.
    """
    t = np.linspace(0, 2*np.pi, numpts).reshape(1, -1)
    circle = np.vstack((np.cos(t), np.sin(t)))
    sqrtM = matrix_sqrt(np.linalg.inv(M))
    ellipse = c + sqrtM @ circle
    if ax:
        ax.plot(ellipse[0, :], ellipse[1, :], **kwargs)
    plt.plot(ellipse[0, :], ellipse[1, :], **kwargs)


def plot_ellipsoid(M, c, ax=None, numpts=25, **kwargs):
    """
    Plot a 3D ellipsoid defined by x^T M x = 1 centered at c.

    Args:
        M: 3x3 positive definite matrix defining the ellipsoid
        c: 3x1 center point
        ax: Matplotlib 3D axes (if None, uses current axes)
        numpts: Number of points for mesh grid
        **kwargs: Additional plotting arguments (color, alpha, etc.)
    """
    # Create parametric sphere
    u = np.linspace(0, 2*np.pi, numpts)
    v = np.linspace(0, np.pi, numpts)
    U, V = np.meshgrid(u, v)

    # Unit sphere in spherical coordinates
    sphere_x = np.cos(U) * np.sin(V)
    sphere_y = np.sin(U) * np.sin(V)
    sphere_z = np.cos(V)

    # Stack into shape (3, numpts*numpts)
    sphere = np.vstack([
        sphere_x.ravel(),
        sphere_y.ravel(),
        sphere_z.ravel()
    ])

    # Transform sphere to ellipsoid using sqrt(M^-1)
    inv_sqrt_M = matrix_sqrt(np.linalg.inv(M))
    ellipsoid = c + inv_sqrt_M @ sphere

    # Reshape back to grid
    X = ellipsoid[0, :].reshape(numpts, numpts)
    Y = ellipsoid[1, :].reshape(numpts, numpts)
    Z = ellipsoid[2, :].reshape(numpts, numpts)

    # Plot surface
    if ax is None:
        ax = plt.gcf().add_subplot(projection="3d")

    ax.plot_surface(X, Y, Z, **kwargs)

    return ax

