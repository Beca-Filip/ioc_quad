"""
Multi-Objective Quadratic Optimization & Inverse Optimal Control

A library for solving multi-objective quadratic optimization problems and their inverse variants.
"""

from .core import (
    MultiObjectiveOptimizer,
    InverseOptimalControl,
    QuadraticObjective,
    MaximumEntropyIRL_ParetoSamples_Casadi,
    MaximumEntropyIRL_ParetoSamples_Scipy, 
    MaximumEntropyIRL_withoutParetoSamples
)

from .math_utils import (
    simplex_grid,
    simplex_grid_rec,
    matrix_sqrt,
    random_psd,
    random_shell,
    random_quadfun,
    fact,
    choose,
)

from .plot_utils import (
    plot_ellipse,
    plot_ellipsoid,
)

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "MultiObjectiveOptimizer",
    "InverseOptimalControl",
    "QuadraticObjective",
    # Math utilities
    "simplex_grid",
    "simplex_grid_rec",
    "matrix_sqrt",
    "random_psd",
    "random_shell",
    "random_quadfun",
    "fact",
    "choose",
    # Plot utilities
    "plot_ellipse",
    "plot_ellipsoid",
]
