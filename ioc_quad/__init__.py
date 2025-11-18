"""
Multi-Objective Quadratic Optimization & Inverse Optimal Control

A library for solving multi-objective quadratic optimization problems and their inverse variants.
"""

from .core import (
    MultiObjectiveOptimizer,
    InverseOptimalControl,
    QuadraticObjective,
    simplex_grid,
    plot_ellipse,
    sample_random_quadratics,
    sample_positive_definite,
)

__version__ = "0.1.0"

__all__ = [
    "MultiObjectiveOptimizer",
    "InverseOptimalControl",
    "QuadraticObjective",
    "simplex_grid",
    "plot_ellipse",
    "sample_random_quadratics",
    "sample_positive_definite",
]
