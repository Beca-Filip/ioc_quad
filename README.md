# Multi-Objective Quadratic Optimization & Inverse Optimal Control

A Python library for solving multi-objective quadratic optimization problems and their inverse variants. Built on CasADi for efficient nonlinear optimization.

## Installation

### From source

```bash
git clone https://github.com/Beca-Filip/ioc-quad.git
cd ioc-quad
pip install -e .
```

Or install just the dependencies:

```bash
pip install -r requirements.txt
```

### Requirements

- Python >= 3.9 (tested on 3.13.3)
- numpy >= 2.0.0
- casadi >= 3.6.0
- matplotlib >= 3.8.0
- alphashape >= 1.3.0

## What does it do?

This library lets you work with multiple quadratic objectives and find trade-offs between them. The main use cases are:

1. **Forward problem**: Given a weighted combination of objectives, find the optimal solution
2. **Inverse problem**: Given a reference solution, find the weights that would produce it

The inverse problem is particularly useful when you observe someone's behavior and want to understand their underlying preferences or cost function.

## Quick Start

### Forward Multi-Objective Optimization

```python
import numpy as np
from ioc_quad import MultiObjectiveOptimizer

# Create an optimizer for 2D problems with 3 objectives
optimizer = MultiObjectiveOptimizer(n_vars=2, n_objectives=3)

# Generate random quadratic objectives (or set your own)
optimizer.generate_random_objectives(rhorange=[1, 2], sigmarange=[0.5, 1.5])

# Solve for a specific weight vector (must sum to 1)
theta = np.array([[0.5], [0.3], [0.2]])
solution = optimizer.solve(theta)

# Compute the entire Pareto front
pareto_solutions = optimizer.compute_pareto_front(resolution=15)
```

### Inverse Optimal Control

```python
from ioc_quad import InverseOptimalControl

# Say you observed someone make this decision
reference_solution = np.array([[0.5], [-1.2]])

# Create the inverse problem solver
ioc = InverseOptimalControl(optimizer, reference_vector=reference_solution)

# Find the weights that best explain this behavior
optimal_theta, optimal_z, loss = ioc.solve_inverse()

print(f"Inferred preferences: {optimal_theta.T}")
print(f"Reconstruction error: {loss}")
```

## Core Classes

### `MultiObjectiveOptimizer`

Handles the forward optimization problem. Each objective is a quadratic function:

```
f_i(z) = 0.5 * z^T * Q_i * z + p_i^T * z
```

The optimizer solves:

```
min_z  Σ θ_i * f_i(z)
```

where θ is a probability vector (sums to 1, non-negative).

**Key methods:**
- `set_objectives(qlist, plist)` - Define your own Q matrices and p vectors
- `solve(theta)` - Solve for specific weights
- `compute_pareto_front(resolution)` - Sample the entire solution space
- `plot_individual_solutions()` - Visualize individual objective minima (2D only)
- `plot_pareto_front()` - Plot the Pareto front with alphashapes (2D only)

### `InverseOptimalControl`

Solves the inverse problem: given a reference solution, find the weights that would produce it.

This is formulated as a bilevel optimization:
- **Outer level**: Minimize distance between z and reference
- **Inner level**: z must satisfy KKT conditions (i.e., it's optimal for the weighted objectives)

**Key methods:**
- `solve_inverse()` - Simultaneously optimize for z and θ
- `ioc_loss(theta)` - Evaluate loss for a given weight vector
- `evaluate_loss_grid(resolution)` - Sample the loss landscape

**Distance metrics supported:**
- `'l2'` - Euclidean distance (default)
- `'l1'` - Manhattan distance
- `'weighted'` - Custom weighted distance

## Custom Objectives

You can define your own objectives instead of using random ones:

```python
Q1 = np.array([[2.0, 0.0], [0.0, 1.0]])
p1 = np.array([[-1.0], [0.5]])

Q2 = np.array([[1.0, 0.5], [0.5, 2.0]])
p2 = np.array([[0.0], [-1.0]])

optimizer.set_objectives([Q1, Q2], [p1, p2])
```

**Important**: All Q matrices must be positive definite. The library will check this automatically.

## Visualization

For 2D problems, you get built-in plotting:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# Plot individual solutions with their objective ellipses
optimizer.plot_individual_solutions(ax=ax, plot_ellipses=True)

# Plot the Pareto front as an alphashape
optimizer.plot_pareto_front(ax=ax, resolution=15, alpha=0.5)

plt.show()
```

## Running Examples

Check out the `examples/` directory for complete working examples:

```bash
# Basic multi-objective optimization and IOC demo
python examples/basic_example.py

# Test alphashape plotting
python examples/test_alphashape.py
```

## Tips and Gotchas

- **Initial guesses matter**: The inverse problem is non-convex. If you're getting weird results, try different initial guesses for theta and z.

- **Pareto resolution**: Higher resolution gives you more points but takes longer. Start with `resolution=10` and increase if needed.

- **Distance metrics**: For the inverse problem, `'l2'` works well in most cases. Use `'weighted'` if some dimensions are more important than others.

- **Solver convergence**: If IPOPT is struggling, you can pass custom solver options:
  ```python
  ioc.solve_inverse(solver_opts={'ipopt.max_iter': 5000, 'ipopt.tol': 1e-6})
  ```

## Example Use Cases

1. **Robotics**: Infer cost function weights from demonstrated trajectories
2. **Game theory**: Understand player preferences from observed strategies
3. **Economics**: Recover utility functions from consumer choices
4. **Control systems**: Tune controller objectives to match desired behavior

## Under the Hood

The forward problem is solved using CasADi's Opti interface with IPOPT. The inverse problem uses KKT conditions to enforce that z is optimal for the inferred weights:

```
∇_z (Σ θ_i * f_i(z)) = Σ θ_i * (Q_i * z + p_i) = 0
```

This constraint ensures consistency between the solution z and the weights θ.
