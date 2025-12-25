"""
Basic example demonstrating the multi-objective optimizer and inverse optimal control.
"""

import numpy as np
import matplotlib.pyplot as plt
from ioc_quad import MultiObjectiveOptimizer, InverseOptimalControl

np.random.seed(42)

def main():
    """Demonstration of forward and inverse optimal control"""
    n = 3  # Number of decision variables
    m = 4  # Number of objectives

    # Create optimizer and generate random objectives
    optimizer = MultiObjectiveOptimizer(n_vars=n, n_objectives=m)
    optimizer.generate_random_objectives(rho_range=[4, 10], lambda_range=[0.5, 1.5])

    # Create figure
    if n == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
    if n == 3:
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection":"3d"})

    # Plot individual objective solutions with ellipses
    zcost = optimizer.plot_individual_solutions(ax=ax, plot_levelsets=True)

    # Plot Pareto front as alphashape
    zpareto = optimizer.plot_pareto_solutions(ax=ax, resolution=15, alpha=2)

    # Find pareto centroid and define step-size away from centroid as std
    centroid_pareto = np.mean(zpareto, axis=1, keepdims=True)
    step_size = np.std(zpareto, axis=1, keepdims=True)

    # Demonstrate IOC with forward loss evaluation
    reference = zcost[:, [0]] + (zcost[:, [0]] - centroid_pareto) * step_size # Use first objective solution as reference, moved-away from centroid by std
    ioc = InverseOptimalControl(optimizer, reference_vector=reference, distance_metric='l2')

    # Compute loss for a test theta
    # test_theta = np.array([[0.5], [0.3], [0.2]])
    test_theta = np.random.rand(m, 1)
    test_theta /= np.sum(test_theta)
    loss = ioc.ioc_loss(test_theta)
    print(f"IOC loss for theta={test_theta.T}: {loss:.4f}")

    # Solve inverse problem to find optimal theta
    print("\nSolving inverse optimal control problem...")
    optimal_theta, optimal_z, final_loss = ioc.solve_inverse()

    print(f"Optimal theta: {optimal_theta.T}")
    print(f"Optimal z: {optimal_z.T}")
    print(f"Final loss: {final_loss:.6f}")
    print(f"Distance to reference: {np.linalg.norm(optimal_z - reference):.6f}")

    # Verify: solve forward problem with optimal theta
    z_forward = optimizer.solve(optimal_theta)
    print(f"Forward solution with optimal theta: {z_forward.T}")
    print(f"Difference between inverse z and forward z: {np.linalg.norm(optimal_z - z_forward):.6e}")

    # Plot reference and recovered solution
    ax.plot(reference[0], reference[1], 'g*', markersize=20, label='Reference')
    ax.plot(optimal_z[0], optimal_z[1], 'b^', markersize=15, label='Recovered (inverse)')

    ax.set_xlabel('z₁')
    ax.set_ylabel('z₂')
    ax.set_title('Multi-Objective Optimization and Inverse Optimal Control')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
