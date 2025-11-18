import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import alphashape
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass


def simplex_grid(m, r):
    d = 0
    j = np.zeros((m, 1))
    sigma = 0
    grid = []
    grid = simplex_grid_rec(d, m, r, j, sigma, grid)
    return grid


def simplex_grid_rec(d, m, r, j, sigma, grid):
    if d >= m - 1:
        j[d] = (r-1) - sigma
        grid.append(j / (r-1))
        return grid

    for i in range(r - sigma):
        j[d] = i
        sigmai = sigma + i
        grid = simplex_grid_rec(d+1, m, r, j, sigmai, grid)
    return grid


def npmsqrt(M):
    U, S, Vh = np.linalg.svd(M)
    sqrtM = U * np.diag(np.sqrt(S)) * Vh
    return sqrtM


def plot_ellipse(M, c, ax=None, numpts=50, **kwargs):
    t = np.linspace(0, 2*np.pi, numpts).reshape(1, -1)
    circle = np.vstack((np.cos(t), np.sin(t)))
    sqrtM = npmsqrt(np.linalg.inv(M))
    ellipse = c + sqrtM @ circle
    if ax:
        ax.plot(ellipse[0, :], ellipse[1, :], **kwargs)
    plt.plot(ellipse[0, :], ellipse[1, :], **kwargs)


def sample_positive_definite(N, sigmarange=[0.5, 2]):
    ang = np.random.rand(1, N)
    phi1 = np.vstack((np.cos(ang), np.sin(ang)))
    phi2 = np.vstack((np.cos(ang+np.pi/2), np.sin(ang+np.pi/2)))
    sig = np.diff(sigmarange, n=1, axis=0) * np.random.rand(2, N) + sigmarange[0]
    qlist = []
    for i in range(N):
        Phi = np.hstack((phi1[:, [i]], phi2[:, [i]]))
        Sig = np.diag(sig[:, i])
        qlist.append(Phi @ Sig @ Phi.T)
    return qlist


def sample_random_quadratics(N, rhorange=[1, 2], sigmarange=[0.5, 1.5]):
    phi = 2 * np.pi * np.random.rand(1, N)
    rho = np.diff(rhorange, n=1, axis=0) * np.random.rand(1, N) + rhorange[0]
    xsol = rho * np.vstack((np.cos(phi), np.sin(phi)))
    qlist = sample_positive_definite(N, sigmarange=sigmarange)
    plist = [-qlist[i] @ xsol[:, [i]] for i in range(N)]
    return qlist, plist


def fact(n):
    if n > 1:
        return fact(n-1) * n
    return 1


def choose(m, n):
    if m > n:
        return fact(m) / (fact(m-n) * fact(n))
    else:
        raise ValueError("m must be > n.")


@dataclass
class QuadraticObjective:
    """
    Represents a single quadratic objective: f(z) = 0.5 * z^T Q z + p^T z
    """
    Q: np.ndarray  # Hessian matrix (n x n)
    p: np.ndarray  # Linear term (n x 1)

    def __post_init__(self):
        """Validate the objective parameters"""
        if self.Q.shape[0] != self.Q.shape[1]:
            raise ValueError("Q must be square")
        if self.p.shape[0] != self.Q.shape[0]:
            raise ValueError("p dimensions must match Q")
        if not self._is_positive_definite():
            raise ValueError("Q must be positive definite")

    def _is_positive_definite(self) -> bool:
        """Check if Q is positive definite"""
        try:
            eigenvalues = np.linalg.eigvals(self.Q)
            return np.all(eigenvalues > 0)
        except np.linalg.LinAlgError:
            return False

    def evaluate(self, z: np.ndarray) -> float:
        """Evaluate the objective at point z"""
        return float(0.5 * z.T @ self.Q @ z + self.p.T @ z)


class MultiObjectiveOptimizer:
    """
    Multi-objective quadratic optimization solver.
    Solves: min_z sum(theta_i * (0.5 * z^T Q_i z + p_i^T z))
    """

    def __init__(self, n_vars: int, n_objectives: int, solver_opts: Optional[dict] = None):
        """
        Initialize the optimizer.

        Args:
            n_vars: Dimension of decision variables
            n_objectives: Number of objectives
            solver_opts: Optional solver options for ipopt
        """
        self.n_vars = n_vars
        self.n_objectives = n_objectives
        self.solver_opts = solver_opts or {}

        self.objectives: List[QuadraticObjective] = []
        self._opti: Optional[ca.Opti] = None
        self._z = None
        self._theta = None
        self._cost = None
        self._initialized = False

    def generate_random_objectives(self, rhorange: List[float] = [1, 2],
                                   sigmarange: List[float] = [0.5, 1.5]) -> None:
        """
        Generate random quadratic objectives with known solutions.

        Args:
            rhorange: Range for solution magnitude
            sigmarange: Range for eigenvalues of Q matrices
        """
        qlist, plist = sample_random_quadratics(self.n_objectives, rhorange, sigmarange)
        self.set_objectives(qlist, plist)

    def set_objectives(self, qlist: List[np.ndarray], plist: List[np.ndarray]) -> None:
        """
        Set custom objectives.

        Args:
            qlist: List of Q matrices (each n_vars x n_vars)
            plist: List of p vectors (each n_vars x 1)
        """
        if len(qlist) != self.n_objectives or len(plist) != self.n_objectives:
            raise ValueError(f"Expected {self.n_objectives} objectives")

        self.objectives = [QuadraticObjective(Q=q, p=p) for q, p in zip(qlist, plist)]
        self._initialize_problem()

    def _initialize_problem(self) -> None:
        """Initialize the CasADi optimization problem (called once)"""
        self._opti = ca.Opti()
        self._z = self._opti.variable(self.n_vars, 1)
        self._theta = self._opti.parameter(self.n_objectives, 1)

        # Build the weighted sum of objectives
        costvec = ca.vertcat(*[
            (self._z.T @ obj.Q @ self._z / 2 + obj.p.T @ self._z)
            for obj in self.objectives
        ])
        self._cost = ca.sum1(self._theta * costvec)
        self._opti.minimize(self._cost)

        # Configure solver
        solver_opts = {'ipopt.print_level': 0, 'print_time': 0}
        solver_opts.update(self.solver_opts)
        self._opti.solver('ipopt', solver_opts)

        self._initialized = True

    def solve(self, theta: np.ndarray, initial_guess: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Solve the weighted multi-objective problem.

        Args:
            theta: Weight vector (n_objectives x 1), should sum to 1
            initial_guess: Optional initial guess for z (n_vars x 1)

        Returns:
            Optimal solution z (n_vars x 1)
        """
        if not self._initialized:
            raise RuntimeError("Objectives not set. Call set_objectives() or generate_random_objectives() first")

        if theta.shape != (self.n_objectives, 1):
            raise ValueError(f"theta must be ({self.n_objectives}, 1)")

        # Validate theta is on simplex (approximately)
        if not np.isclose(np.sum(theta), 1.0):
            raise ValueError("theta must sum to 1")

        self._opti.set_value(self._theta, theta)

        if initial_guess is None:
            initial_guess = np.zeros((self.n_vars, 1))
        self._opti.set_initial(self._z, initial_guess)

        try:
            sol = self._opti.solve()
            return np.array(sol.value(self._z)).reshape(self.n_vars, 1)
        except RuntimeError as e:
            raise RuntimeError(f"Solver failed: {e}")

    def compute_pareto_front(self, resolution: int = 4) -> np.ndarray:
        """
        Compute the Pareto front by solving for a grid of weight vectors.

        Args:
            resolution: Grid resolution (higher = more points)

        Returns:
            Array of solutions (n_vars x num_points)
        """
        if not self._initialized:
            raise RuntimeError("Objectives not set")

        thetagrid = simplex_grid(self.n_objectives, resolution)
        num_points = len(thetagrid)

        solutions = np.zeros((self.n_vars, num_points))
        z_prev = np.zeros((self.n_vars, 1))

        for i, theta in enumerate(thetagrid):
            solutions[:, [i]] = self.solve(theta, initial_guess=z_prev)
            z_prev = solutions[:, [i]]

        return solutions

    def get_objectives_list(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return the Q and p lists for plotting/analysis"""
        qlist = [obj.Q for obj in self.objectives]
        plist = [obj.p for obj in self.objectives]
        return qlist, plist

    def plot_individual_solutions(self,
                                  ax: Optional[plt.Axes] = None,
                                  plot_ellipses: bool = True,
                                  marker: str = 'ro',
                                  markersize: int = 25,
                                  label: str = 'Individual objectives',
                                  **kwargs) -> np.ndarray:
        """
        Solve and plot individual objective solutions (unit weight vectors).

        Args:
            ax: Matplotlib axes object (creates new if None)
            plot_ellipses: Whether to plot objective ellipses
            marker: Marker style for solutions
            markersize: Size of markers
            label: Label for legend
            **kwargs: Additional arguments passed to plot_ellipse

        Returns:
            Array of individual solutions (n_vars x n_objectives)
        """
        if not self._initialized:
            raise RuntimeError("Objectives not set")

        if self.n_vars != 2:
            raise ValueError("Plotting only supported for 2D problems")

        # Solve for each individual objective
        solutions = np.zeros((self.n_vars, self.n_objectives))
        for i in range(self.n_objectives):
            theta = np.zeros((self.n_objectives, 1))
            theta[i] = 1.0
            solutions[:, [i]] = self.solve(theta)

        # Create axes if not provided
        if ax is None:
            fig, ax = plt.subplots()

        # Plot solutions
        ax.plot(solutions[0, :], solutions[1, :], marker,
                markersize=markersize, label=label)

        # Plot ellipses if requested
        if plot_ellipses:
            qlist, _ = self.get_objectives_list()
            ellipse_kwargs = {'color': 'b', 'linewidth': 2}
            ellipse_kwargs.update(kwargs)
            for i in range(len(qlist)):
                plot_ellipse(qlist[i], solutions[:, [i]], ax, **ellipse_kwargs)

        return solutions

    def plot_pareto_front(self,
                         pareto_solutions: Optional[np.ndarray] = None,
                         resolution: int = 15,
                         alpha: float = 0.5,
                         ax: Optional[plt.Axes] = None,
                         plot_points: bool = True,
                         point_marker: str = 'k.',
                         point_size: int = 12,
                         point_label: str = 'Pareto front',
                         patch_color: str = 'blue',
                         patch_alpha: float = 0.2) -> np.ndarray:
        """
        Plot Pareto front solutions as an alphashape.

        Args:
            pareto_solutions: Precomputed Pareto front (n_vars x num_points).
                            If None, computes using resolution parameter.
            resolution: Grid resolution for computing Pareto front (ignored if pareto_solutions provided)
            alpha: Alpha parameter for alphashape (controls tightness of hull)
            ax: Matplotlib axes object (creates new if None)
            plot_points: Whether to plot the individual points
            point_marker: Marker style for Pareto points
            point_size: Size of point markers
            point_label: Label for Pareto points in legend
            patch_color: Color of the alphashape patch
            patch_alpha: Transparency of the alphashape patch

        Returns:
            Array of Pareto front solutions (n_vars x num_points)
        """
        if not self._initialized:
            raise RuntimeError("Objectives not set")

        if self.n_vars != 2:
            raise ValueError("Plotting only supported for 2D problems")

        # Compute Pareto front if not provided
        if pareto_solutions is None:
            pareto_solutions = self.compute_pareto_front(resolution=resolution)

        # Create axes if not provided
        if ax is None:
            fig, ax = plt.subplots()

        # Convert to points array for alphashape (num_points x 2)
        points = pareto_solutions.T

        # Create and plot alphashape
        alpha_shape = alphashape.alphashape(points, alpha)
        coords = np.array(alpha_shape.exterior.coords)

        ax.add_patch(Polygon(coords, alpha=patch_alpha))
        
        # Plot points if requested
        if plot_points:
            ax.scatter(points[:, 0], points[:, 1],
                      marker=point_marker[1] if len(point_marker) > 1 else 'o',
                      c=point_marker[0] if len(point_marker) > 1 else 'k',
                      s=point_size,
                      label=point_label)

        return pareto_solutions


class InverseOptimalControl:
    """
    Inverse Optimal Control for multi-objective quadratic problems.
    Computes loss between reference vector and solution from given weights.
    """

    def __init__(self,
                 optimizer: MultiObjectiveOptimizer,
                 reference_vector: np.ndarray,
                 distance_metric: str = 'l2'):
        """
        Initialize IOC solver.

        Args:
            optimizer: The forward optimizer to use
            reference_vector: Target solution (n_vars x 1)
            distance_metric: Distance metric ('l2', 'l1', 'weighted')
        """
        self.optimizer = optimizer
        self.reference_vector = reference_vector
        self.distance_metric = distance_metric

        if reference_vector.shape[0] != optimizer.n_vars:
            raise ValueError("Reference vector dimensions must match optimizer")

        # For weighted distance (can be customized)
        self.distance_weights = np.ones((optimizer.n_vars, 1))

    def set_optimizer(self, optimizer: MultiObjectiveOptimizer) -> None:
        """Swap in a different optimizer"""
        if optimizer.n_vars != self.reference_vector.shape[0]:
            raise ValueError("New optimizer must have same n_vars")
        self.optimizer = optimizer

    def set_distance_weights(self, weights: np.ndarray) -> None:
        """Set custom weights for weighted distance metric"""
        if weights.shape != (self.optimizer.n_vars, 1):
            raise ValueError(f"Weights must be ({self.optimizer.n_vars}, 1)")
        self.distance_weights = weights

    def _compute_distance(self, solution: np.ndarray) -> float:
        """Compute distance between solution and reference"""
        diff = solution - self.reference_vector

        if self.distance_metric == 'l2':
            return float(np.linalg.norm(diff))
        elif self.distance_metric == 'l1':
            return float(np.sum(np.abs(diff)))
        elif self.distance_metric == 'weighted':
            return float(np.sqrt(np.sum((self.distance_weights * diff) ** 2)))
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def ioc_loss(self, theta: np.ndarray, initial_guess: Optional[np.ndarray] = None) -> float:
        """
        Compute IOC loss: solve with given weights and compute distance to reference.

        Args:
            theta: Weight vector (n_objectives x 1)
            initial_guess: Optional initial guess for solver

        Returns:
            Distance from solution to reference vector
        """
        solution = self.optimizer.solve(theta, initial_guess=initial_guess)
        return self._compute_distance(solution)

    def evaluate_loss_grid(self, resolution: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate IOC loss over a grid of theta values.

        Args:
            resolution: Grid resolution

        Returns:
            Tuple of (theta_grid, losses)
        """
        thetagrid = simplex_grid(self.optimizer.n_objectives, resolution)
        losses = np.array([self.ioc_loss(theta) for theta in thetagrid])
        return thetagrid, losses

    def solve_inverse(self,
                     initial_theta: Optional[np.ndarray] = None,
                     initial_z: Optional[np.ndarray] = None,
                     solver_opts: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solve the inverse optimal control problem by simultaneously optimizing for z and theta.

        This creates a bilevel optimization where:
        - z and theta are both decision variables
        - theta must lie on the simplex (sum to 1, all >= 0)
        - z must satisfy KKT optimality conditions for the forward problem given theta
        - We minimize the distance between z and the reference vector

        Args:
            initial_theta: Initial guess for theta (n_objectives x 1)
            initial_z: Initial guess for z (n_vars x 1)
            solver_opts: Optional solver options for ipopt

        Returns:
            Tuple of (optimal_theta, optimal_z, final_loss)
        """
        if not self.optimizer._initialized:
            raise RuntimeError("Optimizer objectives not set")

        # Create new optimization problem
        opti = ca.Opti()

        # Decision variables
        z = opti.variable(self.optimizer.n_vars, 1)
        theta = opti.variable(self.optimizer.n_objectives, 1)

        # Simplex constraints on theta
        opti.subject_to(ca.sum1(theta) == 1.0)
        opti.subject_to(theta >= 0)

        # Build KKT optimality condition: gradient of weighted objective should be zero
        # ∇_z (sum_i theta_i * f_i(z)) = sum_i theta_i * (Q_i z + p_i) = 0
        gradient = 0
        for i, obj in enumerate(self.optimizer.objectives):
            gradient += theta[i] * (obj.Q @ z + obj.p)

        # Add KKT condition as equality constraint
        opti.subject_to(gradient == 0)

        # Objective: minimize distance to reference
        diff = z - self.reference_vector

        if self.distance_metric == 'l2':
            cost = ca.sumsqr(diff)
        elif self.distance_metric == 'l1':
            cost = ca.sum1(ca.fabs(diff))
        elif self.distance_metric == 'weighted':
            cost = ca.sumsqr(ca.sqrt(self.distance_weights) * diff)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        opti.minimize(cost)

        # Set initial guesses
        if initial_theta is None:
            initial_theta = np.ones((self.optimizer.n_objectives, 1)) / self.optimizer.n_objectives
        if initial_z is None:
            initial_z = self.reference_vector

        opti.set_initial(theta, initial_theta)
        opti.set_initial(z, initial_z)

        # Configure solver
        default_solver_opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 1000,
            'ipopt.tol': 1e-8
        }
        if solver_opts:
            default_solver_opts.update(solver_opts)

        opti.solver('ipopt', default_solver_opts)

        # Solve
        try:
            sol = opti.solve()
            optimal_theta = np.array(sol.value(theta)).reshape(self.optimizer.n_objectives, 1)
            optimal_z = np.array(sol.value(z)).reshape(self.optimizer.n_vars, 1)
            final_loss = float(sol.value(cost))

            return optimal_theta, optimal_z, final_loss
        except RuntimeError as e:
            raise RuntimeError(f"Inverse optimization solver failed: {e}")
