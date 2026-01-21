import numpy as np
import casadi as ca
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import Slider
import alphashape
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
from shapely.geometry import Polygon as ShapelyPolygon
from scipy.optimize import minimize

from .math_utils import simplex_grid, random_quadfun
from .plot_utils import plot_ellipse, plot_ellipsoid

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

    def evaluate(self, z: np.ndarray | ca.MX) -> float | ca.MX:
        """Evaluate the objective at point z"""
        return (z.T @ self.Q @ z / 2 + self.p.T @ z)


@dataclass
class LinearEqualityConstraint:
    """
    Represents a single linear equality constraint: h(z) = a^T z - b
    """
    a: np.ndarray # Constraint gradient vector (n x 1)
    b: float | np.float64 # Constraint affine trem

    def __post_init__(self):
        """Validate the constraint parameters""" 
        if not isinstance(self.a, np.ndarray) or not np.ndim(self.a) != 2 or self.a.shape[1] != 1:
            raise ValueError("a must be a NumPy column vector.")
        if not isinstance(self.b, float) or not isinstance(self.b, np.float64):
            raise ValueError("b must be a float or NumPy float.")
        
        self._normalize()
    
    def _normalize(self, tolerance: float = 1e-6):
        norm_a = np.linalg.norm(self.a)
        if norm_a < tolerance:
            raise ValueError(f"a must have norm superior to f{tolerance}.")
        self.a /= norm_a
        self.b /= norm_a

    def evaluate(self, z: np.ndarray | ca.MX) -> float | ca.MX:
        return (self.a.T @ z + self.b)


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

    def generate_random_objectives(self, rho_range: List[float] = [1, 2],
                                   lambda_range: List[float] = [0.5, 1.5]) -> None:
        """
        Generate random quadratic objectives with known solutions.

        Args:
            rho_range: Range for solution magnitude
            lambda_range: Range for eigenvalues of Q matrices
        """
        qlist, plist = random_quadfun(self.n_vars, self.n_objectives, rho_range, lambda_range)
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
            obj.evaluate(self._z) for obj in self.objectives
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

    def evaluate_objectives(self, z: np.ndarray) -> np.ndarray:
        """
        Evaluate all objectives at given point(s) in decision space.
        Maps z-space → phi-space (cost function vector space).

        Args:
            z: Decision variables (n_vars x num_points) or (n_vars x 1)

        Returns:
            Cost function values (n_objectives x num_points)
        """
        if not self._initialized:
            raise RuntimeError("Objectives not set")

        # Handle single point or multiple points
        if z.ndim == 1:
            z = z.reshape(-1, 1)

        num_points = z.shape[1]
        phi = np.zeros((self.n_objectives, num_points))

        for i, obj in enumerate(self.objectives):
            for j in range(num_points):
                phi[i, j] = obj.evaluate(z[:, [j]])

        return phi

    def compute_pareto_solutions(self, resolution: int = 4) -> np.ndarray:
        """
        Compute Pareto optimal solutions in decision space (z-space).

        Args:
            resolution: Grid resolution (higher = more points)

        Returns:
            Array of solutions in z-space (n_vars x num_points)
        """
        if not self._initialized:
            raise RuntimeError("Objectives not set")

        thetagrid = simplex_grid(self.n_objectives, resolution)
        num_points = len(thetagrid)

        solutions = np.zeros((self.n_vars, num_points))
        z_prev = np.zeros((self.n_vars, 1))

        print(f"Computing pareto solutions for {self.n_vars}-dimensional problem with {self.n_objectives} objective functions and resolution {resolution}. Running {num_points} optimizations...") 
        for i, theta in enumerate(thetagrid):
            solutions[:, [i]] = self.solve(theta, initial_guess=z_prev)
            z_prev = solutions[:, [i]]
        print(f"Successfully computed pareto solutions from {num_points} optimizations.")

        return solutions

    def compute_pareto_front(self, resolution: int = 4) -> np.ndarray:
        """
        Compute Pareto front in objective space (phi-space).

        Args:
            resolution: Grid resolution (higher = more points)

        Returns:
            Array of objective values in phi-space (n_objectives x num_points)
        """
        z_solutions = self.compute_pareto_solutions(resolution)
        return self.evaluate_objectives(z_solutions)

    def compute_pareto_both_spaces(self, resolution: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Pareto set in both decision and objective spaces.

        Args:
            resolution: Grid resolution (higher = more points)

        Returns:
            Tuple of (z_solutions, phi_values) where:
                - z_solutions: Decision space (n_vars x num_points)
                - phi_values: Objective space (n_objectives x num_points)
        """
        z_solutions = self.compute_pareto_solutions(resolution)
        phi_values = self.evaluate_objectives(z_solutions)
        return z_solutions, phi_values

    def get_objectives_list(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return the Q and p lists for plotting/analysis"""
        qlist = [obj.Q for obj in self.objectives]
        plist = [obj.p for obj in self.objectives]
        return qlist, plist

    def plot_individual_solutions(self,
                                  ax: Optional[plt.Axes] = None,
                                  plot_levelsets: bool = True,
                                  marker: str = 'ro',
                                  markersize: int = 15,
                                  label: str = 'Individual objectives',
                                  **kwargs) -> np.ndarray:
        """
        Solve and plot individual objective solutions (unit weight vectors).

        Args:
            ax: Matplotlib axes object (creates new if None)
            plot_levelsets: Whether to plot objective 1-level sets
            marker: Marker style for solutions
            markersize: Size of markers
            label: Label for legend
            **kwargs: Additional arguments passed to plot_ellipse

        Returns:
            Array of individual solutions (n_vars x n_objectives)
        """
        if not self._initialized:
            raise RuntimeError("Objectives not set")

        if self.n_vars != 2 and self.n_vars != 3:
            raise ValueError("Plotting only supported for 2D and 3D problems")

        # Solve for each individual objective
        solutions = np.zeros((self.n_vars, self.n_objectives))
        for i in range(self.n_objectives):
            theta = np.zeros((self.n_objectives, 1))
            theta[i] = 1.0
            solutions[:, [i]] = self.solve(theta)

        # Create axes if not provided
        if ax is None:
            if self.n_vars == 2:
                fig, ax = plt.subplots()
            if self.n_vars == 3:
                fig, ax = plt.subplots(subplot_kw={"projection":"3d"})

        # Plot solutions
        if self.n_vars == 2:
            ax.plot(solutions[0, :], solutions[1, :], marker,
                    markersize=markersize, label=label)
        if self.n_vars == 3:
            ax.plot(solutions[0, :], solutions[1, :], solutions[2, :], marker,
                    markersize=markersize, label=label)

        # Plot ellipses if requested
        if plot_levelsets:
            qlist, _ = self.get_objectives_list()
            ellipse_kwargs = {'color': 'b'}
            for i in range(len(qlist)):
                if self.n_vars == 2:
                    ellipse_kwargs['linewidth'] = 2
                    ellipse_kwargs.update(kwargs)
                    plot_ellipse(qlist[i], solutions[:, [i]], ax, **ellipse_kwargs)
                if self.n_vars == 3:
                    ellipse_kwargs['alpha'] = 0.3
                    ellipse_kwargs.update(kwargs)
                    plot_ellipsoid(qlist[i], solutions[:, [i]], ax, **ellipse_kwargs)

        return solutions

    def plot_pareto_solutions(self,
                         pareto_solutions: Optional[np.ndarray] = None,
                         resolution: int = 15,
                         alpha: float = 0.5,
                         ax: Optional[plt.Axes] = None,
                         plot_points: bool = True,
                         point_marker: str = 'k.',
                         point_size: int = 12,
                         point_label: str = 'Pareto solutions',
                         patch_color: str = 'blue',
                         patch_alpha: float = 0.2) -> np.ndarray:
        """
        Plot Pareto solutions in decision space (z-space) as an alphashape.

        Args:
            pareto_solutions: Precomputed Pareto solutions (n_vars x num_points).
                            If None, computes using resolution parameter.
            resolution: Grid resolution for computing Pareto solutions (ignored if pareto_solutions provided)
            alpha: Alpha parameter for alphashape (controls tightness of hull)
            ax: Matplotlib axes object (creates new if None)
            plot_points: Whether to plot the individual points
            point_marker: Marker style for Pareto points
            point_size: Size of point markers
            point_label: Label for Pareto points in legend
            patch_color: Color of the alphashape patch
            patch_alpha: Transparency of the alphashape patch

        Returns:
            Array of Pareto solutions in z-space (n_vars x num_points)
        """
        if not self._initialized:
            raise RuntimeError("Objectives not set")

        if self.n_vars != 2 and self.n_vars != 3:
            raise ValueError("Plotting only supported for 2D and 3D problems")

        # Compute Pareto solutions if not provided
        if pareto_solutions is None:
            pareto_solutions = self.compute_pareto_solutions(resolution=resolution)

        # Create axes if not provided
        if ax is None:
            if self.n_vars == 2:
                fig, ax = plt.subplots()
            if self.n_vars == 3:
                fig, ax = plt.subplots(subplot_kw={"projection":"3d"})

        # Convert to points array for alphashape (num_points x n_vars)
        points = pareto_solutions.T

        # Create and plot alphashape
        # alpha = alphashape.optimizealpha(points=points, max_iterations=1000, lower=0.02, upper=10.)
        # print(f"\n\nOptimized alpha {alpha}. \n\n")
        alpha_shape = alphashape.alphashape(points, alpha)

        if self.n_vars == 2:
            # 2D case: Use Polygon patch
            # FIX (old code breaks)
            def plot_shapely_poly(geom):
                if hasattr(geom, 'exterior'):
                    coords = np.array(geom.exterior.coords)
                    ax.add_patch(Polygon(coords, color=patch_color, alpha=patch_alpha))

            if alpha_shape is not None:
                if isinstance(alpha_shape, ShapelyPolygon):
                    plot_shapely_poly(alpha_shape)
                
                elif hasattr(alpha_shape, 'geoms'):
                    for geom in alpha_shape.geoms:
                        plot_shapely_poly(geom)

            # Plot points if requested
            if plot_points:
                ax.scatter(points[:, 0], points[:, 1],
                          marker=point_marker[1] if len(point_marker) > 1 else 'o',
                          c=point_marker[0] if len(point_marker) > 1 else 'k',
                          s=point_size,
                          label=point_label)

        elif self.n_vars == 3:
            # 3D case: Use Poly3DCollection for surface
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            # Extract vertices and faces from the 3D mesh
            vertices = alpha_shape.vertices
            faces = alpha_shape.faces

            # Create the 3D surface
            mesh = Poly3DCollection(vertices[faces], alpha=patch_alpha,
                                   facecolor=patch_color, edgecolor='darkgray',
                                   linewidths=0.5)
            ax.add_collection3d(mesh)

            # Plot points if requested
            if plot_points:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                          marker=point_marker[1] if len(point_marker) > 1 else 'o',
                          c=point_marker[0] if len(point_marker) > 1 else 'k',
                          s=point_size,
                          label=point_label)

        return pareto_solutions

    def plot_z_to_phi_mapping(self,
                              z_grid_resolution: int = 10,
                              ax_z: Optional[plt.Axes] = None,
                              ax_phi: Optional[plt.Axes] = None,
                              show_pareto: bool = True,
                              pareto_resolution: int = 15,
                              connection_alpha: float = 0.3,
                              connection_color: str = 'gray') -> Tuple[np.ndarray, np.ndarray]:
        """
        Visualize how a grid in decision space (z-space) maps to objective space (phi-space).
        Only supports 2D decision space (n_vars=2) and 2D objective space (n_objectives=2).

        Args:
            z_grid_resolution: Resolution for z-space grid
            ax_z: Matplotlib axes for z-space plot
            ax_phi: Matplotlib axes for phi-space plot
            show_pareto: Whether to highlight Pareto solutions/front
            pareto_resolution: Resolution for Pareto computation
            connection_alpha: Transparency for connection lines
            connection_color: Color for connection lines

        Returns:
            Tuple of (z_grid, phi_grid) where:
                - z_grid: Grid points in z-space (n_vars x num_grid_points)
                - phi_grid: Mapped points in phi-space (n_objectives x num_grid_points)
        """
        if not self._initialized:
            raise RuntimeError("Objectives not set")

        if self.n_vars != 2:
            raise ValueError("z-to-phi mapping visualization only supported for 2D decision space")

        if self.n_objectives != 2:
            raise ValueError("z-to-phi mapping visualization only supported for 2D objective space")

        # Create figure if axes not provided
        if ax_z is None or ax_phi is None:
            fig, (ax_z, ax_phi) = plt.subplots(1, 2, figsize=(12, 5))

        # Create grid in z-space
        if show_pareto:
            z_pareto = self.compute_pareto_solutions(resolution=pareto_resolution)
            z_min = np.min(z_pareto, axis=1, keepdims=True)
            z_max = np.max(z_pareto, axis=1, keepdims=True)
            margin = 0.2 * (z_max - z_min)
            z_min -= margin
            z_max += margin
        else:
            # Default grid bounds
            z_min = np.array([[-3], [-3]])
            z_max = np.array([[3], [3]])

        z1_vals = np.linspace(z_min[0, 0], z_max[0, 0], z_grid_resolution)
        z2_vals = np.linspace(z_min[1, 0], z_max[1, 0], z_grid_resolution)
        Z1, Z2 = np.meshgrid(z1_vals, z2_vals)

        # Flatten grid
        z_grid = np.vstack([Z1.ravel(), Z2.ravel()])
        num_grid_points = z_grid.shape[1]

        # Map to phi-space
        phi_grid = self.evaluate_objectives(z_grid)

        # Plot z-space grid
        ax_z.scatter(z_grid[0, :], z_grid[1, :], c='lightblue', s=20, alpha=0.6, label='Grid points')
        ax_z.set_xlabel('z₁')
        ax_z.set_ylabel('z₂')
        ax_z.set_title('Decision Space (z-space)')
        ax_z.grid(True, alpha=0.3)

        # Plot phi-space mapped points
        ax_phi.scatter(phi_grid[0, :], phi_grid[1, :], c='lightcoral', s=20, alpha=0.6, label='Mapped points')
        ax_phi.set_xlabel('φ₁ (cost 1)')
        ax_phi.set_ylabel('φ₂ (cost 2)')
        ax_phi.set_title('Objective Space (φ-space)')
        ax_phi.grid(True, alpha=0.3)

        # Plot Pareto solutions/front if requested
        if show_pareto:
            z_pareto, phi_pareto = self.compute_pareto_both_spaces(resolution=pareto_resolution)

            ax_z.plot(z_pareto[0, :], z_pareto[1, :], 'go-', linewidth=2,
                     markersize=6, label='Pareto solutions', zorder=10)
            ax_phi.plot(phi_pareto[0, :], phi_pareto[1, :], 'mo-', linewidth=2,
                       markersize=6, label='Pareto front', zorder=10)

        ax_z.legend()
        ax_phi.legend()
        ax_z.set_aspect('equal', adjustable='box')
        ax_phi.set_aspect('equal', adjustable='box')

        return z_grid, phi_grid

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
                     solver_opts: Optional[dict] = None,
                     visualize: bool = False,
                     resolution: int = 15) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
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
        
        if visualize:
            Visualizer = IOCVisualizer(self.optimizer, self.reference_vector, resolution)

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
            initial_z = self.optimizer.solve(initial_theta)

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

        recorder = IOCHistoryRecorder(opti, self.optimizer, theta, z, "IOC")
        opti.callback(recorder)

        opti.solver('ipopt', default_solver_opts)

        # Solve
        try:
            start_time = time.perf_counter()
            sol = opti.solve()
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            if visualize and len(recorder.hist_z) > 0:
                Visualizer.plot_solver_history(recorder)

            optimal_theta = np.array(sol.value(theta)).reshape(self.optimizer.n_objectives, 1)
            optimal_z = np.array(sol.value(z)).reshape(self.optimizer.n_vars, 1)
            final_loss = float(sol.value(cost))

            iterations = sol.stats()['iter_count']

            return optimal_theta, optimal_z, final_loss, elapsed_time, iterations
        except RuntimeError as e:
            if visualize: 
                plt.show() # Show what happened before crash
            raise RuntimeError(f"Inverse optimization solver failed: {e}")

class IOCHistoryRecorder:
    """
    Callback-like class to record optimization history without plotting in real-time.
    """
    def __init__(self, opti, optimizer, theta_var, z_var = None, name = " "):
        self.opti = opti
        self.optimizer = optimizer
        self.z_var = z_var
        self.theta_var = theta_var
        self.name = name
        
        self.hist_z = []
        self.hist_phi = []
        self.hist_theta = []
    
        self.evaluated_solutions = []

    def __call__(self, i):
        try:
            theta_val = self.opti.debug.value(self.theta_var).flatten()

            if self.z_var is not None:
                # z CasADi variable
                z_val = np.array(self.opti.debug.value(self.z_var)).flatten()
            else:
                # Use optimizer to solve for z given theta
                z_val = self.optimizer.solve(theta_val).flatten()
            
            phi_val = self.optimizer.evaluate_objectives(z_val).flatten()
            
            self.hist_z.append(z_val)
            self.hist_theta.append(theta_val)
            self.hist_phi.append(phi_val)

        except Exception:
            pass

class IOCVisualizer:
    def __init__(self, 
                 optimizer: MultiObjectiveOptimizer, 
                 reference_vector: np.ndarray,
                 pareto_z: Optional[np.ndarray] = None,
                 pareto_phi: Optional[np.ndarray] = None,
                 resolution: int = 15):
        self.optimizer = optimizer
        self.ref_z = reference_vector.flatten()
        self.ref_phi = optimizer.evaluate_objectives(self.ref_z).flatten()
        if pareto_z is not None and pareto_phi is not None:
            self.pareto_z = pareto_z
            self.pareto_phi = pareto_phi
        else:
            self.pareto_z, self.pareto_phi = self.optimizer.compute_pareto_both_spaces(resolution)

    def _draw_alphashape(self, ax, points, n_vars, alpha=1.5, color='blue', p_alpha=0.15):
        import alphashape
        from matplotlib.patches import Polygon as MplPolygon
        from shapely.geometry import Polygon as ShapelyPolygon
        
        pts = points.T if points.shape[0] == n_vars else points
        shape = alphashape.alphashape(pts, alpha)

        if n_vars == 2:
            def add_poly(geom):
                if hasattr(geom, 'exterior'):
                    coords = np.array(geom.exterior.coords)
                    ax.add_patch(MplPolygon(coords, color=color, alpha=p_alpha, zorder=1))

            if isinstance(shape, ShapelyPolygon):
                add_poly(shape)
            elif hasattr(shape, 'geoms'):
                for g in shape.geoms: add_poly(g)

        elif n_vars == 3:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            try:
                vertices = shape.vertices
                faces = shape.faces
                mesh = Poly3DCollection(vertices[faces], alpha=p_alpha, 
                                      facecolor=color, edgecolor='none', zorder=1)
                ax.add_collection3d(mesh)
            except AttributeError:
                print("W: Alphashape nije mogao da generiše 3D mesh.")

    def plot_solver_history(self, recorder: IOCHistoryRecorder):
        hist_z = np.array(recorder.hist_z)
        hist_phi = np.array(recorder.hist_phi)
        hist_th = np.array(recorder.hist_theta)
     
        n_iters = len(hist_z)
        n_obj = self.optimizer.n_objectives
        n_vars = hist_z.shape[1] 

        fig = plt.figure(figsize=(16, 6))
        plt.subplots_adjust(bottom=0.25)
        fig.canvas.manager.set_window_title(f'Solver History {recorder.name}')

        if n_vars == 3:
            ax_z = fig.add_subplot(131, projection='3d')
            ax_z.set_title("1. Decision Space (z)", fontweight='bold')
            
            ax_z.scatter(self.pareto_z[0,:], self.pareto_z[1,:], self.pareto_z[2,:], c='k', alpha=0.1, label='Pareto Set')
            ax_z.scatter(self.ref_z[0], self.ref_z[1], self.ref_z[2], c='g', marker='*', s=100, label='Ref')
            
            line_z, = ax_z.plot([], [], [], 'r-', alpha=0.5)
            point_z, = ax_z.plot([], [], [], 'ro', markersize=8, label='Current Solution')
            
            ax_z.set_xlabel('$z_1$'); ax_z.set_ylabel('$z_2$'); ax_z.set_zlabel('$z_3$')
            
        else:
            ax_z = fig.add_subplot(131)
            ax_z.set_title("1. Decision Space (z)", fontweight='bold')
            ax_z.plot(self.pareto_z[0,:], self.pareto_z[1,:], 'k.', alpha=0.1, label='Pareto Set')
            ax_z.plot(self.ref_z[0], self.ref_z[1], 'g*', markersize=15, label='Ref')
            line_z, = ax_z.plot([], [], 'r-', alpha=0.5)
            point_z, = ax_z.plot([], [], 'ro', markersize=8, label='Current Solution')
            ax_z.set_xlabel('$z_1$'); ax_z.set_ylabel('$z_2$')
            ax_z.grid(True, alpha=0.3)

    
        self._draw_alphashape(ax_z, self.pareto_z, n_vars)

        def project_simplex(thetas):
            sqrt3_2 = np.sqrt(3) / 2
            x = 0.5 * (thetas[:, 2] - thetas[:, 1])
            y = sqrt3_2 * thetas[:, 0]
            return x, y
        
        simplex_x, simplex_y = None, None
        if n_obj == 3:
            simplex_x, simplex_y = project_simplex(hist_th)
        
        use_high_dim = (n_obj > 3)
        if not use_high_dim:
            if n_obj == 3:
                # 3D Objective Plot
                ax_phi = fig.add_subplot(132, projection='3d')
                ax_phi.scatter(self.pareto_phi[0,:], self.pareto_phi[1,:], self.pareto_phi[2,:], c='k', alpha=0.05, label='Pareto Front')
                ax_phi.scatter(self.ref_phi[0], self.ref_phi[1], self.ref_phi[2], c='g', marker='*', s=100, label='Ref')
                scat_phi = ax_phi.scatter([], [], [], c='r', s=30, label='Current Solution')

                ax_phi.set_title("2. Phi Space ($\\phi$)", fontweight='bold')
                ax_phi.set_xlabel('$\\phi_1$'); ax_phi.set_ylabel('$\\phi_2$'); ax_phi.set_zlabel('$\\phi_3$')
                ax_phi.legend(loc='upper left', fontsize='small')

                # 3D Weight Simplex
                ax_th = fig.add_subplot(133)
                ax_th.set_title("3. Weights ($\\theta$)", fontweight='bold')
                
                v_top = [0, np.sqrt(3)/2]
                v_left = [-0.5, 0]
                v_right = [0.5, 0]
 
                ax_th.plot([v_top[0], v_left[0]], [v_top[1], v_left[1]], 'k-', lw=1, alpha=0.6)
                ax_th.plot([v_left[0], v_right[0]], [v_left[1], v_right[1]], 'k-', lw=1, alpha=0.6)
                ax_th.plot([v_right[0], v_top[0]], [v_right[1], v_top[1]], 'k-', lw=1, alpha=0.6)
        
                ax_th.text(v_top[0], v_top[1]+0.05, '$\\theta_1$', ha='center', fontweight='bold')
                ax_th.text(v_left[0]-0.05, v_left[1], '$\\theta_2$', ha='right', fontweight='bold')
                ax_th.text(v_right[0]+0.05, v_right[1], '$\\theta_3$', ha='left', fontweight='bold')

                line_th, = ax_th.plot([], [], 'b-', alpha=0.5, lw=2, label='Weight Path')
                point_th, = ax_th.plot([], [], 'bo', markersize=8, label='Current Solution')
                
                ax_th.set_xlim(-0.6, 0.6)
                ax_th.set_ylim(-0.1, 1.0)
                ax_th.axis('off') 
                ax_th.legend(loc='upper right', fontsize='small')

            else:
                ax_phi = fig.add_subplot(132)
                ax_phi.plot(self.pareto_phi[0,:], self.pareto_phi[1,:], 'k.', alpha=0.1, label='Pareto Front')
                ax_phi.plot(self.ref_phi[0], self.ref_phi[1], 'g*', markersize=15, label='Ref')
                line_phi, = ax_phi.plot([], [], 'r-', alpha=0.5)
                point_phi, = ax_phi.plot([], [], 'ro', markersize=8, label='Current Solution')
                ax_phi.set_title("2. Phi Space ($\\phi$)", fontweight='bold')
                ax_phi.set_xlabel("$\\phi_1$"); ax_phi.set_ylabel("$\\phi_2$")
                ax_phi.grid(True, alpha=0.3)
                ax_phi.legend(loc='upper left', fontsize='small')

                ax_th = fig.add_subplot(133)
                ax_th.set_title("3. Weights ($\\theta$)", fontweight='bold')
                ax_th.plot([0,1], [1,0], 'k--')
                point_th, = ax_th.plot([], [], 'bo', markersize=8, label='Current Solution')
                ax_th.legend(loc='upper right', fontsize='small')

        else:
            ax_phi = fig.add_subplot(132); ax_th = fig.add_subplot(133)
            ax_phi.set_title("2. Phi Space ($\\phi$)", fontweight='bold'); 
            ax_th.set_title("3. Weights ($\\theta$)", fontweight='bold')
            step = max(1, self.pareto_phi.shape[1] // 100)
            ax_phi.plot(self.pareto_phi[:, ::step], color='grey', alpha=0.1)
            ax_phi.plot(self.pareto_phi[:, :1:step], 'grey', alpha=0.4, label='Pareto Front')
            ax_phi.xaxis.set_ticks(range(n_obj))
            ax_phi.set_xticklabels([f'$\\phi_{i+1}$' for i in range(n_obj)])

            ax_phi.plot(self.ref_phi, 'g--o', alpha=0.7, label='Ref')
            line_phi, = ax_phi.plot([], [], 'r-o', linewidth=2, markersize=6, label='Current Solution')
            bar_container_labels = [f'$\\theta_{i+1}$' for i in range(n_obj)]
            bar_container = ax_th.bar(bar_container_labels, np.zeros(n_obj), color='blue')
            ax_th.set_ylim(0, 1)
            ax_phi.set_ylabel('Cost Values'); ax_th.set_ylabel('Weight Values')
            ax_phi.grid(True, alpha=0.3)
            ax_phi.legend(loc='upper left', fontsize='small')
        
        scat_eval = None
        if recorder.name == "MaxEntIOC":
            all_evals = np.vstack(recorder.evaluated_solutions)
            
            if n_vars == 3:
                scat_eval = ax_z.scatter([], [], [], c='black', marker='x', s=40, 
                                    label='Evaluated Solutions', zorder=5)
            else:
                scat_eval = ax_z.scatter([], [], c='black', marker='x', s=40, 
                                    label='Evaluated Solutions', zorder=5)
        ax_z.legend(loc='upper left', fontsize='small')

        def update(val):
            i = int(self.slider.val)
            
            if n_vars == 3:
                point_z.set_data([hist_z[i,0]], [hist_z[i,1]])
                point_z.set_3d_properties([hist_z[i,2]])
            else:
                point_z.set_data([hist_z[i,0]], [hist_z[i,1]])

            if not use_high_dim:
                if n_obj == 3:
                    scat_phi._offsets3d = (hist_phi[i:i+1,0], hist_phi[i:i+1,1], hist_phi[i:i+1,2])
         
                    point_th.set_data([simplex_x[i]], [simplex_y[i]])
                    line_th.set_data(simplex_x[:i+1], simplex_y[:i+1])
                else:
                    point_phi.set_data([hist_phi[i,0]], [hist_phi[i,1]])
                    point_th.set_data([hist_th[i,0]], [hist_th[i,1]])
            else:
                line_phi.set_data(range(n_obj), hist_phi[i])
                for rect, h in zip(bar_container, hist_th[i]): rect.set_height(h)


            if recorder.name == "MaxEntIOC" and scat_eval is not None:
                current_points = all_evals[:i+1] 
                
                if n_vars == 3:
                    scat_eval._offsets3d = (current_points[:, 0], current_points[:, 1], current_points[:, 2])
                else:
                    scat_eval.set_offsets(current_points[:, :2])

                fig.canvas.draw_idle()

        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.slider = Slider(ax_slider, 'Iter', 0, n_iters - 1, valinit=0, valstep=1)
        self.slider.on_changed(update)

        print("\nInteractive Slider Open.")
        print("--> Close the slider window to generate the final main plot.\n")
        
        plt.show(block=True)

class MaximumEntropyIRL:
    """Maximum Entropy Inverse Reinforcement Learning"""
    def __init__(self,
                 optimizer : MultiObjectiveOptimizer = None,
                 reference_vector : np.ndarray = None, 
                 initial_theta: Optional[np.ndarray] = None, 
                 resolution: int = 20
                 ):
        self.optimizer = optimizer
        self.reference_vector = reference_vector
        if initial_theta is None:
            n_obj = self.optimizer.n_objectives
            initial_theta = np.ones(n_obj) / n_obj
        else:
            self.initial_theta = initial_theta
        
        self.initial_z = self.optimizer.solve(initial_theta.reshape(-1, 1))
        self.initial_phi = self.optimizer.evaluate_objectives(self.initial_z).flatten()

        if reference_vector is not None:
            self.phi_ref = self.optimizer.evaluate_objectives(reference_vector).flatten()
            
            self.evaluated_solutions = [self.initial_z.flatten(), reference_vector.flatten()]
            self.evaluated_costs = [self.initial_phi, self.phi_ref]
        else:
            print("Warning: No reference vector provided for IOC.")
      
    def ioc_loss(self, theta : ca.MX) -> ca.MX:
        """Computes the MaxEnt loss using the current pool of evaluated costs."""
        weighted_costs = ca.mtimes(theta.T, np.array(self.evaluated_costs).T)

        neg_costs = -weighted_costs
        max_val = ca.mmax(neg_costs)
        log_Z = max_val + ca.log(ca.sum2(ca.exp(neg_costs - max_val)))

        return ca.mtimes(theta.T, self.phi_ref.reshape(-1, 1)) + log_Z

    def solve_inverse(self,
                     initial_theta: Optional[np.ndarray] = None,
                     visualize: bool = False,
                     solver_opts: Optional[dict] = None,
                     max_iterations: int = 15,
                     tolerance: float = 1e-4) -> Tuple[np.ndarray, np.ndarray, float, float, float]:

        n_obj = self.optimizer.n_objectives
        current_theta = initial_theta if initial_theta is not None else np.ones(n_obj)/n_obj

        recorder = IOCHistoryRecorder(opti=None, optimizer=self.optimizer, theta_var=None, name="MaxEntIOC")
        recorder.hist_z.append(self.initial_z.flatten())
        recorder.hist_phi.append(self.initial_phi)
        recorder.hist_theta.append(current_theta.flatten())
        recorder.evaluated_solutions = [self.initial_z.flatten(), self.reference_vector.flatten()]

        average_time_per_iter = 0.0
        elapsed_time = 0.0
        iterations = 0
        converged = False

        start_time = time.perf_counter()
        for i in range(max_iterations):
            opti = ca.Opti()
            theta = opti.variable(self.optimizer.n_objectives, 1)

            loss_exp = self.ioc_loss(theta)
            
            opti.minimize(loss_exp)
            opti.subject_to(ca.sum1(theta) == 1.0)
            opti.subject_to(theta >= 1e-4)
            
            opti.set_initial(theta, current_theta.reshape(-1, 1))
            opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0})
            
            iter_start_time = time.perf_counter()
            sol = opti.solve()
            current_loss_num = float(sol.value(loss_exp))
            current_theta = sol.value(theta).flatten()

            z_new = self.optimizer.solve(current_theta.reshape(-1, 1))
            phi_new = self.optimizer.evaluate_objectives(z_new).flatten()

            iter_end_time = time.perf_counter()
            average_time_per_iter = ((average_time_per_iter * i) + (iter_end_time - iter_start_time)) / (i + 1)
            print(f"Iteration {i+1}: Loss = {current_loss_num:.6f}, Theta = {current_theta}, Time = {iter_end_time - iter_start_time:.4f}s")

            self.evaluated_solutions.append(z_new.flatten())
            self.evaluated_costs.append(phi_new)

            recorder.hist_z.append(z_new.flatten())
            recorder.hist_theta.append(current_theta.flatten())
            recorder.hist_phi.append(phi_new)
            recorder.evaluated_solutions.append(z_new.flatten())

            if i > 0:
                diff = np.linalg.norm(z_new.flatten() - recorder.hist_z[-2])
                if diff < tolerance:                    
                    end_time = time.perf_counter()
                    elapsed_time = end_time - start_time
                    iterations = i + 1
                    converged = True
                    break

        if not converged:           
            iterations = max_iterations
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
                    
        if visualize:
            viz = IOCVisualizer(self.optimizer, self.reference_vector)
            viz.plot_solver_history(recorder)

        return current_theta, z_new, current_loss_num, elapsed_time, average_time_per_iter, iterations
