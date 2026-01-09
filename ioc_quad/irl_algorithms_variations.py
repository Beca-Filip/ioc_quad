from typing import Optional, Tuple
import numpy as np
import casadi as ca
from scipy.optimize import minimize
from ioc_quad import MultiObjectiveOptimizer, IOCVisualizer, IOCHistoryRecorder

class MaximumEntropyIRL_ParetoSamples_Casadi:
    """Maximum Entropy Inverse Reinforcement Learning using pre-computed Pareto samples, using a Casadi optimizer"""
    def __init__(self,
                 optimizer : MultiObjectiveOptimizer = None,
                 reference_vector : np.ndarray = None,
                 resolution: int = 20):
        
        self.optimizer = optimizer
        self.reference_vector = reference_vector
        
        if reference_vector is not None:
            self.phi_ref = self.optimizer.evaluate_objectives(reference_vector).flatten()

        print(f"Pre-computing {resolution} resolution Pareto samples for MaxEnt base...")
        self.pareto_z, self.pareto_phi = self.optimizer.compute_pareto_both_spaces(resolution)

    def ioc_loss(self, theta : np.ndarray, recorder: Optional[object] = None) -> float:
        theta = theta.flatten()
        sample_costs = self.pareto_phi.T @ theta
        
        max_val = np.max(-sample_costs)
        log_Z = max_val + np.log(np.sum(np.exp(-sample_costs - max_val)))
        
        cost_expert = np.dot(self.phi_ref, theta)
        loss = cost_expert + log_Z

        if recorder is not None:
            z_curr = self.optimizer.solve(theta.reshape(-1, 1))
            phi_curr = self.optimizer.evaluate_objectives(z_curr).flatten()
            recorder.hist_z.append(z_curr.flatten())
            recorder.hist_theta.append(theta.copy())
            recorder.hist_phi.append(phi_curr)
            
        return float(loss)

    def solve_inverse(self,
                     initial_theta: Optional[np.ndarray] = None,
                     visualize: bool = False,
                     solver_opts: Optional[dict] = None,
                     resolution: int = 15) -> Tuple[np.ndarray, np.ndarray, float]:
        
        n_obj = self.optimizer.n_objectives
        n_samples = self.pareto_phi.shape[1]

        opti = ca.Opti()
        theta = opti.variable(n_obj, 1)

        phi_ref = self.phi_ref.reshape(n_obj, 1)
        phi_samples = self.pareto_phi

        weighted_costs = ca.mtimes(theta.T, phi_samples)
        neg_costs = -weighted_costs
        
        max_val = ca.mmax(neg_costs)
        log_Z = max_val + ca.log(ca.sum2(ca.exp(neg_costs - max_val)))
        
        cost_expert = ca.mtimes(theta.T, phi_ref)
        loss = cost_expert + log_Z
        
        opti.minimize(loss)

        opti.subject_to(ca.sum1(theta) == 1.0)
        opti.subject_to(theta >= 1e-4) 

        recorder = IOCHistoryRecorder(opti, self.optimizer, theta, z_var=None, name="MaxEntCasadi")
        opti.callback(recorder)

        default_opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-9}
        if solver_opts: default_opts.update(solver_opts)
        opti.solver('ipopt', default_opts)

        if initial_theta is not None:
            opti.set_initial(theta, initial_theta)
        else:
            opti.set_initial(theta, np.ones((n_obj, 1)) / n_obj)

        try:
            sol = opti.solve()
            final_theta = np.array(sol.value(theta)).reshape(-1, 1)
            print(final_theta)
            print("MaxEnt Optimization converged successfully.")
        except RuntimeError:
            print("MaxEnt Optimization failed to converge, using last debug values.")
            final_theta = np.array(opti.debug.value(theta)).reshape(-1, 1)

        final_z = self.optimizer.solve(final_theta)

        if visualize and len(recorder.hist_z)>0:
            print("Launching Visualizer...")
            viz = IOCVisualizer(self.optimizer, self.reference_vector)
            viz.pareto_z = self.pareto_z
            viz.pareto_phi = self.pareto_phi
            viz.plot_solver_history(recorder)
        
        final_loss = self.ioc_loss(final_theta)

        return final_theta, final_z, final_loss
    
class MaximumEntropyIRL_ParetoSamples_Scipy:
    """Maximum Entropy Inverse Reinforcement Learning using pre-computed Pareto samples, using a Scipy optimizer"""
    def __init__(self,
            optimizer : MultiObjectiveOptimizer = None,
            reference_vector : np.ndarray = None,
            resolution: int = 20):
        self.optimizer = optimizer
        self.reference_vector = reference_vector
        
        if reference_vector is not None:
            self.phi_ref = self.optimizer.evaluate_objectives(reference_vector).flatten()

        print(f"Pre-computing {resolution} resolution Pareto samples for MaxEnt base...")
        self.pareto_z, self.pareto_phi = self.optimizer.compute_pareto_both_spaces(resolution)

    def ioc_loss(self, theta : np.ndarray, recorder: Optional[object] = None) -> float:
        theta = theta.flatten()
        sample_costs = self.pareto_phi.T @ theta
        
        max_val = np.max(-sample_costs)
        log_Z = max_val + np.log(np.sum(np.exp(-sample_costs - max_val)))
        
        cost_expert = np.dot(self.phi_ref, theta)
        loss = cost_expert + log_Z

        if recorder is not None:
            z_curr = self.optimizer.solve(theta.reshape(-1, 1))
            phi_curr = self.optimizer.evaluate_objectives(z_curr).flatten()
            recorder.hist_z.append(z_curr.flatten())
            recorder.hist_theta.append(theta.copy())
            recorder.hist_phi.append(phi_curr)
            
        return float(loss)
              
    def solve_inverse(self,
                    initial_theta: Optional[np.ndarray] = None,
                    visualize: bool = False,
                    solver_opts: Optional[dict] = None,
                    resolution: int = 15) -> Tuple[np.ndarray, np.ndarray, list]:
        
        n_obj = self.optimizer.n_objectives
        if initial_theta is None:
            initial_theta = np.ones(n_obj) / n_obj

        class History:
            def __init__(self):
                self.hist_z, self.hist_phi, self.hist_theta, self.loss_history = [], [], [], []
                self.name = "MaxEntScipy"
        recorder = History()

        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = [(0.0, 1.0) for _ in range(n_obj)]

        res = minimize(
            self.ioc_loss,
            initial_theta,
            args=(recorder,),
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 100, 'ftol': 1e-9, 'disp': True}
        )

        opt_theta = res.x.reshape(-1, 1)
        opt_z = self.optimizer.solve(opt_theta)

        if visualize:
            viz = IOCVisualizer(self.optimizer, self.reference_vector)
            viz.pareto_z = self.pareto_z
            viz.pareto_phi = self.pareto_phi
            viz.plot_solver_history(recorder)

        return opt_theta, opt_z, recorder.loss_history