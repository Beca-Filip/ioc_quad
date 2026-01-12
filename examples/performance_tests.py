from ioc_quad.core import MultiObjectiveOptimizer, InverseOptimalControl, MaximumEntropyIRL

import numpy as np
import matplotlib.pyplot as plt
import time
import random

def run_performance_comparison(n_problems=30):
    results = {
        "Bilevel": {"times": [], "iters": []},
        "MaxEnt": {"times": [], "iters": []}
    }
    
    combinations = []
    for n in [2, 3]:
        for m in [2, 3, 4, 5]:
            if not (n == 3 and m == 2):
                combinations.append((n, m))

    print(f"Benchmarking {n_problems} random problems...")

    for i in range(n_problems):
        n, m = random.choice(combinations)
        
        opt = MultiObjectiveOptimizer(n_vars=n, n_objectives=m)
        opt.generate_random_objectives()
        
        random_theta = np.random.dirichlet(np.ones(m)).reshape(-1, 1)
        z_ref = opt.solve(random_theta)
        
        ioc = InverseOptimalControl(opt, z_ref)
        _, _, _, t_b, it_b = ioc.solve_inverse() 
        results["Bilevel"]["times"].append(t_b)
        results["Bilevel"]["iters"].append(it_b)
        
        maxent = MaximumEntropyIRL(opt, z_ref)
        _, _, _, t_m, _, it_m = maxent.solve_inverse()
        results["MaxEnt"]["times"].append(t_m)
        results["MaxEnt"]["iters"].append(it_m)
        
        print(f"Problem {i+1}: n={n}, m={m} | Bilevel: {it_b} it, {t_b:.3f}s | MaxEnt: {it_m} it, {t_m:.3f}s")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    _plot_on_ax(ax1, {k: v["iters"] for k, v in results.items()}, "Iterations comparison")
    _plot_on_ax(ax2, {k: v["times"] for k, v in results.items()}, "Solving time comparison")
    
    plt.tight_layout()
    plt.show()

def _plot_on_ax(ax, data_dict, title):
    all_vals = np.array(list(data_dict.values()))
    min_vals = np.min(all_vals, axis=0)
    
    colors = ['#1f77b4', '#ff7f0e'] 
    
    for i, (name, vals) in enumerate(data_dict.items()):
        vals = np.array(vals)
        ratios = vals / np.clip(min_vals, 1e-9, None)
        sorted_ratios = np.sort(ratios)
        y = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
        
        ax.step(sorted_ratios, y, label=name, color=colors[i], where='post', linewidth=2)

    ax.set_xscale('log')
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.set_xlabel(r'Performance ratio $\tau$')
    ax.set_ylabel('Ratio of problems solved')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1)
    ax.set_xlim(left=1.0)

np.random.seed(42)
run_performance_comparison()