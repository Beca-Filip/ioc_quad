from ioc_quad.core import MultiObjectiveOptimizer, InverseOptimalControl, MaximumEntropyIRL
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Dict, List, Optional

# Plotting utilities for performance tests

def plot_cdf_performance(results: Dict[str, Dict[str, List[float]]],
                         metric: str = "time",
                         ax: Optional[plt.Axes] = None,
                         title: Optional[str] = None) -> plt.Figure:
    """
    Plot empirical CDF of performance ratio (each method compared to best per-problem).
    `results` is {method: {metric: [vals]}} and assumes same length lists per method.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    method_names = list(results.keys())
    lists = [np.array(results[m].get(metric, []), dtype=float) for m in method_names]
    min_len = min([len(l) for l in lists if l.size > 0], default=0)
    if min_len == 0:
        ax.text(0.5, 0.5, "No data", ha='center')
        return fig

    trimmed = [l[:min_len] for l in lists]
    stacked = np.vstack(trimmed) 
    min_per_problem = np.nanmin(stacked, axis=0)

    for name, vals in zip(method_names, trimmed):
        ratios = vals / np.clip(min_per_problem, 1e-12, None)
        ratios = ratios[~np.isnan(ratios)]
        if ratios.size == 0:
            continue
        sorted_ratios = np.sort(ratios)
        y = np.arange(1, len(sorted_ratios)+1) / len(sorted_ratios)
        ax.step(sorted_ratios, y, where='post', label=name, linewidth=2)

    ax.set_xscale('log')
    ax.set_xlabel("Performance ratio")
    ax.set_ylabel("Ratio of problems solved")
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.set_ybound(0, 1)
    ax.legend()
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"CDF of {metric}")
    plt.tight_layout()
    return fig


def plot_scaling(raw_data: Dict[str, Dict[int, dict]],
                 x_values: List[int],
                 metric: str = "time",
                 ax: Optional[plt.Axes] = None,
                 title: Optional[str] = None) -> plt.Figure:
    """
    Plot median ± [min,max] across methods for scaling data.
    raw_data: {method: {x_value: {metric: [vals]}}}
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    methods = list(raw_data.keys())
    styles = plt.cm.tab10.colors

    for i, method in enumerate(methods):
        medians = []
        low = []
        high = []
        for x in x_values:
            vals = np.array(raw_data[method][x].get(metric, []), dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                medians.append(np.nan); low.append(0); high.append(0)
                continue
            med = np.median(vals)
            medians.append(med)
            low.append(med - np.min(vals))
            high.append(np.max(vals) - med)
        ax.errorbar(x_values, medians, yerr=[low, high], fmt='-o', color=styles[i % len(styles)],
                    label=method, capsize=3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('dimension')
    ax.set_ylabel(metric)
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_grouped_bar(mean_data: Dict[str, List[float]],
                     labels: List[str],
                     ax: Optional[plt.Axes] = None,
                     title: Optional[str] = None) -> plt.Figure:
    """
    Plot grouped bar chart.
    mean_data: {method: [mean for each label index]}
    labels: list of label strings (x ticks)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    methods = list(mean_data.keys())
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(methods))
    for i, method in enumerate(methods):
        ax.bar(x + (i - len(methods)/2) * width + width/2, mean_data[method], width, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean time [s]")
    ax.set_yscale('log')
    ax.legend()
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return fig

# Performance testing functions (specific IOC methods)

def run_performance_comparison(n_problems=50, n_range=[2,3], m_range=[2,3,4,5]):
    """Run performance comparison on combinations of n and m."""
    results = {
        "Bilevel": {"times": [], "iters": []},
        "MaxEnt": {"times": [], "iters": []}
    }
    
    combinations = []
    for n in n_range:
        for m in m_range:
            if not (n > m):
                combinations.append((n, m))

    print(f"Benchmarking {n_problems} random problems...")

    for i in range(n_problems):
        n, m = random.choice(combinations)
        
        opt = MultiObjectiveOptimizer(n_vars=n, n_objectives=m)
        opt.generate_random_objectives()
        
        #random_theta = np.random.dirichlet(np.ones(m)).reshape(-1, 1)
        #z_ref = opt.solve(random_theta)
        
        zcost = opt.compute_pareto_solutions(resolution=2)
        centroid_cost = np.mean(zcost, axis=1, keepdims=True)
        step_size = np.std(zcost - centroid_cost, axis=1, keepdims=True)

        z_ref = zcost[:, [0]] + (zcost[:, [0]] - centroid_cost) * step_size
        
        ioc = InverseOptimalControl(opt, z_ref)
        _, _, _, t_b, it_b = ioc.solve_inverse() 
        results["Bilevel"]["times"].append(t_b)
        results["Bilevel"]["iters"].append(it_b)
        
        maxent = MaximumEntropyIRL(opt, z_ref)
        _, _, _, t_m, _, it_m = maxent.solve_inverse(max_iterations=100)
        results["MaxEnt"]["times"].append(t_m)
        results["MaxEnt"]["iters"].append(it_m)
        
        print(f"Problem {i+1}: n={n}, m={m} | Bilevel: {it_b} it, {t_b:.3f}s | MaxEnt: {it_m} it, {t_m:.3f}s")

    fig1 = plot_cdf_performance(results, metric="times", title="CDF of Solving Times")
    fig2 = plot_cdf_performance(results, metric="iters", title="CDF of Number of Iterations")
    plt.show()  

def plot_scaling_n(n_range=[2, 4, 8, 16], m_static = 3, trials=5):
    methods = ["Bilevel", "MaxEnt"]
    raw_data = {
        m: {
            n: {"times": [], "iters": []} for n in n_range
        } for m in methods
    }
    
    time_limit = 100.0 
    iter_limit = 1000.0 

    for n in n_range:
        for _ in range(trials):
            m = m_static
            opt = MultiObjectiveOptimizer(n_vars=n, n_objectives=m)
            opt.generate_random_objectives()
            zcost = opt.compute_pareto_solutions(resolution=2)
            centroid_cost = np.mean(zcost, axis=1, keepdims=True)
            step_size = np.std(zcost - centroid_cost, axis=1, keepdims=True)

            z_ref = zcost[:, [0]] + (zcost[:, [0]] - centroid_cost) * step_size
            
            try:
                ioc = InverseOptimalControl(opt, z_ref)
                _, _, _, t_b, it_b = ioc.solve_inverse()
                raw_data["Bilevel"][n]["times"].append(t_b if t_b < time_limit else time_limit)
                raw_data["Bilevel"][n]["iters"].append(it_b if it_b < iter_limit else iter_limit)
            except:
                raw_data["Bilevel"][n]["times"].append(time_limit)
                raw_data["Bilevel"][n]["iters"].append(iter_limit)

            try:
                maxent = MaximumEntropyIRL(opt, z_ref)
                _, _, _, t_m, _, it_m = maxent.solve_inverse()
                raw_data["MaxEnt"][n]["times"].append(t_m if t_m < time_limit else time_limit)
                raw_data["MaxEnt"][n]["iters"].append(it_m if it_m < iter_limit else iter_limit)
            except:
                raw_data["MaxEnt"][n]["times"].append(time_limit)
                raw_data["MaxEnt"][n]["iters"].append(iter_limit)
        

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    plot_scaling(raw_data, x_values=n_range, metric="times", ax=ax1, title="Solving times")
    plot_scaling(raw_data, x_values=n_range, metric="iters", ax=ax2, title="Number of iterations")
    plt.tight_layout()

    plt.show()

def plot_scaling_m(n_static = 4, m_range = [2,4,8,16,32], trials=5):
    methods = ["Bilevel", "MaxEnt"]
    raw_data = {
        method: {
            m: {"times": [], "iters": []} for m in m_range
        } for method in methods
    }
    
    time_limit = 100.0 
    iter_limit = 1000.0 


    for m in m_range:
        for _ in range(trials):
            n = n_static
            opt = MultiObjectiveOptimizer(n_vars=n, n_objectives=m)
            opt.generate_random_objectives()
            zcost = opt.compute_pareto_solutions(resolution=2)
            centroid_cost = np.mean(zcost, axis=1, keepdims=True)
            step_size = np.std(zcost - centroid_cost, axis=1, keepdims=True)

            z_ref = zcost[:, [0]] + (zcost[:, [0]] - centroid_cost) * step_size
            
            try:
                ioc = InverseOptimalControl(opt, z_ref)
                _, _, _, t_b, it_b = ioc.solve_inverse()
                raw_data["Bilevel"][m]["times"].append(t_b if t_b < time_limit else time_limit)
                raw_data["Bilevel"][m]["iters"].append(it_b if it_b < iter_limit else iter_limit)
            except:
                raw_data["Bilevel"][m]["times"].append(time_limit)
                raw_data["Bilevel"][m]["iters"].append(iter_limit)
            try:
                maxent = MaximumEntropyIRL(opt, z_ref)
                _, _, _, t_m, _, it_m = maxent.solve_inverse()
                raw_data["MaxEnt"][m]["times"].append(t_m if t_m < time_limit else time_limit)
                raw_data["MaxEnt"][m]["iters"].append(it_m if it_m < iter_limit else iter_limit)
            except:
                raw_data["MaxEnt"][m]["times"].append(time_limit)
                raw_data["MaxEnt"][m]["iters"].append(iter_limit)
        

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    plot_scaling(raw_data, x_values=m_range, metric="times", ax=ax1, title="Solving times")
    plot_scaling(raw_data, x_values=m_range, metric="iters", ax=ax2, title="Number of iterations")
        
    plt.tight_layout()
    plt.show()

def plot_grouped_bar_performance(n_range=[2, 4, 8, 16], m_static = 3, trials=5):
    bilevel_times = []
    maxent_times = []
    
    color_bilevel = '#2c7fb8'
    color_maxent = '#7fcdbb'  

    for n in n_range:
        b_temp, m_temp = [], []
        for _ in range(trials):
            m = m_static
            opt = MultiObjectiveOptimizer(n_vars=n, n_objectives=m)
            opt.generate_random_objectives()
            zcost = opt.compute_pareto_solutions(resolution=2)
            centroid_cost = np.mean(zcost, axis=1, keepdims=True)
            step_size = np.std(zcost - centroid_cost, axis=1, keepdims=True)

            z_ref = zcost[:, [0]] + (zcost[:, [0]] - centroid_cost) * step_size
            
            try:
                ioc = InverseOptimalControl(opt, z_ref)
                _, _, _, t_b, _ = ioc.solve_inverse()
                b_temp.append(t_b)
            except: b_temp.append(np.nan)
            
            try:
                maxent = MaximumEntropyIRL(opt, z_ref)
                _, _, _, t_m, _ , _ = maxent.solve_inverse()
                m_temp.append(t_m)
            except: m_temp.append(np.nan)
            
        bilevel_times.append(np.nanmean(b_temp))
        maxent_times.append(np.nanmean(m_temp))

    x = np.arange(len(n_range))  
    width = 0.35                

    fig, ax = plt.subplots(figsize=(10, 6))

    fig = plot_grouped_bar(
        mean_data={
            "Bilevel": bilevel_times,
            "MaxEnt": maxent_times
        },
        labels=[str(n) for n in n_range],
        ax=ax,
        title="Mean Solving Times by Method and Problem Size"
    )

    plt.tight_layout()
    plt.show()

np.random.seed(42)

run_performance_comparison(n_range=[4,16, 32, 64], m_range=[2,4,8,16], n_problems=30)
plot_scaling_n(n_range=[4, 16, 32, 64], m_static=3, trials=5)
plot_scaling_m(n_static=4, m_range=[4, 16, 32, 64], trials=5)
plot_grouped_bar_performance(n_range=[4,8,16,32], m_static=3, trials=5)