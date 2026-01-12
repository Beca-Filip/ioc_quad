from ioc_quad.core import MultiObjectiveOptimizer, InverseOptimalControl, MaximumEntropyIRL

import numpy as np
import matplotlib.pyplot as plt
import time
import random

def run_performance_comparison(n_problems=50):
    """Run performance comparison on combinations of n and m."""
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
        _, _, _, t_m, _, it_m = maxent.solve_inverse(max_iterations=100)
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
            z_ref = opt.solve(np.random.dirichlet(np.ones(m)).reshape(-1, 1))
            
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
    
    titles = ["Solving times", "Number of iterations"]
    ylabels = ["Timings (s)", "Number of iterations"]
    metrics = ["times", "iters"]
    limits = [time_limit, iter_limit]
    
    styles = {
        "Bilevel": {"color": "#1f77b4", "marker": "o", "label": "Bilevel IOC"}, 
        "MaxEnt": {"color": "#ff7f0e", "marker": "o", "label": "MaxEnt IRL"}  
    }

    for ax, metric, title, ylabel, limit in zip([ax1, ax2], metrics, titles, ylabels, limits):
        for m in methods:
            medians = []
            err_low = []
            err_high = []
            
            for n in n_range:
                vals = np.array(raw_data[m][n][metric])
                med = np.median(vals)
                medians.append(med)
                err_low.append(med - np.min(vals))
                err_high.append(np.max(vals) - med)
            
            ax.errorbar(n_range, medians, yerr=[err_low, err_high], 
                        fmt=styles[m]['marker'], color=styles[m]['color'],
                        ecolor=styles[m]['color'], capsize=0, elinewidth=1, 
                        label=styles[m]['label'], markersize=6)

        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.yaxis.grid(True, which='both', color='blue', linestyle='-', linewidth=0.5, alpha=0.7)
        ax.xaxis.grid(False) 

        ax.set_xlabel('d', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.legend(loc='lower right', frameon=True)
        
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
            z_ref = opt.solve(np.random.dirichlet(np.ones(m)).reshape(-1, 1))
            
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
    
    titles = ["Solving times", "Number of iterations"]
    ylabels = ["Timings (s)", "Number of iterations"]
    metrics = ["times", "iters"]
    limits = [time_limit, iter_limit]
    
    styles = {
        "Bilevel": {"color": "#1f77b4", "marker": "o", "label": "Bilevel IOC"}, 
        "MaxEnt": {"color": "#ff7f0e", "marker": "o", "label": "MaxEnt IRL"}  
    }

    for ax, metric, title, ylabel, limit in zip([ax1, ax2], metrics, titles, ylabels, limits):
        for method in methods:
            medians = []
            err_low = []
            err_high = []
            
            for m in m_range:
                vals = np.array(raw_data[method][m][metric])
                med = np.median(vals)
                medians.append(med)
                err_low.append(med - np.min(vals))
                err_high.append(np.max(vals) - med)
            
            ax.errorbar(m_range, medians, yerr=[err_low, err_high], 
                        fmt=styles[method]['marker'], color=styles[method]['color'],
                        ecolor=styles[method]['color'], capsize=0, elinewidth=1, 
                        label=styles[method]['label'], markersize=6)

        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.yaxis.grid(True, which='both', color='blue', linestyle='-', linewidth=0.5, alpha=0.7)
        ax.xaxis.grid(False) 

        ax.set_xlabel('d', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.legend(loc='lower right', frameon=True)
        
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
            z_ref = opt.solve(np.random.dirichlet(np.ones(m)).reshape(-1, 1))
            
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
    
    rects1 = ax.bar(x - width/2, bilevel_times, width, label='Bilevel IOC', 
                    color=color_bilevel, edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x + width/2, maxent_times, width, label='MaxEnt IRL', 
                    color=color_maxent, edgecolor='black', linewidth=0.8)

    ax.set_yscale('log')
    
    ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.5, color='gray')
    ax.set_axisbelow(True) 

    ax.set_ylabel('Mean computation time [s]', fontweight='bold')
    ax.set_xlabel('Dimension n (n_vars)', fontweight='bold')
    ax.set_title('Computational Cost Comparison', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'n={n}' for n in n_range])
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, edgecolor='black')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

np.random.seed(42)

run_performance_comparison()
#plot_scaling_n(n_range=[2,4,8,16,32], m_static=3, trials=5)
#plot_scaling_m(n_static=4, m_range=[2,4,8,16,32], trials=5)
#plot_grouped_bar_performance(n_range=[2,4,8,16], m_static=3, trials=5)