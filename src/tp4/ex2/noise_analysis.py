import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from runner_hopfield import read_patterns, train, multi_run_recovery_stats, print_pattern
import random


def analyze_noise_impact(patterns_path, noise_levels, max_steps, n_runs=10):
    """Analyze how different noise levels affect pattern recovery"""
    # Load patterns and train network
    patterns = read_patterns(patterns_path)
    W = train(patterns)
    target = patterns[0]  # Use first pattern as target
    N = len(target)
    
    # Data storage for analysis
    final_similarities = []  # Average final similarity for each noise level
    convergence_steps = []   # Average steps to convergence for each noise level
    recovery_rates = []      # Percentage of runs that fully recover the pattern
    
    # Process each noise level
    for noise in noise_levels:
        print(f"\nAnalyzing noise level: {noise*100:.1f}%")
        
        # Generate noisy patterns for this noise level
        noisy_inits = []
        for _ in range(n_runs):
            flips = max(1, round(noise * N))
            p = target.copy()
            for i in random.sample(range(N), flips):
                p[i] *= -1
            noisy_inits.append(p)
        
        # Run recovery analysis
        energy_stats, sim_stats = multi_run_recovery_stats(
            W, noisy_inits, target, max_steps, n_runs
        )
        
        # Extract metrics
        mean_sims = sim_stats[0]  # Mean similarities at each step
        
        # Calculate convergence step (when similarity stops changing)
        steps_to_converge = []
        for run_idx in range(n_runs):
            converged_at = max_steps - 1  # Default to max if no convergence
            for step in range(1, max_steps):
                if mean_sims[step] == mean_sims[step-1]:
                    converged_at = step
                    break
            steps_to_converge.append(converged_at)
        
        # Calculate recovery rate (perfect recovery = 100% similarity)
        perfect_recoveries = sum(1 for s in mean_sims[-1:] if s == N)
        recovery_rate = perfect_recoveries / n_runs * 100
        
        # Save metrics for this noise level
        final_similarities.append(mean_sims[-1] / N * 100)  # As percentage
        convergence_steps.append(sum(steps_to_converge) / n_runs)
        recovery_rates.append(recovery_rate)
        
        print(f"  Final similarity: {final_similarities[-1]:.2f}%")
        print(f"  Avg. steps to converge: {convergence_steps[-1]:.2f}")
        print(f"  Full recovery rate: {recovery_rates[-1]:.2f}%")
    
    return noise_levels, final_similarities, convergence_steps, recovery_rates


def plot_noise_analysis(noise_levels, final_similarities, convergence_steps, recovery_rates):
    """Create visualizations of noise impact analysis"""
    # Convert noise levels to percentages for better readability
    x_axis = [n*100 for n in noise_levels]
    
    # Figure 1: Final similarity and recovery rates
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot final similarity
    ax1.set_xlabel('Noise Level (%)')
    ax1.set_ylabel('Final Similarity (%)', color='tab:blue')
    ax1.plot(x_axis, final_similarities, '-o', color='tab:blue', label='Final Similarity')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)
    
    # Plot recovery rate on secondary axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Full Recovery Rate (%)', color='tab:green')
    ax2.plot(x_axis, recovery_rates, '-s', color='tab:green', label='Recovery Rate')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title('Impact of Noise on Pattern Recovery')
    plt.tight_layout()
    plt.savefig('noise_impact_similarity.png')
    plt.show()
    
    # Figure 2: Convergence speed
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, convergence_steps, '-o', color='tab:red')
    plt.xlabel('Noise Level (%)')
    plt.ylabel('Average Steps to Convergence')
    plt.title('Impact of Noise on Convergence Speed')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('noise_impact_convergence.png')
    plt.show()


def noise_threshold_analysis(patterns_path, max_steps=50, n_runs=20, step_size=0.05, n_experiments=5):
    """Find the threshold where pattern recovery significantly degrades with statistical robustness"""
    # Generate noise levels from 0.1 (10%) to 0.5 in small increments
    noise_levels = np.arange(0.1, 0.55, step_size)
    
    # Storage for multiple experiment results
    all_similarities = []
    all_convergence = []
    all_recovery_rates = []
    
    print(f"\nRunning {n_experiments} complete experiments for statistical robustness...")
    
    # Run multiple complete experiments
    for exp in range(n_experiments):
        print(f"\nExperiment {exp+1}/{n_experiments}")
        noise_data = analyze_noise_impact(patterns_path, noise_levels, max_steps, n_runs)
        _, similarities, convergence, recovery = noise_data
        all_similarities.append(similarities)
        all_convergence.append(convergence)
        all_recovery_rates.append(recovery)
    
    # Calculate statistics across experiments
    avg_similarities = np.mean(all_similarities, axis=0)
    std_similarities = np.std(all_similarities, axis=0)
    avg_convergence = np.mean(all_convergence, axis=0)
    std_convergence = np.std(all_convergence, axis=0)
    avg_recovery = np.mean(all_recovery_rates, axis=0)
    std_recovery = np.std(all_recovery_rates, axis=0)
    
    # Plot results with error bars
    plot_noise_analysis_with_error_bars(
        noise_levels, 
        avg_similarities, std_similarities,
        avg_convergence, std_convergence,
        avg_recovery, std_recovery
    )
    
    # Find critical thresholds based on average values
    recovery_threshold = None
    for i, rate in enumerate(avg_recovery):
        if rate < 50:
            recovery_threshold = noise_levels[i]
            break
    
    # Find where similarity drops most dramatically (maximum negative slope)
    slopes = [avg_similarities[i+1] - avg_similarities[i] for i in range(len(avg_similarities)-1)]
    max_drop_idx = slopes.index(min(slopes))
    critical_noise = noise_levels[max_drop_idx]
    
    print("\n=== Noise Threshold Analysis ===")
    if recovery_threshold is not None:
        print(f"Recovery rate falls below 50% at noise level: {recovery_threshold*100:.1f}%")
    print(f"Critical noise threshold (max degradation): {critical_noise*100:.1f}%")
    print(f"Slope at critical threshold: {slopes[max_drop_idx]:.2f}% similarity per {step_size*100:.1f}% noise")
    print(f"\nStatistical summary:")
    print(f"- Average standard deviation in similarity: {np.mean(std_similarities):.2f}%")
    print(f"- Maximum standard deviation in similarity: {np.max(std_similarities):.2f}% at noise level {noise_levels[np.argmax(std_similarities)]*100:.1f}%")


def plot_noise_analysis_with_error_bars(noise_levels, avg_similarities, std_similarities,
                                       avg_convergence, std_convergence, avg_recovery, std_recovery):
    """Create visualizations of noise impact analysis with error bars for statistical significance"""
    # Convert noise levels to percentages for better readability
    x_axis = [n*100 for n in noise_levels]
    
    # Figure 1: Final similarity and recovery rates with error bars
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot final similarity with error bars
    ax1.set_xlabel('Noise Level (%)')
    ax1.set_ylabel('Final Similarity (%)', color='tab:blue')
    ax1.errorbar(x_axis, avg_similarities, yerr=std_similarities, fmt='-o', color='tab:blue', 
                 label='Final Similarity', capsize=5)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)
    
    # Plot recovery rate on secondary axis with error bars
    ax2 = ax1.twinx()
    ax2.set_ylabel('Full Recovery Rate (%)', color='tab:green')
    ax2.errorbar(x_axis, avg_recovery, yerr=std_recovery, fmt='-s', color='tab:green', 
                label='Recovery Rate', capsize=5)
    ax2.tick_params(axis='y', labelcolor='tab:green')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title('Impact of Noise on Pattern Recovery (with Statistical Variance)')
    plt.tight_layout()
    plt.savefig('noise_impact_similarity_stats.png')
    plt.show()
    
    # Figure 2: Convergence speed with error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(x_axis, avg_convergence, yerr=std_convergence, fmt='-o', color='tab:red', capsize=5)
    plt.xlabel('Noise Level (%)')
    plt.ylabel('Average Steps to Convergence')
    plt.title('Impact of Noise on Convergence Speed (with Statistical Variance)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('noise_impact_convergence_stats.png')
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2 and len(sys.argv) != 1:
        print("Usage: python noise_analysis.py [config.json]")
        sys.exit(1)
    
    # Default config if none provided
    config = {
        "letters_path": "/Users/saints/Desktop/ITBA/SIA/sia-tp4/data/tp4/letters.txt",
        "max_steps": 50,
        "n_runs": 20,
        "noise_step": 0.05,
        "n_experiments": 5  # New parameter for number of complete experiment repetitions
    }
    
    # Read config if provided
    if len(sys.argv) == 2:
        with open(sys.argv[1], 'r') as f:
            config.update(json.load(f))
    
    # Run the analysis
    print("\n=== Hopfield Network Noise Analysis ===")
    print(f"Pattern file: {config['letters_path']}")
    print(f"Analysis configuration: {config['n_experiments']} complete experiments, "
          f"{config['n_runs']} runs per noise level, max {config['max_steps']} steps, "
          f"{config['noise_step']*100:.1f}% noise increments\n")
    
    noise_threshold_analysis(
        config['letters_path'],
        config['max_steps'],
        config['n_runs'],
        config['noise_step'],
        config['n_experiments']
    )