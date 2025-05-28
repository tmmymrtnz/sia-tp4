import random
import json
import matplotlib.pyplot as plt
import sys

def read_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def read_patterns(file_path):
    with open(file_path, 'r') as f:
        blocks = f.read().strip().split('\n\n')
        patterns = []
        for block in blocks:
            matrix = [list(map(int, row.split())) for row in block.strip().split('\n')]
            flat = [bit for row in matrix for bit in row]
            patterns.append(flat)
        return patterns

def print_pattern(vector):
    for i in range(5):
        row = vector[i*5:(i+1)*5]
        print(' '.join(['*' if x == 1 else ' ' for x in row]))
    print()

def train(patterns):
    N = len(patterns[0])
    W = [[0]*N for _ in range(N)]
    for p in patterns:
        for i in range(N):
            for j in range(N):
                if i != j:
                    W[i][j] += p[i] * p[j]
    return W

def sign(x):
    return 1 if x >= 0 else -1

def update_state(W, state):
    N = len(state)
    new_state = state.copy()
    for i in range(N):
        s = sum(W[i][j] * new_state[j] for j in range(N))
        new_state[i] = sign(s)
    return new_state

def energy(W, state):
    e = 0
    for i in range(len(state)):
        for j in range(len(state)):
            if i != j:
                e -= W[i][j] * state[i] * state[j]
    return e / 2

def similarity(state, target):
    return sum(1 for i in range(len(state)) if state[i] == target[i])

def multi_run_recovery_stats(W, initial_states, target, max_steps, n_runs):
    """
    initial_states: a list of length-n_runs vectors, each the initial state to test
    """
    all_energies = []
    all_sims = []
    for state0 in initial_states:
        state = state0.copy()
        energies = []
        sims = []
        for step in range(max_steps):
            e = energy(W, state)
            s = similarity(state, target)
            energies.append(e)
            sims.append(s)

            new_state = update_state(W, state)
            if new_state == state:
                # pad out the rest
                energies += [e] * (max_steps - step - 1)
                sims     += [s] * (max_steps - step - 1)
                break
            state = new_state

        # if never converged early, energies/sims already length max_steps
        all_energies.append(energies)
        all_sims.append(sims)

    # transpose and compute stats
    mean_e = [sum(col)/n_runs for col in zip(*all_energies)]
    min_e  = [min(col)       for col in zip(*all_energies)]
    max_e  = [max(col)       for col in zip(*all_energies)]
    mean_s = [sum(col)/n_runs for col in zip(*all_sims)]
    min_s  = [min(col)       for col in zip(*all_sims)]
    max_s  = [max(col)       for col in zip(*all_sims)]

    return (mean_e, min_e, max_e), (mean_s, min_s, max_s)

def plot_stats(energy_stats, sim_stats, max_steps, title):
    steps = list(range(1, max_steps+1))
    mean_e, min_e, max_e = energy_stats
    mean_s, min_s, max_s = sim_stats

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Energy', color='tab:blue')
    ax1.plot(steps, mean_e, '-', label='Mean Energy', color='tab:blue')
    ax1.fill_between(steps, min_e, max_e, color='tab:blue', alpha=0.2, label='Energy range')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Similarity', color='tab:green')
    ax2.plot(steps, mean_s, '-', label='Mean Similarity', color='tab:green')
    ax2.fill_between(steps, min_s, max_s, color='tab:green', alpha=0.2, label='Similarity range')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    fig.tight_layout()
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python hopfield.py <config.json>")
        sys.exit(1)
    cfg = read_config(sys.argv[1])

    letters_path   = cfg["letters_path"]
    noise_fraction = cfg["noise_fraction"]
    max_steps      = cfg["max_steps"]
    n_runs         = cfg.get("n_runs", 10)
    plot_results   = cfg.get("plot_results", True)

    patterns = read_patterns(letters_path)
    W = train(patterns)

    # prepare noisy initial states
    N = len(patterns[0])
    noisy_inits = []
    for _ in range(n_runs):
        flips = max(1, round(noise_fraction * N))
        p = patterns[0].copy()
        for i in random.sample(range(N), flips):
            p[i] *= -1
        noisy_inits.append(p)

    print(f"\n--- Multi-run recovery from noisy pattern ({n_runs} runs, noise={noise_fraction*100:.1f}%) ---\n")
    energy_stats_noisy, sim_stats_noisy = multi_run_recovery_stats(
        W, noisy_inits, patterns[0], max_steps, n_runs
    )
    if plot_results:
        plot_stats(energy_stats_noisy, sim_stats_noisy, max_steps,
                   title="Noisy-pattern recovery (mean ± min/max)")

    # prepare random initial states
    random_inits = [[random.choice([1, -1]) for _ in range(N)] for _ in range(n_runs)]

    print(f"\n--- Multi-run recovery from random patterns ({n_runs} runs) ---\n")
    energy_stats_rand, sim_stats_rand = multi_run_recovery_stats(
        W, random_inits, patterns[0], max_steps, n_runs
    )
    if plot_results:
        plot_stats(energy_stats_rand, sim_stats_rand, max_steps,
                   title="Random-start recovery (mean ± min/max)")
