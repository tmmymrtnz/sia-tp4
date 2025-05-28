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

def plot_evolution(energies, similarities):
    steps = list(range(1, len(energies) + 1))
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Energy', color='tab:blue')
    ax1.plot(steps, energies, 'o-', label='Energy', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Similarity (correct bits)', color='tab:green')
    ax2.plot(steps, similarities, 's--', label='Similarity', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    plt.title("Recovery Evolution")
    plt.tight_layout()
    plt.show()

def recover(W, initial_state, target, max_steps=10, show=True, plot=True):
    state = initial_state.copy()
    energies, sims = [], []

    for step in range(max_steps):
        e = energy(W, state)
        energies.append(e)
        s = similarity(state, target)
        sims.append(s)

        if show:
            print(f"Step {step+1}: Energy={e:.2f}, Similarity={s}/{len(target)}")
            print_pattern(state)

        new_state = update_state(W, state)
        if new_state == state:
            if show:
                print("Converged.\n")
            break
        state = new_state

    if plot:
        plot_evolution(energies, sims)

    return state

def add_noise(pattern, noise_fraction=0.1):
    """Flip approximately noise_fraction of bits (min 1)."""
    N = len(pattern)
    flips = max(1, round(noise_fraction * N))
    noisy = pattern.copy()
    indices = random.sample(range(N), flips)
    for i in indices:
        noisy[i] *= -1
    return noisy

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python hopfield.py <config.json>")
        sys.exit(1)
    cfg = read_config(sys.argv[1])
    letters_path   = cfg["letters_path"]
    noise_fraction = cfg["noise_fraction"]
    max_steps      = cfg["max_steps"]
    plot_results   = cfg["plot_results"]

    patterns = read_patterns(letters_path)

    print("Original patterns:\n")
    for p in patterns:
        print_pattern(p)

    W = train(patterns)

    print("\n--- Recovery from noised pattern ---\n")
    noisy_pattern = add_noise(patterns[0], noise_fraction)
    print_pattern(noisy_pattern)
    _ = recover(W, noisy_pattern, patterns[0],
                max_steps=max_steps,
                show=True,
                plot=plot_results)

    print("\n--- Testing with random initial state ---\n")
    random_pattern = [random.choice([1, -1]) for _ in range(25)]
    print_pattern(random_pattern)
    final = recover(W, random_pattern, patterns[0],
                    max_steps=max_steps,
                    show=True,
                    plot=plot_results)
    print("Matched a known pattern." if final in patterns else "Detected a spurious state.")
