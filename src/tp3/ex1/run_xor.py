import sys
from runner import run_experiment

if len(sys.argv) != 2:
    print("Usage: python run_xor.py <config_path.json>")
    sys.exit(1)

X = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
Y = [1, 1, -1, -1]

run_experiment(sys.argv[1], X, Y)