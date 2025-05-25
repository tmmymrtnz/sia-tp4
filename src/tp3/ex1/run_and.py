import sys
sys.path.insert(0, "src")
from ex1.runner import run_experiment

if len(sys.argv) != 2:
    print("Usage: python run_and.py <config_path.json>")
    sys.exit(1)

X = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
Y = [-1, -1, -1, 1]

run_experiment(sys.argv[1], X, Y)