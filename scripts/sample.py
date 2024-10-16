# Take in max_samples and parameter path from command line
import sys
import random

max_samples = int(sys.argv[1])
param_path = sys.argv[2]
seed = float(sys.argv[3])

random.seed(seed)

# Open path for writing parameter samples
with open(param_path, "w") as param_file:
    # Sample R0, I0, gamma from uniform distribution
    for _ in range(max_samples):
        R0 = random.random()
        I0 = random.random()
        gamma = random.random()
        param_file.write(f"{R0},{I0},{gamma}\n")