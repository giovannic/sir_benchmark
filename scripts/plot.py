import csv
import sys
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Read in CLI arguments
n_params = int(sys.argv[1])
run_path = sys.argv[2]
output_path = sys.argv[3]

# Read in csv file
with(open(run_path, 'r')) as f:
    reader = csv.reader(f)
    S, I, R = [], [], []
    current_run = None
    next(reader) # Skip header
    for row in reader:
        if current_run != row[0]:
            S.append([])
            I.append([])
            R.append([])
            current_run = row[0]
        S[-1].append(row[2])
        I[-1].append(row[3])
        R[-1].append(row[4])

# Convert to numpy arrays
S, I, R = np.array(S, dtype=np.int64), np.array(I, dtype=np.int64), np.array(R, dtype=np.int64)

# Plot data
fig, axs = plt.subplots(1, n_params, figsize=(n_params*6, 6))
t = np.arange(len(S[0]))
colors = ['blue', 'red', 'green']  # Define colors for S, I, R
for j in range(len(S) // n_params):
    for i in range(n_params):
        idx = j * n_params + i
        axs[i].plot(t, S[idx], label='S' if j == 0 else '', color=colors[0])
        axs[i].plot(t, I[idx], label='I' if j == 0 else '', color=colors[1])
        axs[i].plot(t, R[idx], label='R' if j == 0 else '', color=colors[2])
        axs[i].legend()
        axs[i].set_title(f'Params {i}')

plt.tight_layout()
plt.savefig(output_path)