import csv
import sys
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Read in CLI arguments
run_path = sys.argv[1]
output_path = sys.argv[2]

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
t = np.arange(len(S[0]))
colors = ['blue', 'red', 'green']  # Define colors for S, I, R
for i in range(len(S)):
    plt.plot(t, S[i], label='S' if i == 0 else None, color=colors[0])
    plt.plot(t, I[i], label='I' if i == 0 else None, color=colors[1])
    plt.plot(t, R[i], label='R' if i == 0 else None, color=colors[2])

plt.legend()
plt.savefig(output_path)