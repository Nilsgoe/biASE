import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# Load data
o_file = "fes_new2.dat"
my_file = "output_bias_1D.txt"

scm = np.loadtxt(o_file, usecols=0)  # Collective variable values
fes = np.loadtxt(o_file, usecols=1)  # FES from MetaD
my_fes = np.loadtxt(my_file, usecols=1)  # FES computed via script

# Check if my_fes and fes match within the tolerance of 1e-4
difference = np.abs(fes - my_fes)
tolerance = 1e-6
max_difference = np.max(difference)
all_within_tolerance = np.all(difference < tolerance)

print(f"Maximum difference: {max_difference:.6e}")
print(f"All values within tolerance (1e-6): {all_within_tolerance}")

# Plot comparison
fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(scm, fes, label="MetaD FES", linewidth=2, color="blue")
ax.plot(scm, my_fes, label="Computed FES (Script)", linestyle="--", linewidth=2, color="orange")

ax.set_xlabel('CV', size=32)
ax.set_ylabel('FES', size=32)
ax.tick_params(axis='y', labelsize=24)
ax.tick_params(axis='x', labelsize=24)
ax.xaxis.set_major_locator(MaxNLocator(nbins=7, prune=None))

# Legend and Title
plt.legend(fontsize=20)
plt.title("Comparison of FES (MetaD vs Computed)", size=40)
plt.tight_layout()

# Save and show plot
plt.savefig('FES_Comparison.png', dpi=300)
plt.show()

