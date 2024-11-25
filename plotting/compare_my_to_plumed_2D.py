import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import cmocean.cm as cmo

# Load data
o_file = "fes_new2D.dat"
my_file = "output_bias_2D.txt"

scm = np.loadtxt(o_file, usecols=0).reshape(301,301)  # Collective variable values
tcm = np.loadtxt(o_file, usecols=1).reshape(301,301) 
fes = np.loadtxt(o_file, usecols=2).reshape(301,301)   # FES from MetaD
scm_my = np.loadtxt(my_file, usecols=0).reshape(301,301)   # Collective variable values
tcm_my = np.loadtxt(my_file, usecols=1).reshape(301,301) 
fes_my = np.loadtxt(my_file, usecols=2).reshape(301,301)   # FES computed via script

# Check if my_fes and fes match within the tolerance of 1e-6
difference = np.abs(fes - fes_my)
print(difference)
tolerance = 1e-6
max_difference = np.max(difference)
all_within_tolerance = np.all(difference < tolerance)

print(f"Maximum difference: {max_difference:.6e}")
print(f"All values within tolerance (1e-6): {all_within_tolerance}")

# Plot comparison
# Plot
fig, ax = plt.subplots(figsize=(10, 10))

# Plot free energy surface
im = ax.contourf(scm, tcm, fes, 10, cmap=cmo.tempo_r)
cp = ax.contour(scm, tcm, fes, 10,
                linestyles='-', colors='darkgray', linewidths=1.3)

im_my = ax.contourf(scm_my, tcm_my, fes_my, 10, cmap=cmo.tempo_i)
cp_my = ax.contour(scm_my, tcm_my, fes_my, 10,
                linestyles='--', colors='blue', linewidths=1.3)


# Plot parameters
#ax.set_xlabel('SCM', fontsize=40)
#ax.set_ylabel('TCM', fontsize=40)
ax.tick_params(axis='y', labelsize=32)
ax.tick_params(axis='x', labelsize=32)
cbar = fig.colorbar(im, ax=ax)
cbar_my = fig.colorbar(im_my, ax=ax)
#cbar.set_label(label=r'FES[$\epsilon$]', fontsize=40)
cbar.ax.tick_params(labelsize=38)
#plt.scatter()
plt.tight_layout()
plt.savefig('Comparison_plot_2D.png',dpi=300)
plt.show()
