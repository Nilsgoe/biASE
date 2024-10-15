import matplotlib.pyplot as plt
import cmocean.cm as cmo
import numpy as np
import subprocess

o_file='fes_new.dat'
# Reconstruct the free energy from HILLS using the plumed tool "sum_hills" :
p = subprocess.Popen(f"plumed sum_hills --hills test3.txt --outfile {o_file}" 
                     " --bin 300,300 --min 0.3,-0.35 --max 1.2,1.56",
                     shell=True, stdout=subprocess.PIPE)
p.wait()

# Import free energy and reshape with the number of bins defined in the
# reconstruction process.
scm = np.loadtxt(o_file, usecols=0).reshape(301, 301)#was 201 everywhere
tcm = np.loadtxt(o_file, usecols=1).reshape(301, 301)
fes = np.loadtxt(o_file, usecols=2).reshape(301, 301)

# Plot
fig, ax = plt.subplots(figsize=(10, 10))

# Plot free energy surface
im = ax.contourf(scm, tcm, fes, 10, cmap=cmo.tempo_r)
cp = ax.contour(scm, tcm, fes, 10,
                linestyles='-', colors='darkgray', linewidths=1.3)

# Plot parameters
#ax.set_xlabel('SCM', fontsize=40)
#ax.set_ylabel('TCM', fontsize=40)
ax.tick_params(axis='y', labelsize=32)
ax.tick_params(axis='x', labelsize=32)
cbar = fig.colorbar(im, ax=ax)
#cbar.set_label(label=r'FES[$\epsilon$]', fontsize=40)
cbar.ax.tick_params(labelsize=38)
#plt.scatter()
plt.tight_layout()
plt.savefig('biASE_plot.png',dpi=300)
plt.show()
