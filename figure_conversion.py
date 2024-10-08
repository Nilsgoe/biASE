import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
from ase.io import Trajectory
from ase.lattice.cubic import FaceCenteredCubic
from ase.optimize import FIRE
from ase.build import fcc100, add_adsorbate
from ase.calculators.lammpsrun import LAMMPS


traj = Trajectory(f'metad.traj')
atoms = traj[-1]

fig,ax= plt.subplots(1,figsize=(7, 7))
color = ['white']*(len(atoms)-2) + ['#006c66'] + ['#006c66']
plot_atoms(atoms, ax, colors=color,radii=0.9)
plt.axis('off')
fig.savefig(f"final_metad.png",transparent=True)

atoms = traj[0]

fig,ax= plt.subplots(1,figsize=(7, 7))
color = ['white']*(len(atoms)-2) + ['#006c66'] + ['#006c66']
plot_atoms(atoms, ax, colors=color,radii=0.9)
plt.axis('off')
fig.savefig(f"initial_metad.png",transparent=True)
