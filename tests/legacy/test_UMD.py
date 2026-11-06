from ase.io import read
from mace.calculators import mace_off
import jax.numpy as jnp
from ase.optimize import BFGS, FIRE
from ase.md.langevin import Langevin
from ase import units
from Umbrella_sampling import UmbrellaMD


def compute_cv(positions):
    # Indices: 2 for O, 3 for C5, 1 for C2, and 5 for C3
    d_O4H8 = jnp.linalg.norm(positions[4] - positions[8]) 
    d_O3H8 = jnp.linalg.norm(positions[3] - positions[8]) 
    return d_O4H8 -d_O3H8

mace_calc_dis = mace_off(model="large",dispersion=True)
atoms1 = read("optimized_malonaldehyde_initial.xyz")
atoms = atoms1.copy()
for i in range(5):
    atoms.calc = mace_calc_dis
    UMD = UmbrellaMD(atoms=atoms,cv_function=compute_cv,x0=compute_cv(atoms1.positions)+.2*i,k=50,md_method=Langevin(atoms, timestep=.5 * units.fs, temperature_K=293, friction=0.1, trajectory=f'umd_{i}.traj',fixcm=False))

    UMD.optimize_in_window(optimizer=BFGS,fmax=0.05,max_steps=500,logfile=f"BFGS_{i}.log")
    UMD.run(200)