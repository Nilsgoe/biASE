from biASE import General_Bias_Calculator
from ase.atoms import Atoms
import numpy as np
from ase.io import read
import csv
from mace.calculators import mace_mp
import jax.numpy as jnp
from ase.build import fcc111, add_adsorbate,molecule
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ase.calculators.mixing import SumCalculator
from functools import partial
from ase.md.langevin import Langevin
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

import argparse

parser = argparse.ArgumentParser(description="Umbrella MD shifted from the a CV by input*0.05")
parser.add_argument('shift',type=int, help="Input value for the shift")
args = parser.parse_args()


def diff_theta(positions,indices=[65,64,66],factor=[0.7,-0.71]):
    p1 = positions[indices[0]]
    p2 = positions[indices[1]]
    p3 = positions[indices[2]]

    bondlength = jnp.linalg.norm(p2 - p1)

    d_p3_p1 = p3 - p1
    norm_p3_p1 = jnp.linalg.norm(d_p3_p1)

    # Get angle with z axis
    z_axis = jnp.array([0, 0, 1])
    theta = jnp.arccos(jnp.dot(z_axis, d_p3_p1)/(jnp.linalg.norm(z_axis)*norm_p3_p1))
    #print(bondlength,theta)
    cv_value = factor[0] * bondlength + factor[1] * theta
    return cv_value

def umbrella_potential(x, x0, k):
    dx = x - x0

    return 0.5 * k * dx**2

mace_calc = mace_mp(model="medium", dispersion=False, default_dtype="float64",device='cuda:1', )


slab = fcc111('Rh', size=(4, 4, 4), vacuum=10.0)

# Define the HCO molecule
HCO = Atoms('HCO', positions=[(0, 0, 0), (1.1, 0, 0), (1.1, 1.2, 0)])

# Define the adsorption height for HCO above the hollow site (adjust as necessary)
adsorption_height = 1.5  # distance in Angstroms from the slab

# Add HCO to the hollow site of the slab (hollow = position between 3 neighboring atoms)
# You can specify 'hollow' directly to target the hollow site
add_adsorbate(slab, HCO, adsorption_height, position=(5,3))

# Freeze the lowest layer of the slab (z_threshold sets the cutoff for the lowest layer)
z_threshold = min(slab.positions[:, 2]) + 0.1
constraint = FixAtoms(indices=[atom.index for atom in slab if atom.position[2] < z_threshold])
slab.set_constraint(constraint)
initial=slab.copy()



mace_calc_dis = mace_mp(model="medium", dispersion=True, default_dtype="float64",device='cuda', )

print(args.shift)
umbrella_bias_calc = General_Bias_Calculator(cv_function=diff_theta,bias_function=partial(umbrella_potential,x0=diff_theta(initial.positions)+0.05*args.shift,k=50.0))
biased_mace_calc=SumCalculator([mace_calc_dis,umbrella_bias_calc])
atoms = initial.copy()
atoms.calc = biased_mace_calc
BFGS(atoms).run(fmax=0.05,steps=200)
MaxwellBoltzmannDistribution(slab, temperature_K=573)
dyn = Langevin(atoms, timestep=.5 * units.fs, temperature_K=573, friction=0.1, trajectory=f'HCO_on_Rh_dis_biased_{args.shift}_large.traj',fixcm=False)#Andersen(atoms, timestep=.5 * units.fs, temperature_K=3000, andersen_prob=0.01,trajectory='md.traj')#


N_steps = int(5e4)
dyn.run(N_steps)
traj = read(f'HCO_on_Rh_dis_biased_{i}_large.traj@2000::1')
cv_list=[]
energy_list=[]
for atoms in traj:
    atoms.calc=biased_mace_calc
    energy=atoms.get_total_energy()
    cv= diff_theta(atoms.positions)
    with open(f"largecv_energy_biased_dis_{args.shift}.csv", 'w') as file:
        writer =csv.writer(file)
        writer.writerow([cv,energy])
        
    energy=atoms.get_total_energy()
    cv= diff_theta(atoms.positions)
    cv_list.append(cv)
    energy_list.append(energy)
with open(f"large_mean_cv_energy_dis_{args.shift}.csv", 'w') as file:
    writer =csv.writer(file)
    writer.writerow([args.shift*0.05+initial.positions,np.mean(np.array(cv_list)),-50*(np.mean(np.array(cv_list))-(diff_theta(initial.positions)+0.05*i)),np.mean(np.array(energy_list))])
        
    