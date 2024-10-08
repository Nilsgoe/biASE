from ase.atoms import Atoms
from ase import units
from ase.io import read


from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution,Stationary, ZeroRotation
from ase.constraints import FixedPlane

import jax.numpy as jnp
from Metadynamics import WT_Metadynamics
from functools import partial
from ase import units
from ase.io import read
from ase.constraints import FixedPlane
from ase.calculators.lj import LennardJones



def _cvs_moments(atoms_positions, cutoff):
    """ Calculate coordination numbers for all atoms in an ASE Atoms object.
        And then calculate the second and third central moments of the coordination number distribution using numpy.
    """
    dist_matrix = jnp.sqrt(jnp.sum((atoms_positions[:, jnp.newaxis] - atoms_positions) ** 2, axis=-1))
    x_matrix = (1 - (dist_matrix / cutoff) ** 8) / (1 - (dist_matrix / cutoff) ** 16)
    #print(x_matrix.shape[0])
    x_matrix = x_matrix.at[jnp.diag_indices(x_matrix.shape[0])].set(0) # there is no jnp.fill_diagonal(x_matrix, 0)  # Set diagonal elements to zero
    coord_nums = jnp.sum(x_matrix, axis=1)
    #print(coord_nums)
    mean = jnp.mean(coord_nums)
    scm = jnp.sum((coord_nums - mean)**2) / len(coord_nums)
    tcm = jnp.sum((coord_nums - mean)**3) / len(coord_nums)
    #print(scm, tcm)
    return scm, tcm

timestep = 0.005
ps = 1000 * units.fs

for i in range(10):
 
    print(i)
    atoms = read('isomer.xyz')
    cons = [FixedPlane(i, [0, 0, 1]) for i in range(7)]
    atoms.set_constraint(cons)
    atoms.set_masses([1, 1, 1, 1, 1, 1, 1])


    
    atoms.calc = LennardJones(rc=2.5, r0=3.0)

    
    dyn = WT_Metadynamics(atoms, timestep=timestep, temperature_K=0.1/units.kB, friction=1, trajectory=None
                        ,fixcm=False,cvs=partial(_cvs_moments, cutoff=1.5),interval_size=500,std_dev=0.1,bias_height=.05,output_file=f'biase_comp_11_{i}.txt',well_temp=0.1,stretch=True) #Langevin(atoms, timestep=.5 * units.fs, temperature_K=0.1/units.kB, friction=1, trajectory='metad.traj',fixcm=False) #
  
    dyn.run(int(1e6))
