from ase import units
from ase.io import read
from ase.md.langevin import Langevin
from ase.constraints import FixedPlane
from ase.calculators.plumed import Plumed
from ase.calculators.lj import LennardJones
import copy
import numpy as np

timestep = 0.005
ps = 1000 * units.fs

setup = [f"UNITS LENGTH=A TIME={1/ps} ENERGY={units.mol/units.kJ}",
         "COM ATOMS=1-7 LABEL=com",
         "DISTANCE ATOMS=1,com LABEL=d1",
         "UPPER_WALLS ARG=d1 AT=2.0 KAPPA=100.",
         "DISTANCE ATOMS=2,com LABEL=d2",
         "UPPER_WALLS ARG=d2 AT=2.0 KAPPA=100.",
         "DISTANCE ATOMS=3,com LABEL=d3",
         "UPPER_WALLS ARG=d3 AT=2.0 KAPPA=100.",
         "DISTANCE ATOMS=4,com LABEL=d4",
         "UPPER_WALLS ARG=d4 AT=2.0 KAPPA=100.",
         "DISTANCE ATOMS=5,com LABEL=d5",
         "UPPER_WALLS ARG=d5 AT=2.0 KAPPA=100.",
         "DISTANCE ATOMS=6,com LABEL=d6",
         "UPPER_WALLS ARG=d6 AT=2.0 KAPPA=100.",
         "DISTANCE ATOMS=7,com LABEL=d7",
         "UPPER_WALLS ARG=d7 AT=2.0 KAPPA=100.",
         "c1: COORDINATIONNUMBER SPECIES=1-7 MOMENTS=2-3" +
         " SWITCH={RATIONAL R_0=1.5 NN=8 MM=16}",
         "METAD ARG=c1.* HEIGHT=0.05 PACE=50 " +
         "SIGMA=0.1,0.1 GRID_MIN=-1.5,-1.5 GRID_MAX=2.5,2.5" +
         " GRID_BIN=5000,5000 FILE=HILLS"] #usually PACE=500 BIASFACTOR=5

atoms = read('isomer.xyz')
cons = [FixedPlane(i, [0, 0, 1]) for i in range(7)]
atoms.set_constraint(cons)
atoms.set_masses([1, 1, 1, 1, 1, 1, 1])
initial =copy.deepcopy(atoms)

atoms.calc = Plumed(calc=LennardJones(rc=2.5, r0=3.0),
                    input=setup,
                    timestep=timestep,
                    atoms=atoms,
                    )#kT=0.1)

initial.calc = LennardJones(rc=2.5, r0=3.0)
e0=initial.get_potential_energy()
print(atoms.get_potential_energy(), initial.get_potential_energy())
startpos= atoms.get_positions()
print(startpos)
import random


# Define the number of atoms
num_atoms = 7

# Define the coordinate ranges
x_range = (-0.1, 0.1)
y_range = (-0.1, 0.1)
z_value = 0

# Generate 10 different random coordinate combinations
for i in range(10):
    # Generate random coordinates for each atom
    positions = []
    for j in range(num_atoms):
        x = startpos[j][0]-random.uniform(*x_range)
        y = startpos[j][1]-random.uniform(*y_range)
        positions.append([x, y, startpos[j][2]])
    

    # Create an ASE Atoms object and set the positions
    atoms.set_positions(positions)
    initial.set_positions(positions)
    
    # Print the positions
    print(atoms.get_potential_energy()== initial.get_potential_energy())
    print(f'Forces: {atoms.get_forces()==initial.get_forces()}')

traj = read('metad.traj@::')
print(len(traj))

pot = np.zeros(len(traj))
xyz = []

for i,snap in enumerate(traj):
    #print(i)
    #print(snap.calc)
    xyz.append(snap.get_positions())
    
    pot[i] = snap.get_potential_energy()

min_index = np.argmin(pot)
print(f'The min index is {min_index}')
min_value = pot[min_index]
min_pos = xyz[min_index]
atoms.set_positions(min_pos)
initial.set_positions(min_pos)
print(min_pos)
print(atoms.get_potential_energy(), initial.get_potential_energy(),min_value)
print(f'E0: {e0}')