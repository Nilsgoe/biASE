from ase import units
from ase.io import read
from ase.md.langevin import Langevin
from ase.constraints import FixedPlane
from ase.calculators.plumed import Plumed
from ase.calculators.lj import LennardJones


timestep = 0.005
#.5 * units.fs
atoms = read('isomer.xyz')
cons = [FixedPlane(i, [0, 0, 1]) for i in range(7)]
#atoms.set_constraint(cons)
atoms.set_masses([1, 1, 1, 1, 1, 1, 1])
atoms.calc = LennardJones(rc=2.5, r0=3.0)

dyn = Langevin(atoms, timestep=.5 * units.fs, temperature_K=0.1/units.kB, friction=1, trajectory='UnbiasMD.traj',fixcm=False) #
N_steps = int(10000)
dyn.run(N_steps) 