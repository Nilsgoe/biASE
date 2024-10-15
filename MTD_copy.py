from ase import units
from ase.io import read
from ase.md.langevin import Langevin
from ase.constraints import FixedPlane
from ase.calculators.plumed import Plumed
from ase.calculators.lj import LennardJones

timestep = 0.005
ps = 1000 * units.fs

setup = [f"UNITS LENGTH=A TIME={1/ps} ENERGY={units.mol/units.kJ}",
         "c1: COORDINATIONNUMBER SPECIES=1-7 MOMENTS=2-3" +
         " SWITCH={RATIONAL R_0=1.5 NN=8 MM=16}",
         "METAD ARG=c1.* HEIGHT=0.05 PACE=50 " +
         "SIGMA=0.1,0.1 GRID_MIN=-3.5,-3.5 GRID_MAX=3.5,3.5" +
         " GRID_BIN=5000,5000 FILE=HILLS"] #usually PACE=500 BIASFACTOR=5

atoms = read('isomer.xyz')
cons = [FixedPlane(i, [0, 0, 1]) for i in range(7)]
atoms.set_constraint(cons)
atoms.set_masses([1, 1, 1, 1, 1, 1, 1])

atoms.calc = Plumed(calc=LennardJones(rc=2.5, r0=3.0),
                    input=setup,
                    timestep=timestep,
                    atoms=atoms,
                    )#kT=0.1)
                    
dyn = Langevin(atoms, timestep, temperature_K=0.1/units.kB, friction=1,
               fixcm=False, trajectory='MTD_wo_wt.traj')

dyn.run(int(1e5))#usually 500000
