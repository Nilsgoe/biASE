import matplotlib.pyplot as plt
from biASE import General_Bias_Calculator

from ase.atoms import Atoms
from ase import units
from ase.io import read


from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution,Stationary, ZeroRotation
from ase.constraints import FixedPlane



from ase.calculators.calculator import Calculator, all_changes
from ase.io import Trajectory

from ase.calculators.mixing import LinearCombinationCalculator,MixedCalculator
import time
import jax.numpy as jnp
from jax import jit
from jax import grad ,jacfwd,jacrev

import numpy as np

from ase.md.md import MolecularDynamics
from ase.parallel import world, DummyMPI
from ase import units
from ase.io import read
from ase.constraints import FixedPlane
from ase.calculators.lj import LennardJones

class WT_Metadynamics(MolecularDynamics):
    """Langevin (constant N, V, T) molecular dynamics."""

    # Helps Asap doing the right thing.  Increment when changing stuff:
    _lgv_version = 4

    def __init__(self, atoms, timestep, temperature=None, friction=None,
                 fixcm=True, *, temperature_K=None, trajectory=None,
                 logfile=None, loginterval=1, communicator=world,
                 rng=None, append_trajectory=False,meta_dyn=True,cvs = None,std_dev=0.1,bias_height = 0.001, interval_size = 10):
        """
        Parameters:

        atoms: Atoms object
            The list of atoms.

        timestep: float
            The time step in ASE time units.

        temperature: float (deprecated)
            The desired temperature, in electron volt.

        temperature_K: float
            The desired temperature, in Kelvin.

        friction: float
            A friction coefficient, typically 1e-4 to 1e-2.

        fixcm: bool (optional)
            If True, the position and momentum of the center of mass is
            kept unperturbed.  Default: True.

        rng: RNG object (optional)
            Random number generator, by default numpy.random.  Must have a
            standard_normal method matching the signature of
            numpy.random.standard_normal.

        logfile: file object or str (optional)
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        trajectory: Trajectory object or str (optional)
            Attach trajectory object.  If *trajectory* is a string a
            Trajectory will be constructed.  Use *None* (the default) for no
            trajectory.

        communicator: MPI communicator (optional)
            Communicator used to distribute random numbers to all tasks.
            Default: ase.parallel.world. Set to None to disable communication.

        append_trajectory: bool (optional)
            Defaults to False, which causes the trajectory file to be
            overwritten each time the dynamics is restarted from scratch.
            If True, the new structures are appended to the trajectory
            file instead.

        The temperature and friction are normally scalars, but in principle one
        quantity per atom could be specified by giving an array.

        RATTLE constraints can be used with these propagators, see:
        E. V.-Eijnden, and G. Ciccotti, Chem. Phys. Lett. 429, 310 (2006)

        The propagator is Equation 23 (Eq. 39 if RATTLE constraints are used)
        of the above reference.  That reference also contains another
        propagator in Eq. 21/34; but that propagator is not quasi-symplectic
        and gives a systematic offset in the temperature at large time steps.
        """
        if friction is None:
            raise TypeError("Missing 'friction' argument.")
        self.fr = friction
        self.temp = units.kB * self._process_temperature(temperature,
                                                         temperature_K, 'eV')
        self.fix_com = fixcm
        if communicator is None:
            communicator = DummyMPI()
        self.communicator = communicator
        if rng is None:
            self.rng = np.random
        else:
            self.rng = rng
        MolecularDynamics.__init__(self, atoms, timestep, trajectory,
                                   logfile, loginterval,
                                   append_trajectory=append_trajectory)
        self.updatevars()
        
        ### Added from med
        self.position_array= None
        print("Initial Array:",self.position_array, atoms.get_positions().shape)
        self.meta_dyn = meta_dyn
        if meta_dyn ==True and cvs==None:
             raise TypeError("Please provide cvs  if you want to use the Metadynamics option (meta_dyn)!")
        else:
            self.cvs = cvs
            self.len_cvs = len(list(self.cvs(atoms.get_positions())))
        self.std_dev =std_dev
        self.main_potential = atoms.get_calculator()
        self.bias_height= bias_height
        self.interval_size = interval_size
        
    def todict(self):
        d = MolecularDynamics.todict(self)
        d.update({'temperature_K': self.temp / units.kB,
                  'friction': self.fr,
                  'fixcm': self.fix_com})
        return d

    def set_temperature(self, temperature=None, temperature_K=None):
        self.temp = units.kB * self._process_temperature(temperature,
                                                         temperature_K, 'eV')
        self.updatevars()

    def set_friction(self, friction):
        self.fr = friction
        self.updatevars()

    def set_timestep(self, timestep):
        self.dt = timestep
        self.updatevars()

    def updatevars(self):
        dt = self.dt
        T = self.temp
        fr = self.fr
        masses = self.masses
        sigma = np.sqrt(2 * T * fr / masses)

        self.c1 = dt / 2. - dt * dt * fr / 8.
        self.c2 = dt * fr / 2 - dt * dt * fr * fr / 8.
        self.c3 = np.sqrt(dt) * sigma / 2. - dt**1.5 * fr * sigma / 8.
        self.c5 = dt**1.5 * sigma / (2 * np.sqrt(3))
        self.c4 = fr / 2. * self.c5

    def step(self, forces=None):
        atoms = self.atoms
        natoms = len(atoms)
         
        if (self.nsteps-1) % self.interval_size ==0: # one could ask why is the update of the calc here?
            if self.meta_dyn == True:                # This is a great question! The reason is that when
                #print(self.nsteps, 'update the calc')# The calc gets updated all saved forces and energies get lost!
                atoms.calc =self.meta_dyn_calc       # So updating the calc in the step it self do not make sense
        
        if forces is None:
            forces = atoms.get_forces(md=True)

        # This velocity as well as xi, eta and a few other variables are stored
        # as attributes, so Asap can do its magic when atoms migrate between
        # processors.
        self.v = atoms.get_velocities()

        self.xi = self.rng.standard_normal(size=(natoms, 3))
        self.eta = self.rng.standard_normal(size=(natoms, 3))

        # When holonomic constraints for rigid linear triatomic molecules are
        # present, ask the constraints to redistribute xi and eta within each
        # triple defined in the constraints. This is needed to achieve the
        # correct target temperature.
        for constraint in self.atoms.constraints:
            if hasattr(constraint, 'redistribute_forces_md'):
                constraint.redistribute_forces_md(atoms, self.xi, rand=True)
                constraint.redistribute_forces_md(atoms, self.eta, rand=True)

        self.communicator.broadcast(self.xi, 0)
        self.communicator.broadcast(self.eta, 0)

        # First halfstep in the velocity.
        self.v += (self.c1 * forces / self.masses - self.c2 * self.v +
                   self.c3 * self.xi - self.c4 * self.eta)

        # Full step in positions
        x = atoms.get_positions()
        if self.fix_com:
            old_com = atoms.get_center_of_mass()
        # Step: x^n -> x^(n+1) - this applies constraints if any.
        atoms.set_positions(x + self.dt * self.v + self.c5 * self.eta)
        if self.fix_com:
            atoms.set_center_of_mass(old_com)

        # recalc velocities after RATTLE constraints are applied
        self.v = (self.atoms.get_positions() - x -
                  self.c5 * self.eta) / self.dt
        forces = atoms.get_forces(md=True)

        # Update the velocities
        self.v += (self.c1 * forces / self.masses - self.c2 * self.v +
                   self.c3 * self.xi - self.c4 * self.eta)

        if self.fix_com:  # subtract center of mass vel
            self.v -= self._get_com_velocity(self.v)

        # Second part of RATTLE taken care of here
        atoms.set_momenta(self.v * self.masses)
        start = time.time()
        ### We update here the potenial if wanted!
        if self.nsteps % self.interval_size ==0:
            if self.meta_dyn == True:
                if self.position_array is None:
                    self.position_array = np.array([self.cvs(atoms.get_positions())]) # save only cv instead of position!!!
                    print(self.position_array)
                else:
                    #print("Old Means: ",self.position_array, " New Mean: ", self.cvs(atoms.get_positions()))
                    self.position_array = np.vstack([self.position_array, self.cvs(atoms.get_positions())]) # save only cv instead of position!!!
                    #print(self.position_array)


                self.bias_potential = self.bias_setup(atoms, self.position_array,self.cvs)
                #print(self.position_array)
                self.meta_dyn_calc = LinearCombinationCalculator([self.main_potential,self.bias_potential],[1.,1.])
                end = time.time()
                #print(self.nsteps, 'set up the calc')
                #print(end-start)
                #print(self.nsteps)
        return forces

    def bias_setup(self,atoms, bias_positions,cvs):
        """Here the bias along the cv will be build
           ATM only one cvs with a one dimensional gaussian will be supported
        
        """
        if self.len_cvs ==1:
            self.gauissian_bias = General_Bias_Calculator(cvs,self.gaussian_1D,use_jit=True)
        elif self.len_cvs ==2:
            self.gauissian_bias = General_Bias_Calculator(cvs,self.gaussian_2D,use_jit=True)
        else:
            raise TypeError("CVs need to be inform of an array with one or two entrys! e.g. jnp.array([cv1,cv2]).")
            
        return self.gauissian_bias
    

    def gaussian_1D(self,cvs): ### here with jax numpy
        """ 
        x (float or numpy array): Input value(s) for evaluating the Gaussian.
            mean (float or numpy array): Mean(s) of the Gaussian(s).
            std_dev (float): Standard deviation of the Gaussian(s).

        Returns:
            float or numpy array: Value(s) of the Gaussian(s) at the given input(s), summed over all means.
            
            
            this function needs to be tested! seems to be the same as the above numpy version but forces need to be tested
        """ 
        x = cvs[0]
        mean_cvs = self.position_array.flatten()
        std_dev = self.std_dev

        # Evaluate the Gaussian for all means at once using numpy broadcasting
        gaussian_vals = jnp.exp(-0.5*((x - mean_cvs[:, jnp.newaxis])**2/(0.5*std_dev**2))) 
        
        # Sum over all means and return the result
        #print(jnp.sum(gaussian_vals, axis=0)[0])
        return self.bias_height*jnp.sum(gaussian_vals, axis=0)[0] # why [0] check that more - is necessay bc out put would be a 1D list with one entrence
    
    
    def gaussian_2D(self,cvs):
        """
        Evaluate a 2D Gaussian potential energy surface for the given input collective variable(s) (cvs).

        Args:
            cvs (float or numpy array): Input value(s) for evaluating the 2D Gaussian.

        Returns:
            float or numpy array: Value(s) of the 2D Gaussian at the given input(s), summed over all means.
        """
        x, y = cvs
        mean_cvs= self.position_array
        std_dev = self.std_dev
        
        #print(mean_cvs[:, 0])
        # Evaluate the Gaussian for all means at once using numpy broadcasting
        gaussian_vals = jnp.exp(-0.5*((x - mean_cvs[:, 0])**2/(0.5*std_dev**2)) \
                                                      - 0.5*((y - mean_cvs[:, 1])**2/(0.5*std_dev**2)))

        # Sum over all means and return the result
        #print(jnp.sum(gaussian_vals, axis=0)[0])
        return self.bias_height*jnp.sum(gaussian_vals,axis=0)#[0] 


    def return_bias(self):
        return self.bias_potential
    
    def return_biased_potential(self):
        return self.meta_dyn_calc



def fill_diagonal(matrix, value):
    num_rows, num_cols = matrix.shape
    diagonal_indices = jnp.arange(min(num_rows, num_cols))
    filled_matrix = jax.ops.index_update(matrix, (diagonal_indices, diagonal_indices), value)
    return filled_matrix

def _cvs_moments(atoms_positions, cutoff):
    """ Calculate coordination numbers for all atoms in an ASE Atoms object.
        And then calculate the second and third central moments of the coordination number distribution using numpy.
    """
    dist_matrix = jnp.sqrt(np.sum((atoms_positions[:, jnp.newaxis] - atoms_positions) ** 2, axis=-1))
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

def plot_2d_bias(trajectory=None,pic_name = '2D_plot.png'):
    plt.plot(trajectory[0],trajectory[1])
    plt.xlabel('scm')
    plt.ylabel('tcm')
    #plt.savefig(pic_name)
    plt.show()


timestep = 0.005
#.5 * units.fs
atoms = read('isomer.xyz')
print(_cvs_moments(atoms.get_positions(), cutoff=1.5))
print(atoms.get_velocities())
cons = [FixedPlane(i, [0, 0, 1]) for i in range(7)]
#atoms.set_constraint(cons)
atoms.set_masses([1, 1, 1, 1, 1, 1, 1])
atoms.calc = LennardJones(rc=2.5, r0=3.0)
'''calc1= LennardJones(rc=2.5, r0=3.0)
calc2 = LennardJones(rc=2.5, r0=3.5)
atoms.calc = LinearCombinationCalculator([calc1,calc2],[0.5,0.5])'''
from functools import partial
dyn = WT_Metadynamics(atoms, timestep=timestep, temperature_K=0.1/units.kB, friction=1, trajectory='metad.traj',fixcm=False,cvs=partial(_cvs_moments, cutoff=1.5),interval_size=500,std_dev=.1,bias_height=.05) #Langevin(atoms, timestep=.5 * units.fs, temperature_K=0.1/units.kB, friction=1, trajectory='metad.traj',fixcm=False) #
N_steps = int(5e4)
dyn.run(N_steps) 


traj = read('metad.traj@::')[:1000]
print(len(traj))


scm = np.zeros(N_steps+1)
tcm = np.zeros(N_steps+1)

# Define an array to store the temperature at each step
temperatures = np.zeros(N_steps+1)
# Define an array to store the energy at each step
energy = np.zeros(N_steps+1)
kin = np.zeros(N_steps+1)
pot = np.zeros(N_steps+1)

for i,snap in enumerate(traj):
    #print(i)
    #print(snap.calc)
    xyz = snap.get_positions()
    scm[i],tcm[i] =_cvs_moments(xyz,cutoff=1.5)
    # Record the temperature at this step
    temperatures[i] = snap.get_temperature()
    pot[i] = snap.get_potential_energy()
    kin[i] = snap.get_kinetic_energy()
    energy[i] =snap.get_total_energy()
    
plot_2d_bias(trajectory=[scm,tcm])
plt.xlabel('Iterations')
plt.ylabel('x')
plt.plot(scm)
plt.plot(tcm)
#plt.savefig('X_movement_DW_C.png')

# Plot the temperature as a function of time
timestep = .5*units.fs
time_ = np.arange(N_steps+1) * timestep / units.fs
plt.plot(time_, temperatures)#/ units.kB)
plt.xlabel('Time (fs)')
plt.ylabel('Temperature (K)')
plt.savefig('NVT_DW_temp.png')
plt.show()
print(np.average(np.array(temperatures)))


plt.xlabel('Iterations')
plt.ylabel('Energy')
#plt.plot(kin)
plt.plot(pot)
#plt.plot(energy)