from biASE import General_Bias_Calculator

from ase import units
import numpy as np
from ase.calculators.mixing import LinearCombinationCalculator
import jax.numpy as jnp
from ase.md.md import MolecularDynamics
from ase.parallel import world, DummyMPI
from ase import units
import jax

def smooth_rising_function(x,height=200, center=0, boundary=1.2, steepness=20):
    # Adjusts the function to be approximately 0 within [center - boundary, center + boundary] and rise rapidly outside
    e_value = height*jnp.tanh(steepness * (jnp.abs(x - center) - boundary))
    return e_value[0]


def asymmetric_hookean_well(x, lower_boundary=-1.2, upper_boundary=1.2, k_lower=50, k_upper=50):
    """
    Asymmetric well function using Hookean springs, compatible with JAX.
    
    Parameters:
    - x (float): The value to evaluate.
    - lower_boundary (float): The lower boundary of the well.
    - upper_boundary (float): The upper boundary of the well.
    - k_lower (float): Spring constant for the lower side (x < lower_boundary).
    - k_upper (float): Spring constant for the upper side (x > upper_boundary).
    
    Returns:
    - float: The value of the potential.
    """
    # Below lower boundary: Hookean spring potential
    below_lower = k_lower * jnp.square(lower_boundary - x)
    
    # Above upper boundary: Hookean spring potential
    above_upper = k_upper * jnp.square(x - upper_boundary)
    
    # Inside the boundaries: Zero potential
    inside_well = 0.0
    
    # Use JAX's `where` to define the piecewise function
    potential = jnp.where(x < lower_boundary, below_lower, 
                 jnp.where(x > upper_boundary, above_upper, inside_well))
    return potential[0]


def asymmetric_hookean_well_2d(x, y, 
                               x_lower_boundary=-1.2, x_upper_boundary=1.2,
                               y_lower_boundary=-1.2, y_upper_boundary=1.2,
                               k_x_lower=50, k_x_upper=50,
                               k_y_lower=50, k_y_upper=50):
    """
    Asymmetric well function using Hookean springs, extended to 2D, compatible with JAX.
    
    Parameters:
    - x, y (float): The coordinates to evaluate.
    - x_lower_boundary, x_upper_boundary (float): Boundaries for the x-dimension.
    - y_lower_boundary, y_upper_boundary (float): Boundaries for the y-dimension.
    - k_x_lower, k_x_upper (float): Spring constants for x-dimension.
    - k_y_lower, k_y_upper (float): Spring constants for y-dimension.
    
    Returns:
    - float: The value of the 2D potential.
    """
    # X-dimension Hookean potential
    below_x_lower = k_x_lower * jnp.square(x_lower_boundary - x)
    above_x_upper = k_x_upper * jnp.square(x - x_upper_boundary)
    x_inside_well = 0.0
    x_potential = jnp.where(x < x_lower_boundary, below_x_lower, 
                   jnp.where(x > x_upper_boundary, above_x_upper, x_inside_well))
    
    # Y-dimension Hookean potential
    below_y_lower = k_y_lower * jnp.square(y_lower_boundary - y)
    above_y_upper = k_y_upper * jnp.square(y - y_upper_boundary)
    y_inside_well = 0.0
    y_potential = jnp.where(y < y_lower_boundary, below_y_lower, 
                   jnp.where(y > y_upper_boundary, above_y_upper, y_inside_well))
    
    # Combine the potentials from both dimensions
    total_potential = x_potential + y_potential
    return total_potential

class Metadynamics(MolecularDynamics):
    """Langevin (constant N, V, T) molecular dynamics."""

    # Helps Asap doing the right thing.  Increment when changing stuff:
    _lgv_version = 5

    def __init__(self, atoms, timestep, temperature=None, friction=None,
                 fixcm=True, *, temperature_K=None, trajectory=None,
                 logfile=None, loginterval=1, communicator=world,output_file = None,
                 rng=None, append_trajectory=False,meta_dyn=True,cvs = None,
                 std_dev=0.1,bias_height = 0.001, interval_size = 10):
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
            A friction coefficient in inverse ASE time units.
            For example, set ``0.01 / ase.units.fs`` to provide
            0.01 fs\\ :sup:`−1` (10 ps\\ :sup:`−1`).

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
        self.old_means= None
        #print("Initial Array:",self.old_means, atoms.get_positions().shape)
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
        self.output_file = output_file
        self.timestep = timestep
        
        if self.len_cvs == 1:
            header = f"\t\t\t\t time \t\t CV1 \t\t std1 \t\t height \t\t biasfactor \n"
        elif self.len_cvs == 2:
            header = f"\t\t\t\t time \t\t CV1 \t\t CV2 \t\t std1 \t std2 \t\t height \t\t biasfactor \n"
        else:
            raise Exception("Currently Metadynamics is only avaliable for one or two CVs")
        
        
        if self.output_file != None:
            with open(self.output_file, "w") as file:
                file.write(header)
        
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
         
        
        if self.meta_dyn == True and self.meta_dyn_calc is not None:                # This is a great question! The reason is that when
            #print(self.nsteps, 'update the calc')# The calc gets updated all saved forces and energies get lost!
            atoms.calc =self.meta_dyn_calc       # So updating the calc in the step it self do not make sense
        
        if forces is None:
            forces = atoms.get_forces(md=True)

        # This velocity as well as xi, eta and a few other variables are stored
        # as attributes, so Asap can do its magic when atoms migrate between
        # processors.
        self.v = atoms.get_velocities()

        xi = self.rng.standard_normal(size=(natoms, 3))
        eta = self.rng.standard_normal(size=(natoms, 3))

        # When holonomic constraints for rigid linear triatomic molecules are
        # present, ask the constraints to redistribute xi and eta within each
        # triple defined in the constraints. This is needed to achieve the
        # correct target temperature.
        for constraint in self.atoms.constraints:
            if hasattr(constraint, 'redistribute_forces_md'):
                constraint.redistribute_forces_md(atoms, xi, rand=True)
                constraint.redistribute_forces_md(atoms, eta, rand=True)

        self.communicator.broadcast(xi, 0)
        self.communicator.broadcast(eta, 0)

        # To keep the center of mass stationary, we have to calculate
        # the random perturbations to the positions and the momenta,
        # and make sure that they sum to zero.
        self.rnd_pos = self.c5 * eta
        self.rnd_vel = self.c3 * xi - self.c4 * eta
        if self.fix_com:
            self.rnd_pos -= self.rnd_pos.sum(axis=0) / natoms
            self.rnd_vel -= (self.rnd_vel *
                             self.masses).sum(axis=0) / (self.masses * natoms)

        # First halfstep in the velocity.
        self.v += (self.c1 * forces / self.masses - self.c2 * self.v +
                   self.rnd_vel)

        # Full step in positions
        x = atoms.get_positions()

        # Step: x^n -> x^(n+1) - this applies constraints if any.
        atoms.set_positions(x + self.dt * self.v + self.rnd_pos)

        # recalc velocities after RATTLE constraints are applied
        self.v = (self.atoms.get_positions() - x - self.rnd_pos) / self.dt
        #s3=time.time()
        forces = atoms.get_forces(md=True)

        # Update the velocities
        self.v += (self.c1 * forces / self.masses - self.c2 * self.v +
                   self.rnd_vel)
        #e2 = time.time()
        # Second part of RATTLE taken care of here
        atoms.set_momenta(self.v * self.masses)
        #start = time.time()
        ### We update here the potenial if wanted!
        if self.nsteps % self.interval_size ==0:
            print(self.nsteps,self.timestep,self.nsteps*self.timestep)
            if self.meta_dyn == True:
                self.current_cvs = self.cvs(atoms.get_positions())
                if self.old_means is None:
                    self.old_means = jnp.array([self.current_cvs]) # save only cv instead of position!!!
                else:
                    self.old_means = jnp.vstack([self.old_means, self.current_cvs]) # save only cv instead of position!!!

                if self.nsteps<self.interval_size:
                    self.bias_potential = self.bias_setup(self.cvs)
                else:
                    #print(self.nsteps, " Update Bias")
                    self.bias_update()
                    
                #print(self.old_means)
                self.meta_dyn_calc = LinearCombinationCalculator([self.main_potential,self.bias_potential],[1.,1.])
                if self.output_file != None:
                        with open(self.output_file, "+a") as file:
                            if self.len_cvs ==1:
                                file.write(f'\t\t\t\t {self.nsteps*self.timestep/(units.fs):>7.2f} \t\t {self.current_cvs[0]:.8f} \t\t {self.std_dev} \t\t {(self.bias_factor/(self.bias_factor-1))*self.bias_height*self.regulation_factor[-1][0]:.8f} \t\t {self.bias_factor}\n')
                            else:
                                file.write(f'\t\t\t\t {self.nsteps*self.timestep/(units.fs):>7.2f} \t\t {self.current_cvs[0]:.8f} \t\t {self.current_cvs[1]:.8f} \t\t {self.std_dev} \t\t {self.std_dev} \t\t {(self.bias_factor/(self.bias_factor-1))*self.bias_height*self.regulation_factor[-1][0]:.8f} \t\t {self.bias_factor}\n')
                            file.close

        else:
            pass
        return forces

    def bias_setup(self,cvs):
        """Here the bias along the cv will be build
           ATM only one cvs with a one dimensional gaussian will be supported
        
        """
        if self.len_cvs ==1:
            self.gaussian_bias = General_Bias_Calculator(cvs,self.gaussian_1D,use_jit=True)
        elif self.len_cvs ==2:
            self.gaussian_bias = General_Bias_Calculator(cvs,self.gaussian_2D,use_jit=True)
        else:
            raise TypeError("CVs need to be inform of an array with one or two entrys! e.g. jnp.array([cv1,cv2]).")
            
        return self.gaussian_bias
    
    def bias_update(self):
        if self.len_cvs ==1 :
            self.bias_potential.update_function(self.gaussian_1D,use_jit=True)
        if self.len_cvs ==2 :
            self.bias_potential.update_function(self.gaussian_2D,use_jit=True)

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
        mean_cvs = self.old_means.flatten()
        std_dev = self.std_dev
        jax.debug.print("mean of the gaussian_2D: {}", mean_cvs) 
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
        mean_cvs= self.old_means
        std_dev = self.std_dev
        jax.debug.print("mean of the gaussian_2D: {}", mean_cvs) 
        #print(mean_cvs[:, 0])
        # Evaluate the Gaussian for all means at once using numpy broadcasting
        gaussian_vals = jnp.exp(- 0.5*((x - mean_cvs[:, 0])**2 / std_dev +
                                       (y - mean_cvs[:, 1])**2 / std_dev))

        # Sum over all means and return the result
        #print(jnp.sum(gaussian_vals, axis=0)[0])
        return self.bias_height*jnp.sum(gaussian_vals,axis=0)#[0] 


    def return_bias(self):
        return self.bias_potential
    
    def return_biased_potential(self):
        return self.meta_dyn_calc



"""Langevin dynamics class.""" #copyed from the v3.22.1 not the master!


import time
class WT_Metadynamics(MolecularDynamics):
    """Well_tempered Metadynmaics with Langevin (constant N, V, T) molecular dynamics."""

    # Helps Asap doing the right thing.  Increment when changing stuff:
    _lgv_version = 5

    def __init__(self, atoms, timestep, temperature=None, friction=None,
                 fixcm=True, *, temperature_K=None, trajectory=None,
                 logfile=None, loginterval=1, communicator=world,
                 rng=None, append_trajectory=False,meta_dyn=True,cvs=None,
                 std_dev=[0.1,], bias_height=0.001, interval_size= 10, bias_factor=5,
                 output_file = None,well_temp=True,stretch=False,input_file=None,max_bias=10000):
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
            A friction coefficient in inverse ASE time units.
            For example, set ``0.01 / ase.units.fs`` to provide
            0.01 fs\\ :sup:`−1` (10 ps\\ :sup:`−1`).

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
        self.old_means= None
        #print("Initial Array:",self.old_means, atoms.get_positions().shape)
        self.meta_dyn = meta_dyn
        self.stretch = stretch
        
        self.cutoff= 6.25 #Taken from PLUMED 
        self.stretchA=1.00193418799744762399 # Taken from PLUMED: 1.0/(1-exp(-self.cutoff));
        self.stretchB=-.00193418799744762399 # Taken from PLUMED: -exp(-self.cutoff)/(1-exp(-self.cutoff))
        
        if meta_dyn ==True and cvs==None:
             raise TypeError("Please provide cvs  if you want to use the Metadynamics option (meta_dyn)!")
        else:
            self.cvs = cvs
            self.len_cvs = len(list(self.cvs(atoms.get_positions())))
            
        #also supports a single std_dev
        if not isinstance(std_dev, list):
            if self.len_cvs==1:
                self.std_dev = [std_dev]
            elif self.len_cvs==2:
                self.std_dev = [std_dev,std_dev]
        else:
            self.std_dev = std_dev
            
        self.main_potential = atoms.get_calculator()
        self.bias_height= bias_height
        self.interval_size = interval_size
        self.bias_factor = bias_factor
        self.output_file = output_file
        self.timestep = timestep
        self.well_temp=well_temp
        self.input_file = input_file
        self.max_bias = max_bias
        if temperature_K != None:
            self.init_temp = temperature_K*units.kB
        elif temperature != None:
            self.init_temp = temperature
            
        if self.input_file == None:
            if self.len_cvs == 1:
                header = f"#!\t Height={self.bias_height}, Pace={self.interval_size}, Step_size={self.timestep/units.fs}, CV_Number=1, Bias_factor={self.bias_factor}, WT={self.well_temp}, Stretch={self.stretch} \n\t\t\t\t time \t\t\t CV1 \t\t\t\t std1 \t\t\t height \t\t biasfactor \t regularizationfactor\n"
            elif self.len_cvs == 2:
                header = f"!#\t Height={self.bias_height}, Pace={self.interval_size}, Step_size={self.timestep/units.fs}, CV_Number=2, Bias_factor={self.bias_factor}, WT={self.well_temp}, Stretch={self.stretch} \n\t\t\t\t time \t\t\t CV1 \t\t\t\t CV2 \t\t\t\t std1 \t\t\t std2 \t\t\t height \t\t biasfactor \tregularizationfactor\n"
            else:
                raise Exception("Currently Metadynamics is only avaliable for one or two CVs")
            if self.output_file != None:
                with open(self.output_file, "w") as file:
                    file.write(header)
        else:
            print("Warning: The output will be saved in the input file!")
            self.output_file=self.input_file
            if self.len_cvs!=1 and self.len_cvs!=2:
                raise Exception("Currently Metadynamics is only avaliable for one or two CVs")
            
        
        
        
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
        s=time.time()             
        atoms = self.atoms
        natoms = len(atoms)
        print("Steps: ", self.nsteps)
        #print(self.nsteps)
        if self.nsteps==0 and self.input_file!=None:
            print("Reading input file")
            data=[]
            with open(self.input_file, "r") as file:
                #Skip Setup and Header
                next(file)
                next(file)
                for line in file:
                    # Skip empty lines or lines that don't contain relevant data
                    if line.strip() and not line.startswith("#"):
                        # Split the line into columns based on whitespace
                        columns = line.split()
                        # Convert values to floats and append to the list
                        data.append([float(value) for value in columns])

            # Convert the list of lists into a JAX array
            data_jax = jnp.array(data)
            if self.len_cvs == 1:
                print("1D")
                self.times, self.cv1s, self.std1, self.heights, self.biasfactor,self.regulation_factor_ = data_jax.T
                self.old_means_ = self.cv1s.reshape(-1,1)
                self.std_dev=[self.std1[0]]
            elif self.len_cvs == 2:
                print("2D")
                self.times, self.cv1s, self.cv2s, self.std1, self.std2, self.heights, self.biasfactor,self.regulation_factor_ = data_jax.T
                self.std_dev=[self.std1[0],self.std2[0]]
                self.old_means_ = np.column_stack((self.cv1s, self.cv2s))
            else:
                raise Exception("Currently Metadynamics is only implemented for one or two CVs")        
            self.nsteps_input = len(self.times)*self.interval_size
            self.bias_factor=self.biasfactor[0]
            if self.well_temp==True:
                print("Well-Tempered Metadynamics")
                self.regulation_factor_=self.regulation_factor_.reshape(-1,1)
                print("Reg ", self.regulation_factor_)
            else:
                raise Exception("This needs well_temp=True")
        elif self.input_file==None:
            self.nsteps_input=0
            
        if self.nsteps!=0: # one could ask why is the update of the calc here?
            if self.meta_dyn == True:                # This is a great question! The reason is that when
                #print(self.nsteps, 'update the calc')# The calc gets updated all saved forces and energies get lost!
                atoms.calc =self.meta_dyn_calc  #So updating the calc in the step it self do not make sense
                        
        if forces is None:
            forces = atoms.get_forces(md=True)
            #print('Forces: ',forces,"E ",atoms.get_total_energy(),atoms.get_calculator())
            #e1=time.time()
        #s2=time.time()    
        # This velocity as well as rnd_pos, rnd_mom and a few other
        # variables are stored as attributes, so Asap can do its magic
        # when atoms migrate between processors.
        self.v = atoms.get_velocities()
        xi = self.rng.standard_normal(size=(natoms, 3))
        eta = self.rng.standard_normal(size=(natoms, 3))

        # When holonomic constraints for rigid linear triatomic molecules are
        # present, ask the constraints to redistribute xi and eta within each
        # triple defined in the constraints. This is needed to achieve the
        # correct target temperature.
        for constraint in self.atoms.constraints:
            if hasattr(constraint, 'redistribute_forces_md'):
                constraint.redistribute_forces_md(atoms, xi, rand=True)
                constraint.redistribute_forces_md(atoms, eta, rand=True)

        self.communicator.broadcast(xi, 0)
        self.communicator.broadcast(eta, 0)

        # To keep the center of mass stationary, we have to calculate
        # the random perturbations to the positions and the momenta,
        # and make sure that they sum to zero.
        self.rnd_pos = self.c5 * eta
        self.rnd_vel = self.c3 * xi - self.c4 * eta
        if self.fix_com:
            self.rnd_pos -= self.rnd_pos.sum(axis=0) / natoms
            self.rnd_vel -= (self.rnd_vel *
                             self.masses).sum(axis=0) / (self.masses * natoms)
        # First halfstep in the velocity.
        #print("V1:",self.v)
        self.v += (self.c1 * forces / self.masses - self.c2 * self.v +
                   self.rnd_vel)
        #print("V2:",self.v)
        # Full step in positions
        x = atoms.get_positions()
        #print("X: ", x,"DT: ",self.dt,"V ", self.v,"RND_pos ", self.rnd_pos)
        # Step: x^n -> x^(n+1) - this applies constraints if any.
        atoms.set_positions(x + self.dt * self.v + self.rnd_pos)
        #print("This should be: ",x + self.dt * self.v + self.rnd_pos)
        #print("NEW pos: ",atoms.get_positions())
        # recalc velocities after RATTLE constraints are applied
        self.v = (self.atoms.get_positions() - x - self.rnd_pos) / self.dt
        #print("NEW V:",self.v)
        #s3=time.time()
        forces = atoms.get_forces(md=True)
        #print("F:",forces,atoms.calc)
        # Update the velocities
        self.v += (self.c1 * forces / self.masses - self.c2 * self.v +
                   self.rnd_vel)
        #print("NEW V:",self.v)
        #e2 = time.time()
        # Second part of RATTLE taken care of here
        atoms.set_momenta(self.v * self.masses)
        start = time.time()
        ### We update here the potenial if wanted!
        if self.nsteps % self.interval_size ==0:
            self.nsteps_total= self.nsteps + self.nsteps_input #Here it should be checked if self.nsteps gets reseted
            print(self.nsteps_total,self.timestep,self.nsteps_total*self.timestep)
            if self.meta_dyn == True:
                self.current_cvs = self.cvs(atoms.get_positions())
                # Initialize with 100 rows of zeros (or any desired size)
                if self.nsteps==0:
                    self.regulation_factor = np.zeros((self.max_bias, 1))  # Assuming single column entries
                    self.old_means = np.zeros((self.max_bias, len(self.current_cvs)))  # Match current_cvs size
                if self.input_file!=None:
                    n_to_update = len(self.regulation_factor_)  # Number of new entries to update
                    indices_to_update = np.arange(n_to_update)  # Sequentially update rows 0 to n_to_update-1

                    # Update the arrays
                    self.regulation_factor[indices_to_update] = self.regulation_factor_
                    self.old_means[indices_to_update, :] = self.old_means_
                
                if self.old_means is None:
                    #self.old_means = jnp.array([self.current_cvs]) # save only cv instead of position!!!
                    self.previous_portential = 0
                    #self.regulation_factor=np.asarray([[np.exp(-self.previous_portential/(self.well_temp*self.bias_factor))]]) #insted pf self.temp regulation_factor=jnp.exp(-self.previous_portential/(self.temp*self.bias_factor)) # kB is already included here but what do to if given in K?                    
                    #print(self.previous_portential)
                else:
                    #print(self.temp,self.well_temp)
                    if self.len_cvs ==1:
                        self.previous_portential = self.gaussian_1D(self.current_cvs,self.old_means.flatten(),self.regulation_factor.flatten())# was current_cvs = self.cvs(atoms.get_positions())
                        #print(self.previous_portential)
                    if self.len_cvs ==2:    
                        self.previous_portential = self.gaussian_2D(self.current_cvs,self.old_means,self.regulation_factor.flatten()) # was current_cvs = self.cvs(atoms.get_positions())
                    '''print("Reg ", np.exp(-self.previous_portential/(self.well_temp*self.bias_factor)))
                    self.regulation_factor=np.vstack([self.regulation_factor, np.exp(-self.previous_portential/(self.well_temp*self.bias_factor))] ) #instead of self.temp
                    print("Reg_f ",self.regulation_factor)
                    #print(self.previous_portential,self.regulation_factor[-1])
                    #print("Old Means: ",self.old_means, " New Mean: ", self.cvs(atoms.get_positions()))
                    self.old_means = np.vstack([self.old_means, self.current_cvs]) # save only cv instead of position!!! # was current_cvs = self.cvs(atoms.get_positions())
                    #print(self.old_means)'''

                    
                    print("Reg_f:", self.regulation_factor.shape)#[:int(self.nsteps_total/self.interval_size)+1])
                    
                    # Update specific entries
                    self.regulation_factor[int(self.nsteps_total/self.interval_size)] = np.exp(-self.previous_portential/(self.init_temp*(self.bias_factor-1)))
                    self.old_means[int(self.nsteps_total/self.interval_size), :] = self.current_cvs

                    # Ensure arrays remain the same size
                    #print("Reg_f:", self.regulation_factor[:int(self.nsteps_total/self.interval_size)+1])
                    #print("Old_means:", self.old_means[:int(self.nsteps_total/self.interval_size)+1])
                    #print("Old_Pot:",self.previous_portential)
                #print(self.regulation_factor)
                if self.nsteps<self.interval_size:
                    self.bias_potential = self.bias_setup(self.cvs)
                 
                else:
                    if self.nsteps_total % (self.interval_size*100) ==0:  
                        print("Clear JIT Cache",)
                        self.bias_update(clear_jit_cache=True)
                    else:
                        self.bias_update()
                
                #print(self.position_array)
                if self.nsteps<self.interval_size:
                    self.meta_dyn_calc = LinearCombinationCalculator([self.main_potential,self.bias_potential],[1.,1.])
                #end = time.time()
                #print(self.nsteps, 'set up the calc')
                #print(end-start)
                #print(self.nsteps)
                if self.output_file != None:
                        with open(self.output_file, "+a") as file:
                            if self.len_cvs ==1:
                                file.write(f'\t\t\t\t {self.nsteps_total*self.timestep/(units.fs):>7.2f} \t\t {self.current_cvs[0]:.8f} \t\t {self.std_dev[0]:.4f} \t\t {(self.bias_factor/(self.bias_factor-1))*self.bias_height*self.regulation_factor[int(self.nsteps_total/self.interval_size)][0]:.8f} \t\t {self.bias_factor} \t {self.regulation_factor[int(self.nsteps_total/self.interval_size)][0]}\n') #maybe self.regulation_factor[int(self.nsteps/self.interval_size)+1][0]
                            else:
                                file.write(f'\t\t\t\t {self.nsteps_total*self.timestep/(units.fs):>7.2f} \t\t {self.current_cvs[0]:.8f} \t\t {self.current_cvs[1]:.8f} \t\t {self.std_dev[0]:.4f} \t\t {self.std_dev[1]:.4f} \t\t {(self.bias_factor/(self.bias_factor-1))*self.bias_height*self.regulation_factor[int(self.nsteps_total/self.interval_size)][0]:.8f} \t\t {self.bias_factor}\t {self.regulation_factor[int(self.nsteps_total/self.interval_size)][0]}\n')#maybe self.regulation_factor[int(self.nsteps/self.interval_size)+1][0]
                            file.close
                    #\t{self.current_cvs[1]}\t{self.std_dev}
        else:
            pass
        e=time.time()
        #with open("time.txt", "+a") as file:
        #    file.write(f'{self.nsteps}\t{e-s}\n')
        return forces

    def bias_setup(self,cvs):
        """Here the bias along the cv will be build
           ATM only one cvs with a one dimensional gaussian will be supported
        
        """
        if self.len_cvs ==1:
            print("Means:",self.old_means.flatten())
            self.gaussian_bias = General_Bias_Calculator(cvs,self.gaussian_1D,static_arg1=self.old_means.flatten(),static_arg2=self.regulation_factor.flatten(),use_jit=True)
        elif self.len_cvs ==2:
            self.gaussian_bias = General_Bias_Calculator(cvs,self.gaussian_2D,static_arg1=self.old_means,static_arg2=self.regulation_factor.flatten(),use_jit=True)
        else:
            raise TypeError("CVs need to be inform of an array with one or two entrys! e.g. jnp.array([cv1,cv2]).")
            
        return self.gaussian_bias
    
    def bias_update(self,clear_jit_cache=False):
        print(self.nsteps, " Update Bias")

        if self.len_cvs ==1 :
            self.bias_potential.update_function(self.gaussian_1D,self.old_means.flatten(),self.regulation_factor.flatten(),clear_jit_cache=clear_jit_cache)#self.gaussian_1D,use_jit=True,clear_jit_cache=clear_jit_cache)
        if self.len_cvs ==2 :
            self.bias_potential.update_function(self.gaussian_2D,static_arg1=self.old_means,static_arg2=self.regulation_factor.flatten(),clear_jit_cache=clear_jit_cache)
    
    def gaussian_1D(self,cvs,mean_cvs,reg_f): ### here with jax numpy
        """ 
        x (float or numpy array): Input value(s) for evaluating the Gaussian.
            mean (float or numpy array): Mean(s) of the Gaussian(s).
            std_dev (float): Standard deviation of the Gaussian(s).

        Returns:
            float or numpy array: Value(s) of the Gaussian(s) at the given input(s), summed over all means.
            
            
            this function needs to be tested! seems to be the same as the above numpy version but forces need to be tested
        """ 
        x = cvs[0]
        #mean_cvs = self.old_means.flatten()
        std_dev = self.std_dev[0]
        #print(mean_cvs[:20])
        # Evaluate the Gaussian for all means at once using numpy broadcasting
        gaussian_vals = (jnp.exp(-0.5*((x - mean_cvs[:, jnp.newaxis])**2/(std_dev**2)))).flatten()
        #print("Gauss: ",gaussian_vals)
        #reg_f =self.regulation_factor.flatten()
        #print("reg in Bias",reg_f)
        #print('regulation_factor: ', self.previous_portential,self.nsteps)
        # Sum over all means and return the result
        #print(jnp.sum(gaussian_vals, axis=0)[0])
        #print('Previous Potential: ',  self.bias_height*self.regulation_factor*jnp.sum(gaussian_vals, axis=0)[0],self.nsteps)
        '''print('gauss: ',gaussian_vals)
        print('reg: ' ,reg_f)
        print('gauss times reg',gaussian_vals*reg_f)
        print('sum of gauss times reg ',jnp.sum(gaussian_vals*reg_f, axis=0))'''
        if self.stretch == True:
            #print(x - mean_cvs)
            streched_gaussian_vals=self.stretchA*gaussian_vals+self.stretchB
            streched_gaussian_vals=jnp.maximum(streched_gaussian_vals, 0)
            return self.bias_height*jnp.sum(reg_f*(streched_gaussian_vals),axis=0)
        return self.bias_height*jnp.sum(gaussian_vals*reg_f, axis=0) # why [0] check that more - is necessay bc out put would be a 1D list with one entrence
    
        
    

    
    def gaussian_2D(self,cvs,mean_cvs,reg_f):
        """
        Evaluate a 2D Gaussian potential energy surface for the given input collective variable(s) (cvs).

        Args:
            cvs (float or numpy array): Input value(s) for evaluating the 2D Gaussian.

        Returns:
            float or numpy array: Value(s) of the 2D Gaussian at the given input(s), summed over all means.
        """
        x, y = cvs
        #mean_cvs= self.old_means
        std_dev = self.std_dev
        
        #print(mean_cvs[:, 0])
        # Evaluate the Gaussian for all means at once using numpy broadcasting
        gaussian_vals = jnp.exp(-0.5*((x - mean_cvs[:, 0])**2/(std_dev[0]**2)) \
                                -0.5*((y - mean_cvs[:, 1])**2/(std_dev[1]**2)))
        #regulation_factor=jnp.exp(-self.previous_portential/(self.temp*self.bias_factor)) #the kB here is used to transform from K to eV!
        #print('regulation_factor: ', self.previous_portential,self.nsteps)
        #print(mean_cvs,cvs)
        # Sum over all means and return the result
        #print(jnp.sum(gaussian_vals, axis=0)[0])
        
        '''print('Potential_NOW: ', self.bias_height*regulation_factor*jnp.sum(gaussian_vals,axis=0),self.nsteps)
        print('Sum_Gaussian:', jnp.sum(gaussian_vals,axis=0), 'Previous Potential: ', self.previous_portential,self.nsteps)
        print('Reg_factor: ',regulation_factor, 'T: ', self.temp)'''
        #print('HI')
        #print(self.regulation_factor)
        #reg_f =self.regulation_factor.flatten()
        ''' print('gauss: ',gaussian_vals)
        print('reg: ' ,reg_f)
        print('gauss times reg',gaussian_vals*reg_f)
        print('sum of gauss times reg ',jnp.sum(gaussian_vals*reg_f, axis=0))'''
        if self.stretch ==True:
            streched_gaussian_vals=self.stretchA*gaussian_vals+self.stretchB
            streched_gaussian_vals=jnp.maximum(streched_gaussian_vals, 0)
            #print(f'Stretch: {streched_gaussian_vals}')
            return self.bias_height*jnp.sum(reg_f*(streched_gaussian_vals),axis=0)
        
        return self.bias_height*jnp.sum(gaussian_vals*reg_f,axis=0)

        
    def return_bias(self):
        return self.bias_potential
    
    def return_biased_potential(self):
        return self.meta_dyn_calc
    
    def return_func(self):
        return self.old_means, self.regulation_factor
