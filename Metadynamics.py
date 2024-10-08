from biASE import General_Bias_Calculator

from ase import units
import numpy as np
from ase.calculators.mixing import LinearCombinationCalculator
import jax.numpy as jnp
from ase.md.md import MolecularDynamics
from ase.parallel import world, DummyMPI
from ase import units


class Metadynamics(MolecularDynamics):
    """Langevin (constant N, V, T) molecular dynamics."""

    # Helps Asap doing the right thing.  Increment when changing stuff:
    _lgv_version = 4

    def __init__(self, atoms, timestep, temperature=None, friction=None,
                 fixcm=True, *, temperature_K=None, trajectory=None,
                 logfile=None, loginterval=1, communicator=world,
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
        self.meta_dyn_calc = None
        
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
        #start = time.time()
        ### We update here the potenial if wanted!
        if (self.nsteps+1) % self.interval_size ==0:
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
                #end = time.time()
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
        gaussian_vals = jnp.exp(-0.5*((x - mean_cvs[:, jnp.newaxis])**2/(std_dev**2))) 
        
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
        gaussian_vals = jnp.exp(-0.5*((x - mean_cvs[:, 0])**2/(std_dev**2)) \
                                                      - 0.5*((y - mean_cvs[:, 1])**2/(std_dev**2)))

        # Sum over all means and return the result
        #print(jnp.sum(gaussian_vals, axis=0)[0])
        return self.bias_height*jnp.sum(gaussian_vals,axis=0)#[0] 


    def return_bias(self):
        return self.bias_potential
    
    def return_biased_potential(self):
        return self.meta_dyn_calc



"""Langevin dynamics class.""" #copyed from the v3.22.1 not the master!



class WT_Metadynamics(MolecularDynamics):
    """Well_tempered Metadynmaics with Langevin (constant N, V, T) molecular dynamics."""

    # Helps Asap doing the right thing.  Increment when changing stuff:
    _lgv_version = 4

    def __init__(self, atoms, timestep, temperature=None, friction=None,
                 fixcm=True, *, temperature_K=None, trajectory=None,
                 logfile=None, loginterval=1, communicator=world,
                 rng=None, append_trajectory=False,meta_dyn=True,cvs=None,
                 std_dev=0.1, bias_height=0.001, interval_size= 10, bias_factor=5,
                 output_file = None,well_temp,stretch=False):
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
        self.stretch = stretch
        
        self.cutoff= 6.25 #Taken from PLUMED 
        self.stretchA=1.00193418799744762399 # Taken from PLUMED: 1.0/(1-exp(-self.cutoff));
        self.stretchB=-.00193418799744762399 # Taken from PLUMED: -exp(-self.cutoff)/(1-exp(-self.cutoff))
        
        if meta_dyn ==True and cvs==None:
             raise TypeError("Please provide cvs  if you want to use the Metadynamics option (meta_dyn)!")
        else:
            self.cvs = cvs
            self.len_cvs = len(list(self.cvs(atoms.get_positions())))
        self.std_dev =std_dev
        self.main_potential = atoms.get_calculator()
        self.bias_height= bias_height
        self.interval_size = interval_size
        self.bias_factor = bias_factor
        self.output_file = output_file
        self.timestep = timestep
        self.well_temp=well_temp
        self.meta_dyn_calc=None

        header = f"\t \t\t\t \t time \t \t CV1 \t \t CV2 \t \t \t std1   \t std2 \t height \t biasfactor \n"
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
         # one could ask why is the update of the calc here?
        if self.meta_dyn == True and self.meta_dyn_calc is not None:                # This is a great question! The reason is that when
            #print(self.nsteps, 'update the calc')# The calc gets updated all saved forces and energies get lost!
            self.atoms.calc =self.meta_dyn_calc  #So updating the calc in the step it self do not make sense

                
        atoms = self.atoms
        natoms = len(atoms)
         
        #print(atoms.get_potential_energy(),self.atoms.get_potential_energy())
        
        if forces is None:
            forces = atoms.get_forces(md=True)
            #if (self.nsteps-1) % self.interval_size ==0:
                

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
        #start = time.time()
        ### We update here the potenial if wanted!
        if (self.nsteps+1) % self.interval_size ==0:
            if self.meta_dyn == True:
                self.current_cvs = self.cvs(atoms.get_positions())
                if self.position_array is None:
                    self.position_array = np.array([self.cvs(atoms.get_positions())]) # save only cv instead of position!!!
                    self.previous_portential = 0
                    self.regulation_factor=np.asarray([[np.exp(-self.previous_portential/(self.well_temp*self.bias_factor))]]) #insted pf self.temp regulation_factor=jnp.exp(-self.previous_portential/(self.temp*self.bias_factor)) # kB is already included here but what do to if given in K? 
                    print(self.previous_portential)
                else:
                    #print(self.temp,self.well_temp)
                    if self.len_cvs ==1:
                        self.previous_portential = self.gaussian_1D(self.current_cvs)# was current_cvs = self.cvs(atoms.get_positions())
                        #print(self.previous_portential)
                    if self.len_cvs ==2:    
                        self.previous_portential = self.gaussian_2D(self.current_cvs) # was current_cvs = self.cvs(atoms.get_positions())
                    self.regulation_factor=np.vstack([self.regulation_factor, np.exp(-self.previous_portential/(self.well_temp*self.bias_factor))] ) #instead of self.temp
                    #print(self.previous_portential,self.regulation_factor[-1])
                    #print("Old Means: ",self.position_array, " New Mean: ", self.cvs(atoms.get_positions()))
                    self.position_array = np.vstack([self.position_array, self.current_cvs]) # save only cv instead of position!!! # was current_cvs = self.cvs(atoms.get_positions())
                    #print(self.position_array)

                #print(self.regulation_factor)
                self.bias_potential = self.bias_setup(atoms, self.position_array,self.cvs)
                
                #print(self.position_array)
                self.meta_dyn_calc = LinearCombinationCalculator([self.main_potential,self.bias_potential],[1.,1.])
                #end = time.time()
                #print(self.nsteps, 'set up the calc')
                #print(end-start)
                #print(self.nsteps)
                if self.output_file != None:
                        with open(self.output_file, "+a") as file:
                            if self.len_cvs ==1:
                                file.write(f'\t\t\t\t \t{self.nsteps*self.timestep}\t\t{self.current_cvs[0]}\t\t{self.std_dev}\t\t{(self.bias_factor/(self.bias_factor-1))*self.bias_height*self.regulation_factor[-1][0]}\t\t{self.bias_factor}\n')
                            else:
                                file.write(f'\t\t\t\t \t{self.nsteps*self.timestep}\t\t{self.current_cvs[0]}\t\t{self.current_cvs[1]}\t\t{self.std_dev}\t\t{self.std_dev}\t\t{(self.bias_factor/(self.bias_factor-1))*self.bias_height*self.regulation_factor[-1][0]}\t\t{self.bias_factor}\n')
                            file.close
                    #\t{self.current_cvs[1]}\t{self.std_dev}
                
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
        gaussian_vals = (jnp.exp(-0.5*((x - mean_cvs[:, jnp.newaxis])**2/(std_dev**2)))).flatten()
        reg_f =self.regulation_factor.flatten()

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
        gaussian_vals = jnp.exp(-0.5*((x - mean_cvs[:, 0])**2/(std_dev**2)) \
                                                        - 0.5*((y - mean_cvs[:, 1])**2/(std_dev**2)))
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
        reg_f =self.regulation_factor.flatten()
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
        return self.position_array, self.regulation_factor