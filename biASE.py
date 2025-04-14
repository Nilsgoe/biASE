import numpy as np
from ase.calculators.calculator import Calculator, all_changes
import jax.numpy as jnp
from jax import jit
from jax import grad ,jacfwd,jacrev
import warnings
import jax

class Direct_General_Bias_Calculator(Calculator):
    """ This a class of general bias calculator it can calculate energy, force. For doing so a bias function and a function that defines the variables for the bais function
        are necessary. To obtain the force it is necessary to calculate the derivatives of the given bias function. This done by using the auto gradient function 'grad()'
        from JAX. To obtain the derivative(s) of the collective variables the Jacobian matrix of the these is generated with 'jacfwd()'.
        
        
        Parameters
        ----------                 
        bias_function:  python function, necessary
                        This the function the Calculator is based on! It should only have a single input, the set of collective variable in form of an 1D array!
                        
        use_jit:        boolean, optional
                        Allows just in time compilation of all used functions and thier derivatives
    """
    implemented_properties=['energy','forces']#,'hessian']
    def __init__(self, bias_function,*args, use_jit = True, **kwargs):
        """ Sets up the Calculator from ase for example self.results or allows get_forces (and similar things) to work"""

        Calculator.__init__(self, **kwargs)
        #print("init calc")
        self.use_jit = use_jit
        self.args = args
        if self.use_jit:
            self.bias_function = jit(bias_function)
            self.deriv_bias_function = jit(grad(self.bias_function))
            #self.hessian_bias_funtion = jit(self.calc_hessian(self.bias_function))

        else:
            self.bias_function = bias_function
            self.deriv_bias_function = grad(self.bias_function)
            
            #self.hessian_bias_funtion = self.calc_hessian(self.bias_function)

    def update_function(self, bias_function,*args,use_jit=True):
        self.use_jit = use_jit
        self.args = args
        if self.use_jit:
            self.bias_function = jit(bias_function)
            self.deriv_bias_function = jit(grad(self.bias_function,argnums=0))
            #self.hessian_bias_funtion = jit(self.calc_hessian(self.bias_function))

        else:
            self.bias_function = bias_function
            self.deriv_bias_function = grad(self.bias_function,argnums=0)
        
    def calculate(self, atoms=None, properties=['energy','forces'], system_changes=all_changes):
        """ Calculates all the implemented properties. This method needs to call the .calculate form the Calculator class"""

        Calculator.calculate(self, atoms, properties, system_changes) # for setting up all necessary stuff DO NOT define ANY stuff by hand this method defines! otherwise it won't recoginze changes
       
        self.energy = 0
        
        atoms_pos = atoms.get_positions()

        self.energy += np.array(self.bias_function(atoms_pos,*self.args))

        if 'forces' in properties: 
            self.forces = self.calc_forces(atoms_pos)
            self.results['forces'] = np.asarray(self.forces)
            if np.any(np.isnan(self.results['forces'])):
                raise ValueError("Error: There are nan values in the forces.")
            
        if 'stress' in properties:
            self.stress = +0
            pass

        if 'hessian' in properties:
            self.hessian = self.hessian_bias_funtion(atoms_pos).reshape(len(atoms)*3,len(atoms)*3)

        self.results['energy'] = self.energy
    
        
    
    def calc_forces(self,atoms_pos):
        """ This method calculates the force based on the given function for the initializing the Calculator. Force obtainting the force ther derivative of the energy with respect to 
            the collective varriables and the derivative of the collective variables with respect to the cartesian coordinates of the atom postions are necessary.
            
            transfering to np arrays is super important other wise the constraints are not working and the code will crash
        """
        ### Sets up an force array of zeros with the right dimensions! 
        force_array = self.set_force_array(atoms_pos)

        ### Calcs derivativecalc_deriv_Eof the Energy with respect to the collective variables
        self.deriv_E = self.calc_deriv_E(atoms_pos)
    
        force_array= -self.deriv_E
        return force_array
    
    
    def set_force_array(self,atoms_pos):
        return np.zeros(atoms_pos.shape)
    
    
    def calc_deriv_E(self,atoms_pos):
        ''' Calcs derivative of the Energy with respect to the positions'''
        return np.asarray(self.deriv_bias_function(atoms_pos,*self.args)) # dimensions (collective variables)
    
    def calc_Jac_E(self,atoms_pos):
        ''' Calcs derivative of the Energy with respect to the positions'''
        if self.use_jit:
            self.jac_bias_function = jit(jacfwd(self.bias_function))
        else:
            self.jac_bias_function = jacfwd(self.bias_function)
        return np.asarray(self.jac_bias_function(atoms_pos,*self.args))

    def calc_hessian(self,f):
        ''' Gives the hessian of the energy with respect to the CVs'''
        return jacfwd(jacrev(f))


    def get_hessian(self):
        ''' There is no get_hessian definde in the ASE calculator class'''
        return self.results['hessian']
    
    

class General_Bias_Calculator(Calculator):
    """ This a class of general bias calculator it can calculate energy, force. For doing so a bias function and a function that defines the variables for the bais function
        are necessary. To obtain the force it is necessary to calculate the derivatives of the given bias function. This done by using the auto gradient function 'grad()'
        from JAX. To obtain the derivative(s) of the collective variables the Jacobian matrix of the these is generated with 'jacfwd()'.
        
        
        Parameters
        ----------
        cv_function:    python function (will be optional later)
                        This function should only dependend on the atoms postions of the atom object later used  
                        If all cartesian coordiantes should be used just flatten the atoms postions array as the cv_function (should be tested for systems of different size)
                      
        bias_function:  python function, necessary
                        This the function the Calculator is based on! It should only have a single input, the set of collective variable in form of an 1D array!
                        
        use_jit:        boolean, optional
                        Allows just in time compilation of all used functions and thier derivatives
    """
    implemented_properties=['energy','forces']#,'hessian']
    def __init__(self, cv_function, bias_function,static_arg1=None, static_arg2=None, use_jit = True, **kwargs):
        """ Sets up the Calculator from ase for example self.results or allows get_forces (and similar things) to work"""
        Calculator.__init__(self, **kwargs)

        self.use_jit = use_jit
        self.static_arg1=static_arg1
        self.static_arg2=static_arg2
        print("S:",self.static_arg1, self.static_arg2)
        if self.use_jit:
            self.bias_function = jit(bias_function)
            self.deriv_bias_function = jit(grad(self.bias_function))
            
            self.cv_function = jit(cv_function)
            self.deriv_cv = jit(jacfwd(cv_function))
            self.cv_function_ = cv_function
            #self.hessian_bias_funtion = jit(self.calc_hessian(self.bias_function))

        else:
            self.bias_function = bias_function
            self.deriv_bias_function = grad(self.bias_function)
            
            self.cv_function = cv_function
            self.deriv_cv = jacfwd(cv_function)
            
            #self.hessian_bias_funtion = self.calc_hessian(self.bias_function)

    
    def update_function(self,bias_function,static_arg1,static_arg2,clear_jit_cache=False):#, bias_function,use_jit=True,clear_jit_cache=False):
        
        #self.use_jit = use_jit
        self.static_arg1=static_arg1
        self.static_arg2=static_arg2
        if clear_jit_cache==True:
            print("Clearing")
            if self.use_jit:
                if clear_jit_cache == True:
                    jax.clear_caches()
                    self.cv_function = jit(self.cv_function)
                    self.deriv_cv = jit(jacfwd(self.cv_function))
                self.bias_function = jit(bias_function)
                self.deriv_bias_function = jit(grad(self.bias_function))
                #self.hessian_bias_funtion = jit(self.calc_hessian(self.bias_function))

            else:
                self.bias_function = bias_function
                self.deriv_bias_function = grad(self.bias_function)
        

    def calculate(self, atoms=None, properties=['energy','forces'], system_changes=all_changes):
        """ Calculates all the implemented properties. This method needs to call the .calculate form the Calculator class"""

        Calculator.calculate(self, atoms, properties, system_changes) # for setting up all necessary stuff DO NOT define ANY stuff by hand this method defines! otherwise it won't recoginze changes
       
        self.energy = 0
        
        atoms_pos = atoms.get_positions()
        #print("in calc pos:", atoms_pos)
        self.cvs = self.cv_function(atoms_pos)
        if self.static_arg1 is None and self.static_arg2 is None:
            self.energy += np.array(self.bias_function(self.cvs))
        elif self.static_arg2 is None:
            self.energy += np.array(self.bias_function(self.cvs,self.static_arg1))
        else :
            self.energy += np.array(self.bias_function(self.cvs,self.static_arg1,self.static_arg2))
        #print("cv: ",self.cvs, "E: ",self.energy)

        if 'forces' in properties: 
            self.forces = self.calc_forces(atoms_pos,self.cvs)
            
            self.results['forces'] = np.asarray(self.forces)
            #print("Force;",self.forces)
            if np.any(np.isnan(self.results['forces'])):
                raise ValueError("Error: There are nan values in the forces.")
        if 'stress' in properties:
            self.stress = +0
            pass

        if 'hessian' in properties:
            self.hessian = self.hessian_bias_funtion(atoms_pos).reshape(len(atoms)*3,len(atoms)*3)

        self.results['energy'] = self.energy
        
        
    
    def calc_forces(self,atoms_pos,cvs):
        """ This method calculates the force based on the given function for the initializing the Calculator. Force obtainting the force ther derivative of the energy with respect to 
            the collective varriables and the derivative of the collective variables with respect to the cartesian coordinates of the atom postions are necessary.
            
            transfering to np arrays is super important other wise the constraints are not working and the code will crash
        """
        ### Sets up an force array of zeros with the right dimensions! 
        force_array = self.set_force_array(atoms_pos)
        
        ### Calcs derivative of the Energy with respect to the collective variables
        self.deriv_E = self.calc_deriv_E_to_cvs(cvs)
        
        ### Calcs derivative of the collective varriable with respect to the cartesian coordiantes of the atom postions
        self.deriv_cvs = self.calc_deriv_cvs(atoms_pos)
        #print("dedcv:",self.deriv_E)
        #print("dcvdx:", self.deriv_cvs)
        force_array -=np.dot(self.deriv_E,self.deriv_cvs) #dimensions (number of atom(s), coordinates of the atom)
        return force_array
    
    
    def set_force_array(self,atoms_pos):
        return np.zeros(atoms_pos.shape)
    
    
    def calc_deriv_E_to_cvs(self,cvs):
        ''' Calcs derivative of the Energy with respect to the collective variable'''
        if self.static_arg1 is None and self.static_arg2 is None:

            return np.asarray(self.deriv_bias_function(cvs)) # dimensions (collective variables)
        elif self.static_arg2 is None:
            return np.asarray(self.deriv_bias_function(cvs,self.static_arg1))
        else:
            return np.asarray(self.deriv_bias_function(cvs,self.static_arg1,self.static_arg2)) # dimensions (collective variables)
    
    
    def calc_deriv_cvs(self,atoms_pos):
        '''Calcs derivative of the collective varriable with respect to the cartesian coordiantes of the atom postions'''
        jacobian = np.asarray(self.deriv_cv(atoms_pos))
                              
        if jacobian.ndim == 3:
            return np.moveaxis(jacobian,[0],[1]) #Dimensons (number of atom(s),collective variables, coodrinates of the atom)
        elif jacobian.ndim == 2:
            return jacobian
        else:
            return 'There is an Error in the dimensions of the CV derivative'
    
    def calc_hessian(self,f):
        ''' Gives the hessian of the energy with respect to the CVs'''
        return jacfwd(jacrev(f))


    def get_hessian(self):
        ''' There is no get_hessian definde in the ASE calculator class'''
        return self.results['hessian']
    
    

