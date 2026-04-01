from biASE import General_Bias_Calculator
import jax.numpy as jnp
from ase import units
from ase.calculators.mixing import SumCalculator

class UmbrellaMD:
    def __init__(self,
        atoms,
        cv_function,
        x0,
        k,
        md_method,
    ):
        """
        Initialize the umbrella sampling.

        Parameters:
            atoms: A simulation system object (e.g., ASE Atoms).
            collective_variable: Function to compute the CV from the system.
            k (float): Force constant for the harmonic bias (in eV/A^2).
            x0 (float): Target value of the CV for this umbrella window.
            temperature (float): Temperature of the simulation (in Kelvin).
        """
        self.atoms = atoms
        self.cv_function = cv_function
        self.len_cvs =jnp.size(self.cv_function(atoms.get_positions()))
        self.k = k
        self.x0 = x0
        self.positions = atoms.get_positions()
        self.md_method = md_method
        try:
            self.orignial_calc=atoms.calc
        except:
            raise ValueError("Please set a calculator for the atoms object")
        
        self.setup_umbrella_potential()

    def optimize_in_window(self,optimizer,fmax,max_steps,**kwargs):
        optimizer(atoms=self.atoms,**kwargs).run(fmax=fmax,steps=max_steps)  
            
    def setup_umbrella_potential(self):
        if self.len_cvs == 1:
            umbrella_potential = General_Bias_Calculator(
                cv_function=self.cv_function,
                bias_function=self.harmonic_bias,
                static_arg1=self.x0,
                static_arg2=self.k,
            )
        elif self.len_cvs == 2:
            umbrella_potential = General_Bias_Calculator(
                cv_function=self.cv_function,
                bias_function=self.harmonic_bias_2D,
                static_arg1=self.x0,
                static_arg2=self.k,
            )
        else:
            raise ValueError("Only 1D and 2D CVs are supported")
        self.full_potential = SumCalculator([self.orignial_calc, umbrella_potential])
        self.atoms.calc =self.full_potential
        
    def setup_new_window(self,x0,k):
        self.k = k
        self.x0 = x0
        self.setup_umbrella_potential()
        
             
    def harmonic_bias_2D(self,x,x0=None,k=None):
        center = self.x0 if x0 is None else x0
        force_constant = self.k if k is None else k
        dx = x - center
        return 0.5 * force_constant * jnp.sum(dx**2)


    def harmonic_bias(self,x,x0=None,k=None):
        center = self.x0 if x0 is None else x0
        force_constant = self.k if k is None else k
        dx = x - center
        return 0.5 * force_constant * jnp.sum(dx**2)
    
    def run(self, N_steps):
        self.md_method.run(N_steps)
               


# Example usage
if __name__ == "__main__":
    from ase.io import read
    from mace.calculators import mace_off
    import jax.numpy as jnp
    from ase.optimize import BFGS, FIRE
    from ase.md.langevin import Langevin
    from ase import units



    def compute_cv(positions):
        # Indices: 2 for O, 3 for C5, 1 for C2, and 5 for C3
        d_O4H8 = jnp.linalg.norm(positions[4] - positions[8]) 
        d_O3H8 = jnp.linalg.norm(positions[3] - positions[8]) 
        return d_O4H8 -d_O3H8

    mace_calc_dis = mace_off(model="large",dispersion=True)
    atoms1 = read("optimized_malonaldehyde_initial.xyz")
    atoms = atoms1.copy()
    atoms.calc = mace_calc_dis
    UMD = UmbrellaMD(atoms=atoms,cv_function=compute_cv,x0=compute_cv(atoms1.positions)+.5,k=50,md_method=Langevin(atoms, timestep=.5 * units.fs, temperature_K=293, friction=0.1, trajectory='umd.traj',fixcm=False))

    UMD.optimize_in_window(optimizer=BFGS,fmax=0.05,max_steps=500,logfile="BFGS.log")
    UMD.run(200)
    
