from biASE import General_Bias_Calculator
import jax.numpy as jnp
from ase import units
from ase.calculators.mixing import SumCalculator

class UmbrellaMD:
    def __init__(self,
        cv_function,
        x0,
        k,
        md_method,
        atoms
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
        self.len_cvs = len(list(self.cv_function(atoms.get_positions())))
        self.k = k
        self.x0 = x0
        self.positions = atoms.get_positions()
        self.md_method = md_method
        try:
            self.orignial_calc=atoms.get_calculator()
        except:
            raise ValueError("Please set a calculator for the atoms object")
        
        self.setup_umbrella_potential()
        
    def setup_umbrella_potential(self):
        if self.len_cvs == 1:
            umbrella_potential = General_Bias_Calculator(
                cv_function=self.cv_function,
                bias_function=self.harmonic_bias,
                x0=self.x0,
                k=self.k,
            )
        elif self.len_cvs == 2:
            umbrella_potential = General_Bias_Calculator(
                cv_function=self.cv_function,
                bias_function=self.harmonic_bias_2D,
                x0=self.x0,
                k=self.k,
            )
        else:
            raise ValueError("Only 1D and 2D CVs are supported")
        self.full_potential = SumCalculator([self.orignial_calc, umbrella_potential])
        self.atoms.set_calculator(self.full_potential)
        
    def setup_new_window(self,x0,k):
        self.k = k
        self.x0 = x0
        self.setup_umbrella_potential()
        
             
    def harmonic_bias_2D(self,x):
        dx = x - self.x0
        return 0.5 * self.k * jnp.sum(dx**2)


    def harmonic_bias(self,x):
        dx = x - self.x0
        return 0.5 * self.k * dx**2
    
    def run(self, N_steps):
        self.md_method.run(N_steps)
               


# Example usage
'''if __name__ == "__main__":
    from ase import Atoms
    from ase.md.verlet import VelocityVerlet

    # Define a simple collective variable (e.g., distance between two atoms)
    def distance_cv(system):
        return np.linalg.norm(system.get_positions()[0] - system.get_positions()[1])

    # Create a system and set up umbrella sampling
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]])
    integrator = VelocityVerlet(atoms, timestep=1.0)

    umbrella = UmbrellaSampling(
        system=atoms,
        collective_variable=distance_cv,
        k=10.0,
        target_value=1.0,
        temperature=300.0,
    )

    umbrella.sample(n_steps=1000, integrator=integrator, output_interval=100)
    print("Bias energies:", umbrella.get_bias_energies())'''
