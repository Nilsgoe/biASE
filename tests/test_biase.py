import importlib.util
import unittest

import numpy as np
from ase import Atoms
from ase.calculators.fd import calculate_numerical_forces, calculate_numerical_stress


JAX_AVAILABLE = importlib.util.find_spec('jax') is not None

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp

    jax.config.update('jax_enable_x64', True)

    from biASE import General_Bias_Calculator


@unittest.skipUnless(JAX_AVAILABLE, 'JAX is required for biASE tests')
class TestGeneralBiasCalculator(unittest.TestCase):
    def test_forces_and_stress_match_numerical_derivatives(self):
        def cv_function(positions):
            return jnp.array([positions[0, 0]])

        def bias_function(cvs, x0, k):
            dx = cvs - x0
            return 0.5 * k * jnp.sum(dx**2)

        atoms = Atoms(
            'H',
            positions=[[1.1, 0.2, 0.3]],
            cell=np.diag([4.0, 5.0, 6.0]),
            pbc=True,
        )
        atoms.calc = General_Bias_Calculator(
            cv_function=cv_function,
            bias_function=bias_function,
            static_arg1=jnp.array([0.4]),
            static_arg2=3.5,
            use_jit=True,
        )

        analytical_forces = atoms.get_forces()
        numerical_forces = calculate_numerical_forces(atoms, eps=1e-6)
        np.testing.assert_allclose(analytical_forces, numerical_forces, rtol=5e-6, atol=5e-8)

        analytical_stress = atoms.get_stress()
        numerical_stress = calculate_numerical_stress(atoms, eps=1e-6)
        np.testing.assert_allclose(analytical_stress, numerical_stress, rtol=5e-6, atol=5e-8)


if __name__ == '__main__':
    unittest.main()
