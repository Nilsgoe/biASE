import importlib.util
import unittest

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


JAX_AVAILABLE = importlib.util.find_spec('jax') is not None

if JAX_AVAILABLE:
    import jax.numpy as jnp

    from Umbrella_sampling import UmbrellaMD


class ZeroCalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.results['energy'] = 0.0
        self.results['forces'] = np.zeros((len(atoms), 3))


class DummyMD:
    def __init__(self):
        self.calls = []

    def run(self, n_steps):
        self.calls.append(n_steps)


@unittest.skipUnless(JAX_AVAILABLE, 'JAX is required for umbrella sampling tests')
class TestUmbrellaSampling(unittest.TestCase):
    def test_setup_and_run(self):
        def cv_function(positions):
            return jnp.array([positions[0, 0]])

        atoms = Atoms('H', positions=[[1.0, 0.0, 0.0]])
        atoms.calc = ZeroCalculator()
        md = DummyMD()

        umbrella = UmbrellaMD(
            atoms=atoms,
            cv_function=cv_function,
            x0=jnp.array([0.25]),
            k=4.0,
            md_method=md,
        )

        expected_energy = 0.5 * 4.0 * (1.0 - 0.25) ** 2
        np.testing.assert_allclose(atoms.get_potential_energy(), expected_energy, rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(atoms.get_forces(), [[-3.0, 0.0, 0.0]], rtol=1e-7, atol=1e-10)

        umbrella.run(12)
        self.assertEqual(md.calls, [12])

        umbrella.setup_new_window(x0=jnp.array([0.5]), k=2.0)
        expected_updated_energy = 0.5 * 2.0 * (1.0 - 0.5) ** 2
        np.testing.assert_allclose(
            atoms.get_potential_energy(),
            expected_updated_energy,
            rtol=1e-7,
            atol=1e-10,
        )


if __name__ == '__main__':
    unittest.main()
