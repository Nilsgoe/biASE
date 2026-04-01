import inspect

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jacfwd, jacrev, jit

from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)


class Direct_General_Bias_Calculator(Calculator):
    """Bias calculator for energies defined directly from positions."""

    implemented_properties = ['energy', 'free_energy', 'forces']

    def __init__(self, bias_function, *args, use_jit=True, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.use_jit = use_jit
        self.args = args

        if self.use_jit:
            self.bias_function = jit(bias_function)
            self.deriv_bias_function = jit(grad(self.bias_function))
        else:
            self.bias_function = bias_function
            self.deriv_bias_function = grad(self.bias_function)

    def update_function(self, bias_function, *args, use_jit=True):
        self.use_jit = use_jit
        self.args = args
        if self.use_jit:
            self.bias_function = jit(bias_function)
            self.deriv_bias_function = jit(grad(self.bias_function, argnums=0))
        else:
            self.bias_function = bias_function
            self.deriv_bias_function = grad(self.bias_function, argnums=0)

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.energy = 0.0
        atoms_pos = atoms.get_positions()
        self.energy += float(np.asarray(self.bias_function(atoms_pos, *self.args)))

        if 'forces' in properties:
            self.forces = self.calc_forces(atoms_pos)
            self.results['forces'] = np.asarray(self.forces)
            if np.any(np.isnan(self.results['forces'])):
                raise ValueError('Error: There are nan values in the forces.')

        if 'hessian' in properties:
            self.hessian = self.hessian_bias_funtion(atoms_pos).reshape(
                len(atoms) * 3, len(atoms) * 3
            )

        self.results['energy'] = self.energy
        self.results['free_energy'] = self.energy

    def calc_forces(self, atoms_pos):
        return -self.calc_deriv_E(atoms_pos)

    def calc_deriv_E(self, atoms_pos):
        return np.asarray(self.deriv_bias_function(atoms_pos, *self.args))

    def calc_Jac_E(self, atoms_pos):
        if self.use_jit:
            self.jac_bias_function = jit(jacfwd(self.bias_function))
        else:
            self.jac_bias_function = jacfwd(self.bias_function)
        return np.asarray(self.jac_bias_function(atoms_pos, *self.args))

    def calc_hessian(self, f):
        return jacfwd(jacrev(f))

    def get_hessian(self):
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

    implemented_properties = ['energy', 'free_energy', 'forces', 'stress']

    def __init__(
        self,
        cv_function,
        bias_function,
        static_arg1=None,
        static_arg2=None,
        use_jit=True,
        **kwargs,
    ):
        if static_arg1 is None and 'x0' in kwargs:
            static_arg1 = kwargs.pop('x0')
        if static_arg2 is None and 'k' in kwargs:
            static_arg2 = kwargs.pop('k')

        Calculator.__init__(self, **kwargs)

        self.use_jit = use_jit
        self.static_arg1 = static_arg1
        self.static_arg2 = static_arg2
        self._cv_call_with_context = self._build_cv_wrapper(cv_function)
        self._stress_gradient = None

        if self.use_jit:
            self.bias_function = jit(bias_function)
            self.deriv_bias_function = jit(grad(self.bias_function))
            self.cv_function = jit(self._cv_call_with_context)
            self.deriv_cv = jit(jacfwd(self.cv_function, argnums=0))
        else:
            self.bias_function = bias_function
            self.deriv_bias_function = grad(self.bias_function)
            self.cv_function = self._cv_call_with_context
            self.deriv_cv = jacfwd(self.cv_function, argnums=0)

    def _build_cv_wrapper(self, cv_function):
        try:
            signature = inspect.signature(cv_function)
        except (TypeError, ValueError):
            return lambda positions, cell, pbc: cv_function(positions)

        parameters = list(signature.parameters.values())
        named_parameters = signature.parameters
        cell_name = None
        pbc_name = None
        for candidate in ('cell', 'cell_input'):
            if candidate in named_parameters:
                cell_name = candidate
                break
        for candidate in ('pbc', 'pbc_input'):
            if candidate in named_parameters:
                pbc_name = candidate
                break

        def wrapped_cv_function(positions, cell, pbc):
            kwargs = {}
            extra_args = []
            if cell_name is not None:
                kwargs[cell_name] = cell
            elif len(parameters) >= 2:
                extra_args.append(cell)
            if pbc_name is not None:
                kwargs[pbc_name] = pbc
            elif len(parameters) >= 3:
                extra_args.append(pbc)
            return cv_function(positions, *extra_args, **kwargs)

        return wrapped_cv_function

    def update_function(self, bias_function, static_arg1, static_arg2, clear_jit_cache=False):
        self.static_arg1 = static_arg1
        self.static_arg2 = static_arg2

        if clear_jit_cache and self.use_jit:
            jax.clear_caches()
            self.cv_function = jit(self._cv_call_with_context)
            self.deriv_cv = jit(jacfwd(self.cv_function, argnums=0))

        if self.use_jit:
            self.bias_function = jit(bias_function)
            self.deriv_bias_function = jit(grad(self.bias_function))
        else:
            self.bias_function = bias_function
            self.deriv_bias_function = grad(self.bias_function)

        self._stress_gradient = None

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        atoms_pos = atoms.get_positions()
        cell = np.asarray(atoms.cell)
        pbc = atoms.get_pbc()

        self.cvs = self.cv_function(
            jnp.asarray(atoms_pos),
            jnp.asarray(cell),
            jnp.asarray(pbc, dtype=bool),
        )
        self.energy = self.calc_energy_from_cvs(self.cvs)

        if 'forces' in properties:
            self.forces = self.calc_forces(atoms_pos, self.cvs, cell=cell, pbc=pbc)
            self.results['forces'] = np.asarray(self.forces)
            if np.any(np.isnan(self.results['forces'])):
                raise ValueError('Error: There are nan values in the forces.')

        if 'stress' in properties:
            self.stress = self.calc_stress(
                atoms_pos=atoms_pos,
                cell=cell,
                pbc=pbc,
                volume=atoms.get_volume(),
                cell_rank=atoms.cell.rank,
            )
            self.results['stress'] = np.asarray(self.stress)

        if 'hessian' in properties:
            self.hessian = self.hessian_bias_funtion(atoms_pos).reshape(
                len(atoms) * 3, len(atoms) * 3
            )

        self.results['energy'] = self.energy
        self.results['free_energy'] = self.energy

    def calc_energy_from_cvs(self, cvs):
        if self.static_arg1 is None and self.static_arg2 is None:
            return float(np.asarray(self.bias_function(cvs)))
        if self.static_arg2 is None:
            return float(np.asarray(self.bias_function(cvs, self.static_arg1)))
        return float(np.asarray(self.bias_function(cvs, self.static_arg1, self.static_arg2)))

    def calc_forces(self, atoms_pos, cvs, cell=None, pbc=None):
        force_array = self.set_force_array(atoms_pos)
        deriv_E = self.calc_deriv_E_to_cvs(cvs)
        deriv_cvs = self.calc_deriv_cvs(atoms_pos, cell=cell, pbc=pbc)
        force_array -= np.dot(deriv_E, deriv_cvs)
        return force_array

    def calc_stress(self, atoms_pos, cell, pbc, volume, cell_rank):
        if cell_rank != 3 or volume <= 0.0:
            raise PropertyNotImplementedError(
                'stress requires a full-rank periodic cell with non-zero volume'
            )

        if self._stress_gradient is None:
            self._build_stress_gradient()

        strain = jnp.zeros(6, dtype=jnp.asarray(atoms_pos).dtype)
        strain_grad = self._stress_gradient(
            jnp.asarray(atoms_pos),
            strain,
            jnp.asarray(cell),
            jnp.asarray(pbc, dtype=bool),
        )
        return np.asarray(strain_grad) / volume

    def _build_stress_gradient(self):
        def energy_from_strain(atoms_pos, voigt_strain, cell, pbc):
            deformation = self._voigt_strain_to_deformation(voigt_strain, atoms_pos.dtype)
            strained_positions = atoms_pos @ deformation
            strained_cell = cell @ deformation
            cvs = self.cv_function(strained_positions, strained_cell, pbc)

            if self.static_arg1 is None and self.static_arg2 is None:
                return self.bias_function(cvs)
            if self.static_arg2 is None:
                return self.bias_function(cvs, self.static_arg1)
            return self.bias_function(cvs, self.static_arg1, self.static_arg2)

        if self.use_jit:
            self._stress_gradient = jit(grad(energy_from_strain, argnums=1))
        else:
            self._stress_gradient = grad(energy_from_strain, argnums=1)

    @staticmethod
    def _voigt_strain_to_deformation(voigt_strain, dtype):
        exx, eyy, ezz, eyz, exz, exy = voigt_strain
        half = jnp.asarray(0.5, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        return jnp.array(
            [
                [one + exx, half * exy, half * exz],
                [half * exy, one + eyy, half * eyz],
                [half * exz, half * eyz, one + ezz],
            ],
            dtype=dtype,
        )

    def set_force_array(self, atoms_pos):
        return np.zeros(atoms_pos.shape)

    def calc_deriv_E_to_cvs(self, cvs):
        if self.static_arg1 is None and self.static_arg2 is None:
            return np.asarray(self.deriv_bias_function(cvs))
        if self.static_arg2 is None:
            return np.asarray(self.deriv_bias_function(cvs, self.static_arg1))
        return np.asarray(self.deriv_bias_function(cvs, self.static_arg1, self.static_arg2))

    def calc_deriv_cvs(self, atoms_pos, cell=None, pbc=None):
        if cell is None:
            cell = np.zeros((3, 3))
        if pbc is None:
            pbc = np.zeros(3, dtype=bool)

        jacobian = np.asarray(
            self.deriv_cv(
                jnp.asarray(atoms_pos),
                jnp.asarray(cell),
                jnp.asarray(pbc, dtype=bool),
            )
        )

        if jacobian.ndim == 3:
            return np.moveaxis(jacobian, [0], [1])
        if jacobian.ndim == 2:
            return jacobian
        raise ValueError('There is an error in the dimensions of the CV derivative')

    def calc_hessian(self, f):
        return jacfwd(jacrev(f))

    def get_hessian(self):
        return self.results['hessian']
