# biASE

`biASE` contains small ASE-compatible tools for adding JAX-defined bias potentials to atomistic simulations.

## Main Modules

### `biASE.py`

This file provides the calculators used by the rest of the project.

- `General_Bias_Calculator`
  Evaluates a bias as `bias_function(cv_function(positions))`.
  It returns ASE energies and forces, and now also supports ASE stress for full-rank periodic cells.
  Forces are computed with the original chain-rule structure:
  `dE/dx = (dE/dCV) (dCV/dx)`.
  Stress is only built and evaluated when ASE explicitly requests `stress`, so MD runs that only need energies and forces keep the lighter code path.

- `Direct_General_Bias_Calculator`
  Evaluates a bias directly from positions without a separate collective-variable function.
  This is useful for metadynamics or wall potentials written directly in Cartesian coordinates.

### `Umbrella_sampling.py`

This file defines `UmbrellaMD`, a light wrapper for umbrella sampling with ASE MD drivers.

- It combines the system's original calculator with a harmonic umbrella bias using `SumCalculator`.
- `x0` is the umbrella center.
- `k` is the harmonic force constant.
- One-dimensional and two-dimensional CVs are supported.
- `setup_new_window()` reuses the same object for a new umbrella window.
- `run()` delegates to the provided ASE MD object.

Typical workflow:

1. Attach your physical calculator to an `Atoms` object.
2. Define a JAX-compatible CV function.
3. Construct `UmbrellaMD`.
4. Optionally optimize in the window.
5. Run MD.

### `Metadynamics.py`

This file contains ASE-based metadynamics drivers that build history-dependent Gaussian bias potentials on top of an existing calculator.

- It stores previously visited CV values.
- It adds Gaussian hills at a chosen interval.
- It combines the main potential and the bias with ASE calculator mixing.
- It is intended for bias-enhanced MD and related workflows.

`Metadynamics.py` relies on the calculators from `biASE.py`, so keeping the bias calculator ASE-compatible is important for both MD and metadynamics use.

## Testing

Tests live in `tests/`.

- `tests/test_biase.py`
  Compares analytical forces and stress from `General_Bias_Calculator` against ASE numerical derivatives.

- `tests/test_umbrella_sampling.py`
  Checks that `UmbrellaMD` builds the umbrella calculator correctly and forwards MD execution to the wrapped MD object.

The tests are written with `unittest`. If JAX is not installed in the active Python environment, the JAX-dependent tests are skipped automatically.

Run them with:

```bash
python -m unittest discover -s tests -v
```

## Notebooks

The repository also contains exploratory notebooks. They are useful as examples, but they are not the primary test source of truth.

- `Umbrella_sampling_test.ipynb`
  Still compatible with the current calculator.
  It mostly uses the older notebook-style pattern of building umbrella biases with `General_Bias_Calculator` and `functools.partial(...)` instead of the newer `UmbrellaMD` helper.
  That workflow still works, but new umbrella workflows should prefer `Umbrella_sampling.py`.

- `Metadynamics_test.ipynb`
  Mostly contains inline class definitions and experiments rather than importing the current module implementation.
  It can still be useful for ideas and plots, but it should be treated as a historical notebook, not as a synchronized test of the current `Metadynamics.py`.

- `Metadynamics_updated_testing.ipynb`
  Also contains substantial inline metadynamics code and local experimentation.
  It is closer in spirit to the current direct-bias workflow, but it is still not a guaranteed reflection of the exact current module code.

Practical recommendation:

1. Use `biASE.py`, `Umbrella_sampling.py`, and `Metadynamics.py` as the maintained implementation.
2. Use the notebooks for exploration, plotting, and recipe examples.
3. If a notebook becomes important for regular use, move the critical workflow into a test or script so it stays synchronized with code changes.

## Notes

- Bias and CV functions should be JAX-compatible.
- Stress is available only for systems with a full-rank simulation cell and non-zero volume.
- For periodic CVs that depend on the cell, the CV function may accept optional `cell` / `cell_input` and `pbc` / `pbc_input` arguments.
