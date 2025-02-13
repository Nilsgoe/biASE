import math
import jax.numpy as jnp

def clebsch_gordan_jax(l1: int, l2: int, l3: int) -> jnp.ndarray:
    r"""The Clebsch-Gordan coefficients of the real irreducible representations of :math:`SO(3)`.

    Args:
        l1 (int): the representation order of the first irrep
        l2 (int): the representation order of the second irrep
        l3 (int): the representation order of the third irrep

    Returns:
        jnp.ndarray: the Clebsch-Gordan coefficients
    """
    C = su2_clebsch_gordan_jax(l1, l2, l3)
    Q1 = change_basis_real_to_complex(l1)
    Q2 = change_basis_real_to_complex(l2)
    Q3 = change_basis_real_to_complex(l3)
    C = jnp.einsum("ij,kl,mn,ikn->jlm", Q1, Q2, jnp.conj(Q3.T), C)

    assert jnp.all(jnp.abs(jnp.imag(C)) < 1e-5)
    return jnp.real(C)


def su2_clebsch_gordan_jax(j1: float, j2: float, j3: float) -> jnp.ndarray:
    """Calculates the Clebsch-Gordon matrix."""
    assert isinstance(j1, (int, float))
    assert isinstance(j2, (int, float))
    assert isinstance(j3, (int, float))
    mat = jnp.zeros((int(2 * j1 + 1), int(2 * j2 + 1), int(2 * j3 + 1)))
    if int(2 * j3) in range(int(2 * abs(j1 - j2)), int(2 * (j1 + j2)) + 1, 2):
        for m1 in (x / 2 for x in range(-int(2 * j1), int(2 * j1) + 1, 2)):
            for m2 in (x / 2 for x in range(-int(2 * j2), int(2 * j2) + 1, 2)):
                if abs(m1 + m2) <= j3:
                    mat = mat.at[int(j1 + m1), int(j2 + m2), int(j3 + m1 + m2)].set(
                    _su2_cg_jax((j1, m1), (j2, m2), (j3, m1 + m2))
                )
    return mat / math.sqrt(2 * j3 + 1)

def change_basis_real_to_complex(l: int) -> jnp.ndarray:
    # https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    q = jnp.zeros((2 * l + 1, 2 * l + 1), dtype=jnp.complex64)
    for m in range(-l, 0):
        q = q.at[l + m, l + abs(m)].set(1 / jnp.sqrt(2))
        q = q.at[l + m, l - abs(m)].set(-1j / jnp.sqrt(2))
    
    # Set the m = 0 element.
    q = q.at[l, l].set(1)
    
    # Loop over positive m values.
    for m in range(1, l + 1):
        q = q.at[l + m, l + abs(m)].set(((-1) ** m) / jnp.sqrt(2))
        q = q.at[l + m, l - abs(m)].set(1j * ((-1) ** m) / jnp.sqrt(2))
    
    # Added factor of 1j**l to make the Clebsch-Gordan coefficients real.
    return (-1j) ** l * q


def _su2_cg_jax(idx1, idx2, idx3):
    """Calculates the Clebsch-Gordon coefficient.

    Copied from the ``clebsch`` function in ``qutip``.
    """
    from fractions import Fraction

    j1, m1 = idx1
    j2, m2 = idx2
    j3, m3 = idx3

    if m3 != m1 + m2:
        return 0
    vmin = int(max([-j1 + j2 + m3, -j1 + m1, 0]))
    vmax = int(min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]))

    def f(n):
        assert n == round(n)
        return math.factorial(round(n))

    C = (
        (2.0 * j3 + 1.0)
        * Fraction(
            f(j3 + j1 - j2)
            * f(j3 - j1 + j2)
            * f(j1 + j2 - j3)
            * f(j3 + m3)
            * f(j3 - m3),
            f(j1 + j2 + j3 + 1) * f(j1 - m1) * f(j1 + m1) * f(j2 - m2) * f(j2 + m2),
        )
    ) ** 0.5

    S = 0
    for v in range(vmin, vmax + 1):
        S += (-1.0) ** (v + j2 + m2) * Fraction(
            f(j2 + j3 + m1 - v) * f(j1 - m1 + v),
            f(v) * f(j3 - j1 + j2 - v) * f(j3 + m3 - v) * f(v + j1 - j2 - m3),
        )
    C = C * S
    return C


def test_clebsch_gordan_jax_symmetry():
    assert jnp.allclose(
        clebsch_gordan_jax(1, 2, 3), jnp.swapaxes(clebsch_gordan_jax(1, 3, 2), 1, 2)
    )
    assert jnp.allclose(
        clebsch_gordan_jax(1, 2, 3), jnp.swapaxes(clebsch_gordan_jax(2, 1, 3), 0, 1)
    )
    assert jnp.allclose(
        clebsch_gordan_jax(1, 2, 3), jnp.swapaxes(clebsch_gordan_jax(3, 2, 1), 0, 2)
    )
    assert jnp.allclose(
        clebsch_gordan_jax(1, 2, 3),
        jnp.swapaxes(jnp.swapaxes(clebsch_gordan_jax(3, 1, 2), 0, 1), 1, 2),
    )
    assert jnp.allclose(
        clebsch_gordan_jax(1, 2, 3),
        jnp.swapaxes(jnp.swapaxes(clebsch_gordan_jax(2, 3, 1), 0, 2), 1, 2),
    )
    assert jnp.allclose(
        clebsch_gordan_jax(3, 2, 4), -jnp.swapaxes(clebsch_gordan_jax(3, 4, 2), 1, 2)
    )
    assert jnp.allclose(
        clebsch_gordan_jax(2, 3, 4), -jnp.swapaxes(clebsch_gordan_jax(4, 3, 2), 0, 2)
    )


def unique_triplets(lmax):
    for l1 in range(lmax + 1):
        for l2 in range(l1 + 1):
            for l3 in range(l2 + 1):
                if abs(l1 - l2) <= l3 <= l1 + l2:
                    yield (l1, l2, l3)



'''def test_clebsch_gordan_jax(keys, l1, l2, l3):
    abc = rand_angles(keys[0], (10,))

    C = clebsch_gordan_jax(l1, l2, l3)
    D1 = Irrep(l1, 1).D_from_angles(*abc)
    D2 = Irrep(l2, 1).D_from_angles(*abc)
    D3 = Irrep(l3, 1).D_from_angles(*abc)

    C2 = jnp.einsum("ijk,zil,zjm,zkn->zlmn", C, D1, D2, D3)
    assert jnp.allclose(C, C2, atol=1e-3, rtol=1e-3)'''
    
if __name__ == "__main__":
    print("Testing Clebsch-Gordan symmetry...")
    test_clebsch_gordan_jax_symmetry()
    print("All symmetry tests passed!")
