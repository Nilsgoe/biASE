import jax.numpy as jnp
import numpy as np

# Import the two versions.
# Assume these functions are available from your modules.
from clebsch_gordan import clebsch_gordan  # The normal (non-JAX) version.
from clebsch_gordan_jax import clebsch_gordan_jax  # The JAX version.

def valid_triplets(lmax):
    """
    Generate all triplets (l1, l2, l3) such that:
      0 <= l1, l2 <= lmax and |l1-l2| <= l3 <= l1+l2.
    """
    for l1 in range(lmax + 1):
        for l2 in range(lmax + 1):
            # l3 runs from |l1-l2| to l1+l2.
            for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                yield (l1, l2, l3)

def compare_clebsch_gordan(lmax=3, atol=1e-5):
    """
    Compare the standard and JAX implementations for Clebsch-Gordan coefficients
    for all valid triplets (l1, l2, l3) up to lmax.
    """
    all_match = True
    for l1, l2, l3 in valid_triplets(lmax):
        # Compute the coefficients from each implementation.
        cg_standard = clebsch_gordan(l1, l2, l3)
        cg_jax = clebsch_gordan_jax(l1, l2, l3)
        
        # Convert to JAX array (if necessary) so we can use jnp.allclose.
        cg_standard = jnp.array(cg_standard)
        
        if not jnp.allclose(cg_standard, cg_jax, atol=atol):
            print(f"Mismatch for (l1, l2, l3)=({l1}, {l2}, {l3})")
            # Optionally, print the maximum difference.
            diff = jnp.max(jnp.abs(cg_standard - cg_jax))
            print(f"  Maximum absolute difference: {diff}")
            all_match = False
        else:
            print(f"Match for (l1, l2, l3)=({l1}, {l2}, {l3})")
    return all_match

if __name__ == "__main__":
    # Set the maximum l value for testing.
    lmax_test = 5
    print("Comparing Clebsch-Gordan coefficients between standard and JAX implementations:")
    all_good = compare_clebsch_gordan(lmax=lmax_test)
    if all_good:
        print("All tests passed!")
    else:
        print("There were mismatches.")
