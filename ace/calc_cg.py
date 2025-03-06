import numpy as np
from ace.clebsch_gordan_jax import clebsch_gordan_jax

def precompute_cg(lmax=5):  # using lmax=5 as you mentioned it's enough
    cg_cache = {}
    for l1 in range(lmax + 1):
        for l2 in range(lmax + 1):
            for l3 in range(lmax + 1):
                if abs(l1 - l2) <= l3 <= l1 + l2:
                    # Compute the CG tensor
                    cg_tensor = clebsch_gordan_jax(l1, l2, l3)
                    # Store as a NumPy array.
                    cg_cache[(l1, l2, l3)] = np.array(cg_tensor)
    return cg_cache

cg_cache = precompute_cg(5)

# Convert the keys to strings.
cg_cache_str = {str(key): value for key, value in cg_cache.items()}

np.savez("cg_cache.npz", **cg_cache_str)
