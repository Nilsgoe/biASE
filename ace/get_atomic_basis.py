import jax.numpy as jnp
from jax import random
import math
from ace.radial import bessel
from ace.sp_harm_jax import spherical_harmonics
# Assume these functions are defined elsewhere:
# - bessel(r, n_rbf, x_max): returns an array of shape (n_rbf,)
# - spherical_harmonics((x, y, z), l_max): returns an array of shape ((l_max+1)**2,)
#   where the harmonics are ordered as described above.

def compute_phi(r, rhat, n_rbf, l, x_max):
    """
    Compute the composite basis function
       phi_v(r) = sqrt(4*pi) * R_{n,l}(r) * Y_l^m(rhat)
    
    Args:
        r (scalar): distance |r|.
        rhat (array): normalized direction, shape (3,).
        n_rbf (int): number of radial basis functions.
        l (int): desired angular momentum (we want 2*l+1 components).
        x_max (float): cutoff for the radial basis.
    
    Returns:
        phi (array): shape (n_rbf, 2*l+1)
    """
    # Compute the radial part (n_rbf,)
    R = bessel(r, n_rbf, x_max)
    
    # Compute the full angular part up to l_max = l.
    # This returns an array of shape ((l+1)**2,)
    Y_full = spherical_harmonics(
        (jnp.array([rhat[0]]),
         jnp.array([rhat[1]]),
         jnp.array([rhat[2]])), l)
    Y_full = jnp.squeeze(Y_full, axis=0)  # Now shape is ((l+1)**2,)
    
    # If l==0, then Y_full already has 1 component.
    # For l > 0, extract the block corresponding to l:
    if l > 0:
        start = l * l        # e.g., for l=1, start = 1
        end = (l + 1) ** 2   # for l=1, end = 4 (indices 1,2,3 => 3 components)
        Y = Y_full[start:end]
    else:
        Y = Y_full  # l==0 case.
    
    # Outer product: combine radial and angular parts.
    # R has shape (n_rbf,) and Y has shape (2*l+1,), so the outer product gives (n_rbf, 2*l+1)
    phi = jnp.outer(R, Y)#jnp.sqrt(4 * math.pi) * 
    return phi

def compute_atomic_basis(positions, neighbor_list, n_rbf, l, x_max):
    """
    Compute the atomic basis A_{i,v} for each atom i.
    
    A_{i,v} = sum_{j in neighbors(i)} phi_v(r_{ji}),
    where phi_v(r) = sqrt(4*pi) R_{n,l}(r) Y_l^m(rhat).
    
    Args:
        positions (array): shape (N, 3) positions.
        neighbor_list (list): list of length N with neighbor indices for each atom.
        n_rbf (int): number of radial basis functions.
        l (int): angular momentum (for fixed l, angular part has 2*l+1 components).
        x_max (float): radial cutoff.
    
    Returns:
        A (array): shape (N, n_rbf, 2*l+1)
    """
    N = positions.shape[0]
    
    def compute_A_for_atom(i):
        pos_i = positions[i]
        # Initialize the accumulator for atom i.
        A_i = jnp.zeros((n_rbf, 2*l+1))
        for j in neighbor_list[i]:
            r_vec = positions[j] - pos_i
            r = jnp.linalg.norm(r_vec)
            # Avoid division by zero.
            rhat = jnp.where(r > 0, r_vec / r, jnp.zeros_like(r_vec))
            phi_ij = compute_phi(r, rhat, n_rbf, l, x_max)
            A_i = A_i + phi_ij  # shape: (n_rbf, 2*l+1)
        return A_i

    # Compute the atomic basis for each atom.
    A_list = [compute_A_for_atom(i) for i in range(N)]
    A = jnp.stack(A_list, axis=0)
    return A

###############################################################################
# Example usage:
###############################################################################
if __name__ == "__main__":
    # Suppose we have 4 atoms.
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Define neighbor lists (here, every atom is neighbor to every other).
    neighbor_list = [
        jnp.array([1, 2, 3]), # Neighbors of atom 0
        jnp.array([0, 2, 3]), # Neighbors of atom 1
        jnp.array([0, 1, 3]),
        jnp.array([0, 1, 2])
    ]
    
    n_rbf = 5    # number of radial basis functions
    l = 1        # angular momentum; for l=1 we expect 2*1+1 = 3 angular components.
    x_max = 5.0  # cutoff
    
    A = compute_atomic_basis(positions, neighbor_list, n_rbf, l, x_max)
    print("Shape of atomic basis A:", A.shape)
    # Should be (4, 5, 3) for 4 atoms, 5 radial channels, and 3 angular components.
