
import jax
import jax.numpy as jnp
from jax import random, vmap
import math
from ace.radial_jit import bessel
from ace.sp_harm_jax_jit import spherical_harmonics
from functools import partial

# Mark n_rbf and l as static for compute_phi.
jit_static_phi = partial(jax.jit, static_argnums=(2,3,4))
@jit_static_phi
def compute_phi(r, rhat, n_rbf, l, x_max):
    """
    Compute phi_v(r) = sqrt(4*pi) * R_{n,l}(r) * Y_l^m(rhat)
    for a given distance r and direction rhat.
    Returns an array of shape (n_rbf, 2*l+1).
    """
    # Ensure n_rbf is a concrete integer.
    n_rbf = int(n_rbf)
    R = bessel(r, n_rbf, x_max)  # shape: (n_rbf,)
    
    # Compute full angular part up to order l.
    Y_full = spherical_harmonics(
        jnp.array([rhat[0]]),
         jnp.array([rhat[1]]),
         jnp.array([rhat[2]]), l)
    Y_full = jnp.squeeze(Y_full, axis=0)
    
    # For l > 0, extract the (2*l+1) components; for l==0, use Y_full directly.
    Y = Y_full if l == 0 else Y_full[l * l:(l + 1) ** 2]
    
    phi = jnp.outer(R, Y)  # shape: (n_rbf, 2*l+1)
    return phi

# Mark neighbor_list and n_rbf as static for compute_atomic_basis.
jit_static_basis = partial(jax.jit, static_argnums=(1,2,3,4))
@jit_static_basis
def compute_atomic_basis(positions, neighbor_list, n_rbf, l, x_max):
    # Convert the neighbor_list (which should be hashable, e.g. a tuple of tuples)
    # into a list of JAX arrays for indexing.
    neighbor_list = [jnp.array(nbr) for nbr in neighbor_list]
    
    # Ensure the number of atoms is a concrete Python int.
    N = int(positions.shape[0])
    
    # Build padded neighbor arrays and a mask.
    max_neighbors = max(arr.shape[0] for arr in neighbor_list)
    padded_neighbors = []
    mask = []
    for i in range(N):
        n_neighbors = int(neighbor_list[i].shape[0])
        pad_width = max_neighbors - n_neighbors
        padded = jnp.pad(neighbor_list[i], (0, pad_width), constant_values=-1)
        padded_neighbors.append(padded)
        mask.append(jnp.concatenate([
            jnp.ones(n_neighbors, dtype=bool),
            jnp.zeros(pad_width, dtype=bool)
        ]))
    padded_neighbors = jnp.stack(padded_neighbors)  # shape: (N, max_neighbors)
    mask = jnp.stack(mask)                          # shape: (N, max_neighbors)
    
    def atom_basis(i):
        pos_i = positions[i]
        neigh_indices = padded_neighbors[i]  # shape: (max_neighbors,)
        valid = mask[i]                      # shape: (max_neighbors,)
        # Use jnp.where with indexing: if an index is negative, replace with zeros.
        neigh_pos = jnp.where(neigh_indices[:, None] >= 0,
                              positions[neigh_indices],
                              jnp.zeros((max_neighbors, 3)))
        r_vecs = neigh_pos - pos_i
        rs = jnp.linalg.norm(r_vecs, axis=1)
        rhat = jnp.where(rs[:, None] > 0,
                         r_vecs / rs[:, None],
                         jnp.zeros_like(r_vecs))
        # Vectorize compute_phi over neighbors.
        phis = vmap(lambda r, rhat: compute_phi(r, rhat, n_rbf, l, x_max))(rs, rhat)
        phis = jnp.where(valid[:, None, None], phis, 0.0)
        return jnp.sum(phis, axis=0)
    
    # Use a concrete range for the atoms.
    A = vmap(atom_basis)(jnp.arange(N))
    return A

# Example usage:
if __name__ == "__main__":
    # Suppose we have 4 atoms.
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Define neighbor lists.
    # IMPORTANT: Convert neighbor_list to a tuple of tuples so it's hashable.
    hashable_neighbor_list = (
        (1, 2, 3),  # Neighbors of atom 0
        (0, 2, 3),  # Neighbors of atom 1
        (0, 1, 3),
        (0, 1, 2)
    )
    
    n_rbf = 5    # number of radial basis functions (static)
    l = 1        # angular momentum; for l=1 we expect 3 angular components.
    x_max = 5.0  # cutoff
    
    A = compute_atomic_basis(positions, hashable_neighbor_list, n_rbf, l, x_max)
    print("Shape of atomic basis A:", A.shape)
    # Expected shape: (4, 5, 3)
