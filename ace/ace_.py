import jax
import jax.numpy as jnp
from functools import partial

def cutoff_fn(r, cutoff):
    """Cosine cutoff function."""
    return jnp.where(r <= cutoff, 0.5 * (jnp.cos(jnp.pi * r / cutoff) + 1.0), 0.0)

def radial_function(r, n, cutoff):
    """Radial basis function (polynomial * cutoff)."""
    return (r ** n) * cutoff_fn(r, cutoff)

def real_spherical_harmonics(theta, phi, l_max):
    """Computes real spherical harmonics up to l_max (hardcoded for l_max=2)."""
    Y = []
    sqrt_pi = jnp.sqrt(jnp.pi)
    x = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    
    # l=0
    Y.append((0, 0, 1.0 / (2 * sqrt_pi)))
    if l_max >= 1:
        # l=1
        Y.append((1, -1, jnp.sqrt(3/(4*jnp.pi)) * sin_theta * jnp.sin(phi)))
        Y.append((1, 0, jnp.sqrt(3/(4*jnp.pi)) * x))
        Y.append((1, 1, jnp.sqrt(3/(4*jnp.pi)) * sin_theta * jnp.cos(phi)))
    if l_max >= 2:
        # l=2
        cos_theta_sq = x**2
        Y.append((2, -2, jnp.sqrt(15/(16*jnp.pi)) * sin_theta**2 * jnp.sin(2*phi)))
        Y.append((2, -1, jnp.sqrt(15/(4*jnp.pi)) * sin_theta * x * jnp.sin(phi)))
        Y.append((2, 0, jnp.sqrt(5/(16*jnp.pi)) * (3*cos_theta_sq - 1)))
        Y.append((2, 1, jnp.sqrt(15/(4*jnp.pi)) * sin_theta * x * jnp.cos(phi)))
        Y.append((2, 2, jnp.sqrt(15/(16*jnp.pi)) * sin_theta**2 * jnp.cos(2*phi)))
    
    Y_sorted = sorted(Y, key=lambda x: (x[0], x[1]))
    lm_pairs = [(y[0], y[1]) for y in Y_sorted]
    values = jnp.array([y[2] for y in Y_sorted])
    return lm_pairs, values

def cartesian_to_spherical(pos):
    """Convert Cartesian coordinates to spherical coordinates."""
    x, y, z = pos
    r = jnp.sqrt(x**2 + y**2 + z**2)
    theta = jnp.arccos(z / r)
    phi = jnp.arctan2(y, x)
    return r, theta, phi

def compute_ACE_descriptor(neighbor_positions, cutoff, n_max, l_max):
    """Compute ACE descriptor (power spectrum version) for a central atom."""
    # Convert to spherical coordinates
    r, theta, phi = jax.vmap(cartesian_to_spherical)(neighbor_positions)
    
    # Radial basis (n_max functions)
    n_values = jnp.arange(n_max)
    radial = jax.vmap(lambda ri: jax.vmap(lambda ni: radial_function(ri, ni, cutoff))(n_values))(r)
    
    # Angular basis (real spherical harmonics)
    theta_phi = jnp.stack([theta, phi], axis=1)
    def compute_angular(tp):
        t, p = tp
        _, Y = real_spherical_harmonics(t, p, l_max)
        return Y
    angular = jax.vmap(compute_angular)(theta_phi)
    
    # Compute A_nlm = sum_neighbors (radial_n * angular_lm)
    phi_nlm = jax.vmap(lambda r, a: jnp.outer(r, a))(radial, angular)
    A_nlm = jnp.sum(phi_nlm, axis=0)
    
    # Generate l indices for angular components
    lm_pairs, _ = real_spherical_harmonics(0.0, 0.0, l_max)
    l_indices = jnp.array([lm[0] for lm in lm_pairs])
    
    # Compute power spectrum for each l
    descriptor = []
    for l in range(l_max + 1):
        mask = l_indices == l
        A_l = A_nlm[:, mask]
        p_l = jnp.einsum('nm,km->nk', A_l, A_l)
        descriptor.append(p_l.flatten())
    
    return jnp.concatenate(descriptor)

# Example usage
if __name__ == "__main__":
    cutoff = 5.0
    n_max = 4  # Number of radial basis functions
    l_max = 2  # Maximum angular degree

    # Example neighbor positions (relative to central atom)
    neighbor_positions = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0]
    ])

    descriptor = compute_ACE_descriptor(neighbor_positions, cutoff, n_max, l_max)
    print("ACE descriptor:", descriptor)
    print("Descriptor shape:", descriptor.shape)