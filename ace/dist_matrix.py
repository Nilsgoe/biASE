import jax.numpy as jnp

def compute_pairwise_differences(positions: jnp.ndarray) -> jnp.ndarray:
    """
    Compute pairwise differences between atom positions.
    
    Args:
        positions (jnp.ndarray): Array of shape (N, 3) representing positions of N atoms.
        
    Returns:
        jnp.ndarray: Array of shape (N, N, 3) where the (i, j) entry is positions[i] - positions[j].
    """
    # Using broadcasting, subtract each position from every other.
    # positions[:, None, :] has shape (N, 1, 3)
    # positions[None, :, :] has shape (1, N, 3)
    # Their difference yields shape (N, N, 3)
    return positions[:, None, :] - positions[None, :, :]

def compute_pairwise_distances(positions: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the Euclidean distance matrix between atoms.

    Args:
        positions (jnp.ndarray): Array of shape (N, 3) representing positions of N atoms.

    Returns:
        jnp.ndarray: Distance matrix of shape (N, N) where each element (i, j) is the 
                     Euclidean distance between positions[i] and positions[j].
    """
    # Compute pairwise differences using broadcasting.
    diffs = positions[:, None, :] - positions[None, :, :]  # Shape: (N, N, 3)
    # Compute the Euclidean norm along the last axis.
    distances = jnp.linalg.norm(diffs, axis=-1)            # Shape: (N, N)
    return distances


if __name__ == "__main__":
    import ase
    from ase import Atoms
    import numpy as np

    # Example: Create a small molecule (e.g., water)
    # Note: In practice, you'll use your ASE atoms object.
    water = Atoms("H2O", positions=[[0.0, 0.0, 0.0],
                                    [0.0, 0.96, 0.0],
                                    [0.76, -0.32, 0.0]])
    # Obtain positions from the ASE Atoms object (this is a NumPy array).
    positions_np = water.positions  # shape (3, 3)
    
    # Convert the NumPy array to a JAX array.
    positions_jax = jnp.array(positions_np)
    
    # Compute pairwise differences.
    differences = compute_pairwise_differences(positions_jax)
    
    # Convert to NumPy for printing/plotting if needed.
    differences_np = np.array(differences)
    
    print("Pairwise differences shape:", differences_np.shape)
    print("Pairwise differences:\n", differences_np)
    
    distances = compute_pairwise_distances(positions_jax)
    distances_np = np.array(distances)  # Convert to NumPy for printing if needed.
    
    print("Euclidean distance matrix:")
    print(distances_np)
