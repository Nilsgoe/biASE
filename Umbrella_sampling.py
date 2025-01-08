from biASE import General_Bias_Calculator
import jax.numpy as jnp


def umbrella_potential_2d(x, x0, k):
    dx = x - x0
    #print(dx)
    return 0.5 * k * jnp.sum(dx**2)

def umbrella_potential(x, x0, k):
    """
    Compute a harmonic bias potential for a dihedral angle using JAX, handling boundary conditions smoothly.

    Parameters:
        phi (float): Current dihedral angle in degrees.
        phi_target (float): Target dihedral angle in degrees.
        k (float): Force constant.

    Returns:
        float: Harmonic bias potential.
    """
    # Wrap angles to [-180, 180]
    phi_wrapped = (x + 180) % 360 - 180
    phi_target_wrapped = (x0 + 180) % 360 - 180

    # Compute the potential with smooth boundary handling
    delta_phi = phi_wrapped - phi_target_wrapped
    print(delta_phi)
    delta_phi = (delta_phi + 180) % 360 - 180 
    print(delta_phi)
    # Compute harmonic potential
    bias = 0.5 * k * delta_phi**2
    return bias