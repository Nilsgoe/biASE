import jax.numpy as jnp 
import jax
'''
def bessel(x: jax.Array, n: int, x_max: float = 1.0) -> jax.Array:
    r"""Bessel basis functions.

    They obey the following normalization:

    .. math::

        \int_0^c r^2 B_n(r, c) B_m(r, c) dr = \delta_{nm}

    Args:
        x (jax.Array): input of shape ``[...]``
        n (int): number of basis functions
        x_max (float): maximum value of the input

    Returns:
        jax.Array: basis functions of shape ``[..., n]``

    Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing for Molecular Graphs; ICLR 2020.
    Equation (7)
    """
    x = jnp.asarray(x)
    assert isinstance(n, int)

    x = x[..., None]
    n = jnp.arange(1, n + 1, dtype=x.dtype)
    x_nonzero = jnp.where(x == 0.0, 1.0, x)
    return jnp.sqrt(2.0 / x_max) * jnp.where(
        x == 0,
        n * jnp.pi / x_max,
        jnp.sin(n * jnp.pi / x_max * x_nonzero) / x_nonzero,
    )


def bessel_normalized(x: jnp.ndarray, n: int, x_max: float = 1.0) -> jnp.ndarray:
    """
    Computes normalized spherical Bessel basis functions for l=0.
    
    The basis functions are defined as:
    
        B_n(r) = sqrt(2/x_max) * ( sin(n*pi*r/x_max) / r )   for r > 0,
        B_n(0) = sqrt(2/x_max) * ( n*pi/x_max )                for r = 0.
    
    They are normalized such that:
    
        ∫_0^(x_max) r^2 B_n(r) B_m(r) dr = δ_nm.
    
    Args:
        x (jax.Array): input array of radii, shape [...]
        n (int): number of basis functions (n=1,2,...,n)
        x_max (float): maximum radius (c) for normalization
    
    Returns:
        jax.Array: array of basis functions of shape [..., n]
    """
    # Ensure x is an array and add an axis for the basis function index.
    x = jnp.asarray(x)
    x = x[..., None]  # Now shape [..., 1]
    
    # Create an array of basis function indices 1, 2, ..., n.
    n_arr = jnp.arange(1, n + 1, dtype=x.dtype)
    
    # Avoid division by zero by substituting a safe value (1.0) where x==0.
    x_safe = jnp.where(x == 0.0, 1.0, x)
    
    # Compute the sine term divided by r.
    B = jnp.sin(n_arr * jnp.pi * x_safe / x_max) / x_safe
    
    # Replace values at x==0 with the limiting value n*pi/x_max.
    B = jnp.where(x == 0.0, n_arr * jnp.pi / x_max, B)
    
    # Multiply by the normalization constant.
    return jnp.sqrt(2.0 / x_max) * B



'''
from functools import partial
import jax.numpy as jnp
jit_static_cut = partial(jax.jit, static_argnums=(1,2))

@jit_static_cut
def polynomial_cutoff_custom(r, r_cut, p=5):
    """
    Compute the cutoff function u(d) = 1 - ((p+1)(p+2)/2)*d^p + (p(p+2))*d^(p+1) - (p(p+1)/2)*d^(p+2),
    where d = r/r_cut, for r < r_cut, and returns 0 for r >= r_cut.

    Parameters:
        r (jax.Array): The distance (or an array of distances).
        r_cut (float): The cutoff radius.
        p (int): The exponent parameter (default is 5).

    Returns:
        jax.Array: The cutoff function values.
    """
    d = r / r_cut
    u_d = 1 - ((p + 1) * (p + 2) / 2) * d**p + (p * (p + 2)) * d**(p + 1) - (p * (p + 1) / 2) * d**(p + 2)
    return jnp.where(r < r_cut, u_d, 0.0)

jit_static_cut = partial(jax.jit, static_argnums=(1,2))

@jit_static_cut
def polynomial_cutoff(r, r_cut, p=2):
    """
    A polynomial cutoff function that goes smoothly to zero at r = r_cut.

    Args:
        r (jax.Array): Input distance (can be scalar or array).
        r_cut (float): The cutoff radius.
        p (int): The power of the cutoff; default is 2.

    Returns:
        jax.Array: The cutoff factor, equal to (1 - r/r_cut)^p for r < r_cut and 0 for r >= r_cut.
    """
    # Compute the basic factor (1 - r/r_cut)
    factor = 1 - r / r_cut
    # For values r >= r_cut, we want the cutoff to be 0.
    return jnp.where(r < r_cut, factor ** p, 0.0)


jit_static_bessel = partial(jax.jit, static_argnums=(1,2,3))

@jit_static_bessel
def bessel(x, n, x_max=1.0, p_cut=2):
    """
    Compute the normalized Bessel-like basis functions with a polynomial cutoff.

    The function implements:
    
       j_0^n(r) = sqrt(2/r_cut) * [ sin(n*pi*(r/r_cut)) / r ] * f_cut(r)
       
    where f_cut(r) is a polynomial cutoff function which goes to zero smoothly at r = r_cut.
    
    Args:
        x (jax.Array): Input distance r (or r_ij).
        n (int): Number of basis functions (or the index n in the sine argument).
        x_max (float): The cutoff radius r_cut.
        p_cut (int): The exponent used in the cutoff function (default 2).
    
    Returns:
        jax.Array: An array of shape [..., n] containing the basis functions.
    """
    # Ensure x is a JAX array and add a trailing axis for broadcasting with n.
    x = jnp.asarray(x)
    # Create an axis for the basis index.
    # n_arr goes from 1 to n (inclusive)
    n_arr = jnp.arange(1, n + 1, dtype=x.dtype)
    
    # To avoid division by zero, replace zeros with ones (we'll fix those later).
    x_safe = jnp.where(x == 0.0, 1.0, x)
    
    # Compute the sine term.
    sine_term = jnp.sin(n_arr * jnp.pi * x_safe / x_max) / x_safe
    # For r==0, take the limit: sin(n*pi*0/x_max)/0 -> n*pi/x_max.
    sine_term = jnp.where(x == 0.0, n_arr * jnp.pi / x_max, sine_term)
    
    # Compute the basic normalization factor.
    norm = jnp.sqrt(2.0 / x_max)
    
    # Compute the cutoff function using our polynomial cutoff.
    fcut = polynomial_cutoff_custom(x, x_max, p=p_cut)
    
    # Multiply together: normalization, sine term, and cutoff.
    return norm * sine_term * fcut


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # Define the maximum x value
    x_max = 1.0

    # Generate x values from 0 to x_max.
    x = jnp.linspace(0, x_max+0.1, 400)
    
    
    x_np = np.array(x)
    
    
    # Create a figure.
    plt.figure(figsize=(8, 5))
    ys = []
    # Plot basis functions for orders 0 to 8.
    for xs in x:
        y = bessel(xs, 8, x_max)
        ys.append(y)
    # Convert from JAX array to numpy array for plotting.
    y_np = np.array(ys)
    for i in range(0,8):
        plt.plot(x_np, y_np[:, i], label=f"n={i+1}")
    #Sum plotting
    #plt.plot(x_np, np.sum(y_np,axis=1), label=f"sum")
    plt.xlabel("x")
    plt.ylabel("Bessel basis function")
    plt.title("Bessel basis functions for orders 1 to 8")
    plt.legend()
    plt.grid(True)
    plt.show()