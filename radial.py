import jax.numpy as jnp 
import jax
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

    # Plot basis functions for orders 0 to 8.
  
    y = bessel(x, 8, x_max)
    # Convert from JAX array to numpy array for plotting.
    y_np = np.array(y)
    plt.plot(x_np, y_np, label=f"n={8}")

    plt.xlabel("x")
    plt.ylabel("Bessel basis function")
    plt.title("Bessel basis functions for orders 0 to 8")
    plt.legend()
    plt.grid(True)
    plt.show()