import jax
import jax.numpy as jnp
from jax import random
from functools import partial

jit_static = partial(jax.jit, static_argnums=(3,))

@jit_static
def spherical_harmonics(x: jnp.ndarray, 
                        y: jnp.ndarray, 
                        z: jnp.ndarray, 
                        lmax: int) -> jnp.ndarray:
    # l = 0
    sh_0_0 = jnp.ones_like(x)
    if lmax == 0:
        return jnp.stack([sh_0_0], axis=-1)

    # l = 1 (first-order)
    sqrt3 = jnp.sqrt(3.0)
    sh_1_0 = sqrt3 * x
    sh_1_1 = sqrt3 * y@jax.jit(static_argnums=(3,))
    sh_1_2 = sqrt3 * z
    if lmax == 1:
        return jnp.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2], axis=-1)

    # l = 2 (second-order)
    sqrt15 = jnp.sqrt(15.0)
    y2 = y ** 2
    x2z2 = x**2 + z**2
    sh_2_0 = sqrt15 * x * z
    sh_2_1 = sqrt15 * x * y
    sh_2_2 = jnp.sqrt(5.0) * (y2 - 0.5 * x2z2)
    sh_2_3 = sqrt15 * y * z
    sh_2_4 = 0.5 * sqrt15 * (z**2 - x**2)
    if lmax == 2:
        return jnp.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2,
                          sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4], axis=-1)

    # l = 3 (third-order)
    sqrt42 = jnp.sqrt(42.0)
    sqrt7  = jnp.sqrt(7.0)
    sqrt168 = jnp.sqrt(168.0)
    sh_3_0 = (1/6) * sqrt42 * (sh_2_0 * z + sh_2_4 * x)
    sh_3_1 = sqrt7 * sh_2_0 * y
    sh_3_2 = (1/8) * sqrt168 * (4.0 * y2 - x2z2) * x
    sh_3_3 = 0.5 * sqrt7 * y * (2.0 * y2 - 3.0 * x2z2)
    sh_3_4 = (1/8) * sqrt168 * z * (4.0 * y2 - x2z2)
    sh_3_5 = sqrt7 * sh_2_4 * y
    sh_3_6 = (1/6) * sqrt42 * (sh_2_4 * z - sh_2_0 * x)
    if lmax == 3:
        return jnp.stack([
            sh_0_0, sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6
        ], axis=-1)

    # l = 4 (fourth-order)
    sh_4_0 = (3/4) * jnp.sqrt(2.0) * (sh_3_0 * z + sh_3_6 * x)
    sh_4_1 = (3/4) * sh_3_0 * y + (3/8) * jnp.sqrt(6.0) * sh_3_1 * z + (3/8) * jnp.sqrt(6.0) * sh_3_5 * x
    sh_4_2 = (-3/56) * jnp.sqrt(14.0) * sh_3_0 * z + (3/14) * jnp.sqrt(21.0) * sh_3_1 * y \
             + (3/56) * jnp.sqrt(210.0) * sh_3_2 * z + (3/56) * jnp.sqrt(210.0) * sh_3_4 * x \
             + (3/56) * jnp.sqrt(14.0) * sh_3_6 * x
    sh_4_3 = (-3/56) * jnp.sqrt(42.0) * sh_3_1 * z + (3/28) * jnp.sqrt(105.0) * sh_3_2 * y \
             + (3/28) * jnp.sqrt(70.0) * sh_3_3 * x + (3/56) * jnp.sqrt(42.0) * sh_3_5 * x
    sh_4_4 = - (3/28) * jnp.sqrt(42.0) * sh_3_2 * x + (3/7) * jnp.sqrt(7.0) * sh_3_3 * y \
             - (3/28) * jnp.sqrt(42.0) * sh_3_4 * z
    sh_4_5 = (-3/56) * jnp.sqrt(42.0) * sh_3_1 * x + (3/28) * jnp.sqrt(70.0) * sh_3_3 * z \
             + (3/28) * jnp.sqrt(105.0) * sh_3_4 * y - (3/56) * jnp.sqrt(42.0) * sh_3_5 * z
    sh_4_6 = (-3/56) * jnp.sqrt(14.0) * sh_3_0 * x - (3/56) * jnp.sqrt(210.0) * sh_3_2 * x \
             + (3/56) * jnp.sqrt(210.0) * sh_3_4 * z + (3/14) * jnp.sqrt(21.0) * sh_3_5 * y \
             - (3/56) * jnp.sqrt(14.0) * sh_3_6 * z
    sh_4_7 = - (3/8) * jnp.sqrt(6.0) * sh_3_1 * x + (3/8) * jnp.sqrt(6.0) * sh_3_5 * z + (3/4) * sh_3_6 * y
    sh_4_8 = (3/4) * jnp.sqrt(2.0) * (-sh_3_0 * x + sh_3_6 * z)
    if lmax == 4:
        return jnp.stack([
            sh_0_0, sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8
        ], axis=-1)

    # l = 5 (fifth-order)
    sh_5_0  = (1/10) * jnp.sqrt(110.0) * (sh_4_0 * z + sh_4_8 * x)
    sh_5_1  = (1/5)  * jnp.sqrt(11.0)  * sh_4_0 * y + (1/5)  * jnp.sqrt(22.0) * sh_4_1 * z + (1/5)  * jnp.sqrt(22.0) * sh_4_7 * x
    sh_5_2  = (-1/30) * jnp.sqrt(22.0)  * sh_4_0 * z + (4/15) * jnp.sqrt(11.0)  * sh_4_1 * y \
              + (1/15) * jnp.sqrt(154.0) * sh_4_2 * z + (1/15) * jnp.sqrt(154.0) * sh_4_6 * x \
              + (1/30) * jnp.sqrt(22.0)  * sh_4_8 * x
    sh_5_3  = (-1/30) * jnp.sqrt(66.0)  * sh_4_1 * z + (1/15) * jnp.sqrt(231.0) * sh_4_2 * y \
              + (1/30) * jnp.sqrt(462.0) * sh_4_3 * z + (1/30) * jnp.sqrt(462.0) * sh_4_5 * x \
              + (1/30) * jnp.sqrt(66.0)  * sh_4_7 * x
    sh_5_4  = (-1/15) * jnp.sqrt(33.0)  * sh_4_2 * z + (2/15) * jnp.sqrt(66.0)  * sh_4_3 * y \
              + (1/15) * jnp.sqrt(165.0) * sh_4_4 * x + (1/15) * jnp.sqrt(33.0)  * sh_4_6 * x
    sh_5_5  = (-1/15) * jnp.sqrt(110.0) * sh_4_3 * x + (1/3) * jnp.sqrt(11.0)  * sh_4_4 * y - (1/15) * jnp.sqrt(110.0) * sh_4_5 * z
    sh_5_6  = (-1/15) * jnp.sqrt(33.0)  * sh_4_2 * x + (1/15) * jnp.sqrt(165.0) * sh_4_4 * z \
              + (2/15) * jnp.sqrt(66.0)  * sh_4_5 * y - (1/15) * jnp.sqrt(33.0)  * sh_4_6 * z
    sh_5_7  = (-1/30) * jnp.sqrt(66.0)  * sh_4_1 * x - (1/30) * jnp.sqrt(462.0) * sh_4_3 * x \
              + (1/30) * jnp.sqrt(462.0) * sh_4_5 * z + (1/15) * jnp.sqrt(231.0) * sh_4_6 * y - (1/30) * jnp.sqrt(66.0)  * sh_4_7 * z
    sh_5_8  = (-1/30) * jnp.sqrt(22.0)  * sh_4_0 * x - (1/15) * jnp.sqrt(154.0) * sh_4_2 * x \
              + (1/15) * jnp.sqrt(154.0) * sh_4_6 * z + (4/15) * jnp.sqrt(11.0)  * sh_4_7 * y - (1/30) * jnp.sqrt(22.0)  * sh_4_8 * z
    sh_5_9  = - (1/5)  * jnp.sqrt(22.0) * sh_4_1 * x + (1/5)  * jnp.sqrt(22.0) * sh_4_7 * z + (1/5)  * jnp.sqrt(11.0)  * sh_4_8 * y
    sh_5_10 = (1/10) * jnp.sqrt(110.0) * (-sh_4_0 * x + sh_4_8 * z)
    return jnp.stack([
        sh_0_0, sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
        sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
        sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
        sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10
    ], axis=-1)







if __name__ == "__main__":
    key = random.PRNGKey(0)
    test_sizes = [2, 3, 4, 5]
    lmax_values = [0, 1, 2, 3, 4, 5]
    
    for size in test_sizes:
        x, y, z = random.normal(key, (size,)), random.normal(key, (size,)), random.normal(key, (size,)) 
        print(f"\nInput size: {size}")
        for lmax in lmax_values:
            sh_values = _spherical_harmonics(lmax, x, y, z)
            print(f"lmax={lmax}, Output shape: {sh_values.shape}")      