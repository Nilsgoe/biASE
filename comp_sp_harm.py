from sp_harm import _spherical_harmonics
from sp_harm_jax import _spherical_harmonics as _spherical_harmonics_jax
from sp_harm_jax import spherical_harmonics
from jax import random
import torch
key = random.PRNGKey(0)
test_sizes = [2, 3, 4, 5]
lmax_values = [0, 1, 2, 3, 4, 5]
import numpy as np
from jax import jacfwd 
def arrays_are_close(jax_array, torch_tensor, tol=1e-38):
    return np.allclose(np.array(jax_array), torch_tensor.detach().numpy(), atol=tol)

for size in test_sizes:
    #x, y, z = random.normal(key, (size,)), random.normal(key, (size,)), random.normal(key, (size,)) 
    xyz = random.normal(key, (3, size))  # Shape (3, size)

    x, y, z = xyz 
    print(f"\nInput size: {size}")
    for lmax in lmax_values:
        x_=torch.Tensor(x.tolist())
        y_=torch.Tensor(y.tolist())   
        z_=torch.Tensor(z.tolist())   
        sh_values = _spherical_harmonics(lmax, x_, y_, z_)
        sh_values_jax = _spherical_harmonics_jax(lmax, x, y, z)
        print(f"lmax={lmax}, Output shape torch: {sh_values.shape}, jax: {sh_values_jax.shape}")
        print(f'Are the jax and torch implementation the same: {arrays_are_close(sh_values_jax, sh_values)}')
        print(sh_values[0].detach().numpy()- np.array(sh_values_jax[0]),"\n") # they are exactly the same - crazy
        #print(f'{sh_values_jax[0]-spherical_harmonics(xyz, lmax)[0]}')

grad_fn = jacfwd(spherical_harmonics)
print(grad_fn(xyz, lmax).shape)