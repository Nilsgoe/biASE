import jax
import jax.numpy as jnp
from jax import random
from functools import partial

# Assume that we have a JAX version of the Clebsch–Gordan function available:
from ace.clebsch_gordan_jax import clebsch_gordan_jax as cg_jax
from ace.get_atomic_basis_jit import compute_atomic_basis

###############################################################################
# Helper: mapping from m to index
###############################################################################
def m_index(m, l):
    # For a given angular momentum l and magnetic quantum number m (with -l <= m <= l),
    # the index in an array of length 2*l+1 is given by:
    return int(m + l)

###############################################################################
# 1. First-body (1B) invariant:
###############################################################################
@jax.jit
def compute_B1(A0):
    # A0 is assumed to have shape (N_atoms, n_rbf, 1) for l=0.
    return A0[..., 0]  # shape: (N_atoms, n_rbf)

###############################################################################
# 2. Second-body (2B) invariant:
###############################################################################
# Create a version of jit with static_argnums preset for n_max (argument index 1).
jit_static = partial(jax.jit, static_argnums=(1,))

@jit_static
def compute_B2_l(A_l, n_max):
    """
    Compute the 2B invariant for a single angular momentum l.
    A_l has shape (N_atoms, n_rbf, 2*l+1). We slice to n_max radial channels.
    Here, n_max is treated as static.
    """
    # A_l.shape[1] is assumed to be a static value.
    n_rbf_original = A_l.shape[1]
    n_new = min(n_rbf_original, n_max)
    # Slice along the radial dimension.
    A_l = A_l[:, :n_new, :]  # shape: (N_atoms, n_new, 2*l+1)
    # Contract over the magnetic indices.
    B2_l = jnp.einsum('iam,ibm->iab', A_l, A_l)
    return B2_l  # shape: (N_atoms, n_new, n_new)

def compute_B2(A_dict, n_max=3):
    """
    Compute the second-body invariants for l >= 1.
    Returns a dict mapping l to its corresponding invariant.
    """
    B2 = {}
    for l in sorted(A_dict.keys()):
        if l == 0:
            continue  # Skip l=0 (handled by B1)
        B2_l = compute_B2_l(A_dict[l], n_max)
        B2[l] = B2_l
    return B2

###############################################################################
# 3. Third-body (3B) invariant:
###############################################################################
@jax.jit
def compute_B3_jit(A1, A2, A3, cg_tensor):
    """
    JIT-compiled contraction for the 3B invariant.
    Contracts over the magnetic indices using the provided Clebsch–Gordan tensor.
    """
    return jnp.einsum('iam,ibn,icp,mnp->iabc', A1, A2, A3, cg_tensor)

def compute_B3(A_dict, triplet, n_max,cg_cache):
    """
    Compute the three-body invariant for a given triplet of angular momenta (l1, l2, l3).
    For each angular momentum channel, we limit the radial dimension to n_max.
    Here, n_max is assumed to be a static hyperparameter.
    """
    l1, l2, l3 = triplet
    A1 = A_dict[l1]
    A2 = A_dict[l2]
    A3 = A_dict[l3]
    
    # Limit the radial dimension for each angular momentum channel using Python's min.
    n1 = min(A1.shape[1], n_max)
    n2 = min(A2.shape[1], n_max)
    n3 = min(A3.shape[1], n_max)
    
    A1 = A1[:, :n1, :]  # shape: (N_atoms, n1, 2*l1+1)
    A2 = A2[:, :n2, :]  # shape: (N_atoms, n2, 2*l2+1)
    A3 = A3[:, :n3, :]  # shape: (N_atoms, n3, 2*l3+1)
    t_=time.time()
    # Obtain the Clebsch–Gordan coupling tensor.
    cg_tensor = cg_cache[triplet]#cg_jax(l1, l2, l3)  # shape: (2*l1+1, 2*l2+1, 2*l3+1)
    #print("Time for CG tensor computation:",time.time()-t_)
    # Contract over the magnetic indices.
    B3 = compute_B3_jit(A1, A2, A3, cg_tensor)
    return B3  # shape: (N_atoms, n1, n2, n3)

###############################################################################
# 4. Full ACE descriptor:
###############################################################################
import time
def get_full_ace(positions, neighbor_list, n_rbf, l_max, x_max, n_max,cg_cache):
    """
    Compute the full ACE descriptor from positions and a neighbor list.
    
    For each l = 0, 1, ..., l_max, compute the atomic basis using compute_atomic_basis.
    Then compute B1 (from l=0), B2 invariants (for l>=1), and B3 invariants (for all triplets),
    and finally concatenate them into one descriptor per atom.
    
    Returns an array of shape (N_atoms, D), where D is the total descriptor dimension.
    """
    # Compute atomic basis for each l.
    A_dict = {}
    t1=time.time()
    for l in range(l_max + 1):
        A_dict[l] = compute_atomic_basis(positions, neighbor_list, n_rbf, l, x_max)
    t2=time.time()
    #print("Time for atomic basis computation:",t2-t1)
    # Compute B1 invariant (l=0).
    t3=time.time()
    B1 = compute_B1(A_dict[0])  # shape: (N_atoms, n_rbf)
    #print("Time for B1 computation:",time.time()-t3)
    # Compute B2 invariants for l >= 1 and flatten their extra dimensions.
    B2_dict = compute_B2(A_dict, n_max)
    B2_list = []
    t4=time.time()
    for l in sorted(B2_dict.keys()):
        B2_l = B2_dict[l]  # shape: (N_atoms, n_new, n_new)
        B2_list.append(B2_l.reshape(B2_l.shape[0], -1))
    if B2_list:
        B2_array = jnp.concatenate(B2_list, axis=1)
    else:
        B2_array = jnp.zeros((B1.shape[0], 0))
    #print("Time for B2 computation:",time.time()-t4)
    # Compute B3 invariants for all triplets (l1, l2, l3) with each l in [0, l_max].
    B3_list = []
    valid_ls = list(range(l_max + 1))
    t5=time.time()  
    for l1 in valid_ls:
        for l2 in valid_ls:
            for l3 in valid_ls:
                if abs(l1 - l2) <= l3 <= l1 + l2:
                    B3_val = compute_B3(A_dict, (l1, l2, l3), n_max,cg_cache)
                    B3_list.append(B3_val.reshape(B3_val.shape[0], -1))
    if B3_list:
        B3_array = jnp.concatenate(B3_list, axis=1)
    else:
        B3_array = jnp.zeros((B1.shape[0], 0))
    #print("Time for B3 computation:",time.time()-t5)
    # Concatenate B1, B2, and B3 along the feature axis.
    full_ace = jnp.concatenate([B1, B2_array, B3_array], axis=1)
    return full_ace




###############################################################################
# Example usage:
###############################################################################
if __name__ == "__main__":
    from neighborhood_mace_jax import get_neighborhood
    from itertools import product
    # For demonstration we fabricate a dummy atomic basis A_dict.
    # Suppose we have N_atoms=2, and for each l we have N_rbf=3 radial channels.
    N_atoms = 4
    n_rbf = 8
    A_dict = {}
    l_max = 2  # We include l=0,1,2.
    key = random.PRNGKey(0)
    for l in range(l_max+1): # l       # angular momentum; for l=1 we expect 2*1+1 = 3 angular components.
        # Generate random atomic basis for each l.      
        positions = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Define neighbor lists (here, every atom is neighbor to every other).
        '''neighbor_list = [
            jnp.array([1, 2, 3]), # Neighbors of atom 0
            jnp.array([0, 2, 3]), # Neighbors of atom 1
            jnp.array([0, 1, 3]),
            jnp.array([0, 1, 2])
        ]'''
        neighbor_list = get_neighborhood(positions, 5.0)
        
        hashable_neighbor_list = tuple(tuple(int(x) for x in arr) for arr in neighbor_list)
        x_max = 5.0  # cutoff
        
        A_dict[l] = compute_atomic_basis(jnp.array(positions), hashable_neighbor_list, n_rbf, l, x_max)
        print("Shape of atomic basis A:", A_dict[l].shape)
    
    # Compute B^(1)
    B1 = compute_B1(A_dict[0])
    print("B^(1) shape:", B1.shape)
    
    # Compute B^(2) for each l>0.
    B2 = compute_B2(A_dict,n_max=3)
    for l, B2_l in B2.items():
        print(f"B^(2) for l={l} shape:", B2_l.shape)
    
    # Compute B^(3) for a specific triplet, e.g. (l1,l2,l3)=(0,1,1) if valid.
    l_max=2
    
    import numpy as np
    # Load the precomputed CG tensors.
    loaded = np.load("cg_cache.npz", allow_pickle=True)
    # Convert the keys (which are strings) back to tuples if necessary.
    cg_cache = {eval(key): jnp.array(loaded[key]) for key in loaded.files}
    for l1, l2, l3 in product(range(l_max + 1), repeat=3):
        if not (abs(l1 - l2) <= l3 <= l1 + l2) or ((l1 + l2 + l3) % 2 != 0):
            continue
        triplet = (l1, l2, l3)
    # (Note: For l1=0, the A_dict[0] has only one m component.)
    #try:
        B3 = compute_B3(A_dict, triplet,n_max=2,cg_cache=cg_cache)
        print(f"B^(3) for triplet {triplet} shape:", B3.shape)
    #except Exception as e:
    #    print("B^(3) computation encountered an error:", e)
    
    # For B^(4) and B^(5), you would need to supply the appropriate coupling tensors.
    # Here we just print a message.
    print("B^(4) and B^(5) are not fully implemented in this example.")
