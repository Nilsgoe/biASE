import jax.numpy as jnp
from jax import random

# Assume that we have a JAX version of the Clebsch-Gordan function available:
from clebsch_gordan_jax import clebsch_gordan_jax as cg_jax
from get_atomic_basis import compute_atomic_basis
# (The function cg_jax(l1, l2, l3) returns an array of shape (2*l1+1, 2*l2+1, 2*l3+1)
# corresponding to the 3j–symbols (up to phase factors) or derived Clebsch–Gordan coefficients.)

###############################################################################
# Helper: mapping from m to index
###############################################################################
def m_index(m, l):
    # For a given angular momentum l and magnetic quantum number m (with -l <= m <= l),
    # the index in an array of length 2l+1 is given by:
    return int(m + l)

###############################################################################
# 1. First-body (1B) invariant:
###############################################################################
def compute_B1(A_dict):
    # A_dict is a dict: for each l we have A^{(l)} of shape (N_atoms, N_rbf, 2*l+1)
    # For l=0, there is only m=0.
    A0 = A_dict[0]  # shape (N, n_rbf, 1)
    # B^(1) is simply the m=0 component.
    B1 = A0[..., 0]  # shape (N, n_rbf)
    return B1

###############################################################################
# 2. Second-body (2B) invariant:
###############################################################################
def compute_B2(A_dict, n_max=3):
    """
    Compute B^(2) for each l using Einstein summation.
    
    For each l, A_dict[l] is assumed to have shape (N, n_rbf, 2*l+1) with the m axis
    ordered as m = -l, -l+1, …, l. The invariant is defined as:
    
         B^(2)_{i,n1,n2,l} = sum_{m=-l}^{l}  A^{(l)}_{i, n1, m} * A^{(l)}_{i, n2, m}
    
    This implementation multiplies the A^(l) array by the phase factor (-1)^m, then uses
    the reversed ordering (via jnp.flip) on the m-axis to obtain A^{(l)}_{i, n2, -m}.
    
    Args:
        A_dict (dict): Keys are l; each value is an array of shape (N, n_rbf, 2*l+1).
        n_max (int): Maximum number of radial basis functions to use.
        
    Returns:
        dict: Keys are l; each value is an array of shape (N, n_new, n_new) representing B^(2).
    """
    B2 = {}
    for l, A_l in A_dict.items():
        # Limit radial basis dimension
        n_rbf_original = A_l.shape[1]
        n_new = min(n_rbf_original, n_max)
        A_l = A_l[:, :n_new, :]  # shape (N, n_new, 2*l+1)
        
        # Create phase factors: (-1)^m for m in [-l, ..., l]
        m_vals = jnp.arange(-l, l+1)         # shape (2*l+1,)
        phase = (-1)**m_vals                 # shape (2*l+1,)
        phase = phase.reshape(1, 1, -1)        # shape (1, 1, 2*l+1)
 
        B2_l = jnp.einsum('iam,ibm->iab', A_l, A_l)
        B2[l] = B2_l
    return B2
 

###############################################################################
# 3. Third-body (3B) invariant:
###############################################################################
def compute_B3(A_dict, triplet,n_max):
    """
    Compute the three-body invariant for a given triplet of angular momenta.
    triplet is a tuple (l1, l2, l3). For each such triplet, we assume that the atomic
    basis A_dict has entries for l1, l2, and l3.
    
    The invariant is defined as:
       B^(3)_{i, n1, n2, n3}(l1,l2,l3) =
         sum_{m1, m2, m3} (3j-symbol)_{m1,m2,m3} *
           A^{(l1)}_{i, n1, m1} * A^{(l2)}_{i, n2, m2} * A^{(l3)}_{i, n3, m3}.
    """
    l1, l2, l3 = triplet
    A1 = A_dict[l1]  # shape (N, n1, 2*l1+1)
    A2 = A_dict[l2]  # shape (N, n2, 2*l2+1)
    A3 = A_dict[l3]  # shape (N, n3, 2*l3+1)
    
    # Limit the radial dimension to n_max (or the available number if less)
    n1 = min(A1.shape[1], n_max)
    n2 = min(A2.shape[1], n_max)
    n3 = min(A3.shape[1], n_max)
    A1 = A1[:, :n1, :]  # new shape (N, n1, 2*l1+1)
    A2 = A2[:, :n2, :]  # new shape (N, n2, 2*l2+1)
    A3 = A3[:, :n3, :]  # new shape (N, n3, 2*l3+1)
    # Obtain the coupling tensor (here assumed to be the real 3j-symbols)
    # cg_tensor has shape (2*l1+1, 2*l2+1, 2*l3+1)
    cg_tensor = cg_jax(l1, l2, l3)
    
    # Now contract over the magnetic indices:
    # Using einsum notation:
    #   B3[i, n1, n2, n3] = sum_{m1,m2,m3} A1[i, n1, m1] * A2[i, n2, m2] * A3[i, n3, m3] * cg_tensor[m1, m2, m3]
    #B3 = jnp.einsum('inm, jnm, knm, mnp -> ijk', A1, A2, A3, cg_tensor)
    # Because the atomic index should be common, we actually want:
    #B3 = jnp.einsum('aim, ajn, akm, mnp -> aijk', A1, A2, A3, cg_tensor)
    # Contract over the magnetic indices.
    B3 = jnp.einsum('iam,ibn,icp,mnp->iabc', A1, A2, A3, cg_tensor)
    # Here, a = atom index, i, j, k label the radial channels.
    return B3

###############################################################################
# 4 & 5. Fourth-body (4B) and Fifth-body (5B) invariants:
###############################################################################
# The definitions require additional coupling coefficients (denoted by square brackets)
# that we assume can be constructed from the Clebsch-Gordan coefficients (or using higher-order
# Wigner symbols). For brevity we sketch these as placeholders.
def compute_B4(A_dict, quadruplet, coupling_tensor_4):
    """
    Compute the four-body invariant.
    quadruplet: tuple (l1, l2, l3, l4)
    coupling_tensor_4: a precomputed tensor of shape (2*l1+1, 2*l2+1, 2*l3+1, 2*l4+1)
       representing the appropriate coupling.
    """
    l1, l2, l3, l4 = quadruplet
    A1 = A_dict[l1]  # shape (N, n1, 2*l1+1)
    A2 = A_dict[l2]
    A3 = A_dict[l3]
    A4 = A_dict[l4]
    B4 = jnp.einsum('aim, ajn, akp, alq, mnpq -> aijk', A1, A2, A3, A4, coupling_tensor_4)
    return B4

def compute_B5(A_dict, quintuplet, coupling_tensor_5):
    """
    Compute the five-body invariant.
    quintuplet: tuple (l1, l2, l3, l4, l5)
    coupling_tensor_5: precomputed tensor of shape (2*l1+1, 2*l2+1, 2*l3+1, 2*l4+1, 2*l5+1).
    """
    l1, l2, l3, l4, l5 = quintuplet
    A1 = A_dict[l1]
    A2 = A_dict[l2]
    A3 = A_dict[l3]
    A4 = A_dict[l4]
    A5 = A_dict[l5]
    B5 = jnp.einsum('aim, ajn, akp, alq, arm, mnpqr -> aijklm', A1, A2, A3, A4, A5, coupling_tensor_5)
    return B5

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
        
    
        x_max = 5.0  # cutoff
        
        A_dict[l] = compute_atomic_basis(positions, neighbor_list, n_rbf, l, x_max)
        print("Shape of atomic basis A:", A_dict[l].shape)
    
    # Compute B^(1)
    B1 = compute_B1(A_dict)
    print("B^(1) shape:", B1.shape)
    
    # Compute B^(2) for each l>0.
    B2 = compute_B2(A_dict,n_max=3)
    for l, B2_l in B2.items():
        print(f"B^(2) for l={l} shape:", B2_l.shape)
    
    # Compute B^(3) for a specific triplet, e.g. (l1,l2,l3)=(0,1,1) if valid.
    l_max=2
    for l1, l2, l3 in product(range(l_max + 1), repeat=3):
        if not (abs(l1 - l2) <= l3 <= l1 + l2) or ((l1 + l2 + l3) % 2 != 0):
            continue
        triplet = (l1, l2, l3)
    # (Note: For l1=0, the A_dict[0] has only one m component.)
    #try:
        B3 = compute_B3(A_dict, triplet,n_max=2)
        print(f"B^(3) for triplet {triplet} shape:", B3.shape)
    #except Exception as e:
    #    print("B^(3) computation encountered an error:", e)
    
    # For B^(4) and B^(5), you would need to supply the appropriate coupling tensors.
    # Here we just print a message.
    print("B^(4) and B^(5) are not fully implemented in this example.")
