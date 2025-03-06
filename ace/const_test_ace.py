import jax.numpy as jnp
from jax import random

# Assume that we have a JAX version of the Clebsch-Gordan function available:
from clebsch_gordan_jax import clebsch_gordan_jax as cg_jax
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
def compute_B2(A_dict):
    # We compute B^(2) for each l > 0.
    B2 = {}  # will store results per l
    for l, A_l in A_dict.items():
        if l == 0:
            continue
        # A_l has shape (N, n_rbf, 2*l+1) and m runs from -l to l.
        # We want to compute, for each atom i and for each pair of radial indices (n1, n2):
        #   B^(2)_{i, n1, n2, l} = sum_{m=-l}^{l} (-1)^m * A^{(l)}_{i, n1, m} * A^{(l)}_{i, n2, -m}
        N, n_rbf, _ = A_l.shape
        # Build an array of phases for m values:
        m_vals = jnp.arange(-l, l+1)  # shape (2*l+1,)
        phase = (-1)**m_vals  # shape (2*l+1,)
        # Note: in A_l the index for m is m_index = m + l.
        # The index corresponding to -m is: (-m)+l = 2*l - (m + l) = 2*l - m_index.
        # We can perform the contraction with an explicit sum over the last axis.
        # For clarity, we use an explicit loop over the m index (l is small) but one can also vectorize.
        B2_l = jnp.zeros((N, n_rbf, n_rbf))
        for mi in range(2*l+1):
            # Corresponding index for -m:
            mi_neg = 2*l - mi
            # Update B2_l with contribution from this m:
            B2_l = B2_l + phase[mi] * (A_l[:, :, mi][:, :, None] * A_l[:, :, mi_neg][:, None, :])
            #B2_l = B2_l + phase[mi] * A_l[:, :, mi] * A_l[:, :, mi_neg]
        B2[l] = B2_l  # shape (N, n_rbf, n_rbf)
    return B2

###############################################################################
# 3. Third-body (3B) invariant:
###############################################################################
def compute_B3(A_dict, triplet):
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
    
    # Obtain the coupling tensor (here assumed to be the real 3j-symbols)
    # cg_tensor has shape (2*l1+1, 2*l2+1, 2*l3+1)
    cg_tensor = cg_jax(l1, l2, l3)
    
    # Now contract over the magnetic indices:
    # Using einsum notation:
    #   B3[i, n1, n2, n3] = sum_{m1,m2,m3} A1[i, n1, m1] * A2[i, n2, m2] * A3[i, n3, m3] * cg_tensor[m1, m2, m3]
    B3 = jnp.einsum('inm, jnm, knm, mnp -> ijk', A1, A2, A3, cg_tensor)
    # Because the atomic index should be common, we actually want:
    B3 = jnp.einsum('aim, ajn, akm, mnp -> aijk', A1, A2, A3, cg_tensor)
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
    # For demonstration we fabricate a dummy atomic basis A_dict.
    # Suppose we have N_atoms=2, and for each l we have N_rbf=3 radial channels.
    N_atoms = 2
    N_rbf = 3
    A_dict = {}
    max_l = 2  # We include l=0,1,2.
    key = random.PRNGKey(0)
    for l in range(max_l+1):
        # Create a random A^{(l)} of shape (N_atoms, N_rbf, 2*l+1)
        shape = (N_atoms, N_rbf, 2*l+1)
        A_dict[l] = random.normal(key, shape)
    
    # Compute B^(1)
    B1 = compute_B1(A_dict)
    print("B^(1) shape:", B1.shape)
    
    # Compute B^(2) for each l>0.
    B2 = compute_B2(A_dict)
    for l, B2_l in B2.items():
        print(f"B^(2) for l={l} shape:", B2_l.shape)
    
    # Compute B^(3) for a specific triplet, e.g. (l1,l2,l3)=(0,1,1) if valid.
    triplet = (0, 1, 1)
    # (Note: For l1=0, the A_dict[0] has only one m component.)
    try:
        B3 = compute_B3(A_dict, triplet)
        print(f"B^(3) for triplet {triplet} shape:", B3.shape)
    except Exception as e:
        print("B^(3) computation encountered an error:", e)
    
    # For B^(4) and B^(5), you would need to supply the appropriate coupling tensors.
    # Here we just print a message.
    print("B^(4) and B^(5) are not fully implemented in this example.")
