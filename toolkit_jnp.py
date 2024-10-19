import numpy as np
import itertools
import jax.numpy as jnp
import copy
from jax import grad,jacfwd,jacrev
from functools import partial

TOL = 1E-12
MAX_IT = 100000    # in practice this is not exceeded


################################################################
###############################JNP##############################
################################################################



def get_dihedrals_jnp(atom_pos_list, indices, cell_input=None,mic=False,pbc_input=None):
    """Calculate dihedral angles.

    Calculate dihedral angles (in degrees) between the list of vectors
    a0->a1 and a2->a3, where a0, a1, a2 and a3 are in each row of indices.

    Use mic=True to use the Minimum Image Convention and calculate the
    angles across periodic boundaries.

    Get dihedral angles formed by three lists of vectors.

    Calculate dihedral angle (in degrees) between the vectors a0->a1,
    a1->a2 and a2->a3, written as v0, v1 and v2.

    Set a cell and pbc to enable minimum image
    convention, otherwise angles are taken as-is.
    """
    atom_pos_list = jnp.array(atom_pos_list)
    indices = jnp.array(indices)
    assert indices.shape[1] == 4

    a0s = atom_pos_list[indices[:, 0]]
    a1s = atom_pos_list[indices[:, 1]]
    a2s = atom_pos_list[indices[:, 2]]
    a3s = atom_pos_list[indices[:, 3]]

    # vectors 0->1, 1->2, 2->3
    v0 = a1s - a0s
    v1 = a2s - a1s
    v2 = a3s - a2s

    cell = None
    pbc = None

    if mic:
        if cell_input is None or pbc_input is None:
            raise ValueError("Either cell_input and/or pbc_input are None while using mic=True")
        cell = np.array(cell_input)
        pbc = pbc_input
        

    (v0, v1, v2), (_, nv1, _) = conditional_find_mic_jnp([v0, v1, v2], cell, pbc)

    v1n = v1 / nv1[:, jnp.newaxis]
    # v, w: projection of v0, v2 onto plane perpendicular to v1
    v = -v0 - jnp.einsum('ij,ij,ik->ik', -v0, v1n, v1n)
    w = v2 - jnp.einsum('ij,ij,ik->ik', v2, v1n, v1n)

    # formula returns 0 for undefined dihedrals; prefer ZeroDivisionError
    undefined_v = jnp.all(v == 0.0, axis=1)
    undefined_w = jnp.all(w == 0.0, axis=1)
    if jnp.any(jnp.array([undefined_v, undefined_w])):
        raise ZeroDivisionError('Undefined dihedral for planar inner angle')

    x = jnp.einsum('ij,ij->i', v, w)
    y = jnp.einsum('ij,ij->i', jnp.cross(v1n, v, axis=1), w)
    dihedrals = jnp.arctan2(y, x)            # dihedral angle in [-pi, pi]
    #dihedrals[dihedrals < 0.] += 2 * jnp.pi  # dihedral angle in [0, 2*pi]
    dihedrals = dihedrals.at[dihedrals < 0].add(2 * jnp.pi)
    return jnp.degrees(dihedrals)


def get_angles_jnp(atom_pos_list, indices, cell_input=None,mic=False,pbc_input=None):
        """Get angle formed by three atoms for multiple groupings.

        Calculate angle in degrees between vectors between atoms a2->a1
        and a2->a3, where a1, a2, and a3 are in each row of indices.

        Use mic=True to use the Minimum Image Convention and calculate
        the angle across periodic boundaries.

        Get angles formed by two lists of vectors.

        Calculate angle in degrees between vectors v0 and v1

        Set a cell and pbc to enable minimum image
        convention, otherwise angles are taken as-is.
        """
        
       
        atom_pos_list = jnp.array(atom_pos_list)
        indices = jnp.array(indices)
        assert indices.shape[1] == 3

        a1 = atom_pos_list[indices[:, 0]]
        a2 = atom_pos_list[indices[:, 1]]
        a3 = atom_pos_list[indices[:, 2]]
        v12 = a1 - a2
        v32 = a3 - a2

        cell = None
        pbc = None

        if mic:
            cell = np.array(cell_input)
            pbc = pbc_input

        (v12, v32), (nv12, nv32) = conditional_find_mic_jnp([v12, v32], cell, pbc)
        
        if (nv12 <= 0).any() or (nv32 <= 0).any():
            raise ZeroDivisionError('Undefined angle')
        v12n = v12 / nv12[:, jnp.newaxis]
        v32n = v32 / nv32[:, jnp.newaxis]
        # We just normalized the vectors, but in some cases we can get
        # bad things like 1+2e-16.  These we clip away:
        angles = jnp.arccos(jnp.einsum('ij,ij->i', v12n, v32n).clip(-1.0, 1.0))
        
        return jnp.degrees(angles)






def get_distances_jnp(p1, p2=None, cell=None, pbc=None):
    """Return distance matrix of every position in p1 with every position in p2

    If p2 is not set, it is assumed that distances between all positions in p1
    are desired. p2 will be set to p1 in this case.

    Use set cell and pbc to use the minimum image convention.
    """
    p1 = jnp.atleast_2d(p1)
    if p2 is None:
        np1 = len(p1)
        ind1, ind2 = jnp.triu_indices(np1, k=1)
        D = p1[ind2] - p1[ind1]
    else:
        p2 = jnp.atleast_2d(p2)
        D = (p2[jnp.newaxis, :, :] - p1[:, jnp.newaxis, :]).reshape((-1, 3))

    (D, ), (D_len, ) = conditional_find_mic_jnp([D], cell=cell, pbc=pbc)

    if p2 is None:
        Dout = jnp.zeros((np1, np1, 3))
        Dout =Dout.at[(ind1,ind2)].set(D)
        
        Dout -= jnp.transpose(Dout, axes=(1, 0, 2))

        Dout_len = jnp.zeros((np1, np1))
        Dout_len =Dout_len.at[(ind1,ind2)].set(D_len)
        
        Dout_len += Dout_len.T
        return Dout, Dout_len

    # Expand back to matrix indexing
    D = jnp.reshape(D, (-1, len(p2), 3))
    D_len = jnp.reshape(D_len, (-1, len(p2)))


    return D, D_len

def get_distances(p1, p2=None, cell=None, pbc=None):
    """Return distance matrix of every position in p1 with every position in p2

    If p2 is not set, it is assumed that distances between all positions in p1
    are desired. p2 will be set to p1 in this case.

    Use set cell and pbc to use the minimum image convention.
    """
    p1 = np.atleast_2d(p1)
    if p2 is None:
        np1 = len(p1)
        ind1, ind2 = np.triu_indices(np1, k=1)
        D = p1[ind2] - p1[ind1]
    else:
        p2 = np.atleast_2d(p2)
        D = (p2[np.newaxis, :, :] - p1[:, np.newaxis, :]).reshape((-1, 3))

    (D, ), (D_len, ) = conditional_find_mic([D], cell=cell, pbc=pbc)

    if p2 is None:
        Dout = np.zeros((np1, np1, 3))
        Dout[(ind1, ind2)] = D
       
        Dout -= np.transpose(Dout, axes=(1, 0, 2))

        Dout_len = np.zeros((np1, np1))
        Dout_len[(ind1, ind2)] = D_len
        
        Dout_len += Dout_len.T
        return Dout, Dout_len

    # Expand back to matrix indexing
    D.shape = (-1, len(p2), 3)
    D_len.shape = (-1, len(p2))

    return D, D_len

def get_distances_derivatives(v0, cell=None, pbc=None):
    """Get derivatives of distances for all vectors in v0 w.r.t. Cartesian
    coordinates in Angstrom.

    Set cell and pbc to use the minimum image convention.

    There is a singularity for distances -> 0 for which a ZeroDivisionError is
    raised.
    Derivative output format: [[dx_a0, dy_a0, dz_a0], [dx_a1, dy_a1, dz_a1]].
    """
    (v0, ), (dists, ) = conditional_find_mic([v0], cell, pbc)

    if (dists <= 0.).any():  # identify singularities
        raise ZeroDivisionError('Singularity for distance derivative')

    derivs_d0 = np.einsum('i,ij->ij', -1. / dists, v0)  # derivatives by atom 0
    derivs_d1 = -derivs_d0                              # derivatives by atom 1
    derivs = np.stack((derivs_d0, derivs_d1), axis=1)
    return derivs

def conditional_find_mic(vectors, cell, pbc):
    """Return list of vector arrays and corresponding list of vector lengths
    for a given list of vector arrays. The minimum image convention is applied
    if cell and pbc are set. Can be used like a simple version of get_distances.
    """
    if (cell is None) != (pbc is None):
        raise ValueError("cell or pbc must be both set or both be None")
    if cell is not None:
        mics = [find_mic_jnp(v, cell, pbc) for v in vectors]
        vectors, vector_lengths = zip(*mics)
    else:
        vector_lengths = np.linalg.norm(vectors, axis=2)
    return [np.asarray(v) for v in vectors], vector_lengths

def conditional_find_mic_jnp(vectors, cell, pbc):
    """Return list of vector arrays and corresponding list of vector lengths
    for a given list of vector arrays. The minimum image convention is applied
    if cell and pbc are set. Can be used like a simple version of get_distances.
    """
    vectors = jnp.asarray(vectors)
    if (cell is not None) != (pbc is not None):
        raise ValueError("cell or pbc must be both set or both be None")

    if cell is not None:
        mics = [find_mic_jnp(v, cell, pbc) for v in vectors]
        vectors, vector_lengths = zip(*mics)
        #print('Vectors: ',vectors, vector_lengths,'\n')

    else:
        vector_lengths = jnp.linalg.norm(vectors, axis=2)

    return [jnp.asarray(v) for v in vectors], vector_lengths



def find_mic_jnp(v, cell, pbc=True):
    """Finds the minimum-image representation of vector(s) v using either one
    of two find mic algorithms depending on the given cell, v and pbc."""

    cell = jnp.array(cell)
    pbc = cell.any(1) & pbc2pbc_jnp(pbc)
    #print('find: ',pbc)
    dim = jnp.sum(pbc)
    v = jnp.asarray(v)
    single = v.ndim == 1
    v = jnp.atleast_2d(v)

    if dim > 0:
        naive_find_mic_is_safe = False
        if dim == 3:
            vmin, vlen = naive_find_mic(v, cell)
            # naive find mic is safe only for the following condition
            cell_lengths=cell_lengths_jnp(cell)
            #print(cell_lengths,vlen)
            if (vlen < 0.5 * min(cell_lengths)).all():
                naive_find_mic_is_safe = True  # hence skip Minkowski reduction

        if not naive_find_mic_is_safe:
            raise NotImplementedError("A general mic is currentently not implemented please ensure a 3D pbc and a vlen < 0.5 * min(cell_lengths). This allows you to use the naive mic.")
            vmin, vlen = general_find_mic(v, cell, pbc=pbc)
            #print('Here: ',vmin,vlen)
    else:
        vmin = v.copy()
        vlen = jnp.linalg.norm(vmin, axis=1)

    if single:
        return vmin[0], vlen[0]
    else:
        return vmin, vlen


def pbc2pbc_jnp(pbc):
    newpbc = jnp.empty(3, bool)
    
    newpbc = newpbc.at[:].set(pbc)
    
    return jnp.asarray(newpbc)
def naive_find_mic(v, cell):
    """Finds the minimum-image representation of vector(s) v.
    Safe to use for (pbc.all() and (norm(v_mic) < 0.5 * min(cell.lengths()))).
    Can otherwise fail for non-orthorhombic cells.
    Described in:
    W. Smith, "The Minimum Image Convention in Non-Cubic MD Cells", 1989,
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.57.1696."""
    f = cell_scaled_positions_jnp(cell,v)
    f -= jnp.floor(f + 0.5)
    vmin = f @ cell
    vlen = jnp.linalg.norm(vmin, axis=1)
    return vmin, vlen
def cell_scaled_positions_jnp(cell,positions) -> jnp.ndarray:
        """Calculate scaled positions from Cartesian positions.

        The scaled positions are the positions given in the basis
        of the cell vectors.  For the purpose of defining the basis, cell
        vectors that are zero will be replaced by unit vectors as per
        :meth:`~ase.cell.Cell.complete`."""
        return jnp.linalg.solve(cell_complete_jnp(cell).T, jnp.transpose(positions)).T

def cell_scaled_positions_jnp(cell,positions) -> jnp.ndarray:
        """Calculate scaled positions from Cartesian positions.

        The scaled positions are the positions given in the basis
        of the cell vectors.  For the purpose of defining the basis, cell
        vectors that are zero will be replaced by unit vectors as per
        :meth:`~ase.cell.Cell.complete`."""
        return jnp.linalg.solve(cell_complete_jnp(cell).T, jnp.transpose(positions)).T


def cell_complete_jnp(cell):
    """Convert missing cell vectors into orthogonal unit vectors."""
    """Calculate complete cell with missing lattice vectors.

    Returns a new 3x3 ndarray.
    """

    
    cell = jnp.array(cell, dtype=float)
    missing = jnp.nonzero(~cell.any(axis=1))[0]

    if len(missing) == 3:
        cell.flat[::4] = 1.0
    if len(missing) == 2:
        # Must decide two vectors:
        V, s, WT = jnp.linalg.svd(cell.T)
        sf = [s[0], 1, 1]
        cell = (V @ jnp.diag(sf) @ WT).T
        if jnp.sign(jnp.linalg.det(cell)) < 0:
            cell[missing[0]] = -cell[missing[0]]
    elif len(missing) == 1:
        i = missing[0]
        cell[i] = jnp.cross(cell[i - 2], cell[i - 1])
        cell[i] /= jnp.linalg.norm(cell[i])

    return cell


def cell_lengths_jnp(cell):
        """Return the length of each lattice vector as an array."""
        return jnp.linalg.norm(cell, axis=1)



def cell_angles_jp(cell):
        """Return an array with the three angles alpha, beta, and gamma."""
        return cellpar_jp(cell)[3:].copy()

def cellpar_jp(cell, radians=False):
    """Returns the cell parameters [a, b, c, alpha, beta, gamma].

    Angles are in degrees unless radian=True is used.
    """
    lengths = [jnp.linalg.norm(v) for v in cell]
    angles = []
    for i in range(3):
        j = i - 1
        k = i - 2
        ll = lengths[j] * lengths[k]
        if ll > 1e-16:
            x = jnp.dot(cell[j], cell[k]) / ll
            angle = 180.0 / jnp.pi * jnp.arccos(x)
        else:
            angle = 90.0
        angles.append(angle)
    if radians:
        angles = [angle * jnp.pi / 180 for angle in angles]
    return jnp.array(lengths + angles)



def filter_non_zero_rows(array):
    # Convert the input array to a NumPy array
    np_array = np.array(array)

    # Reshape the array to a 2D shape
    flat_array = np_array.reshape(-1, np_array.shape[-1])

    # Find rows where all elements are zeros
    zero_rows = np.all(flat_array == 0, axis=1)

    # Filter out zero rows from the flattened array
    non_zero_rows = flat_array[~zero_rows]

    # Reshape the filtered array back to the original shape
    result_array = non_zero_rows.reshape(np_array.shape[:-2] + (-1, np_array.shape[-1]))

    return result_array


import jax.numpy as jnp

def minimum_image_convention_triclinic(positions, box_vectors):
    """
    Apply the minimum image convention for a non-orthogonal (triclinic) box.

    Parameters:
    - positions: (N, 3) array of particle positions in Cartesian coordinates.
    - box_vectors: (3, 3) array where each row is a lattice vector of the box.

    Returns:
    - (N, 3) array of positions after applying the minimum image convention.
    """
    # Construct the transformation matrix from box vectors
    H = box_vectors  # (3, 3)
    
    # Transform positions to fractional coordinates
    H_inv = jnp.linalg.inv(H)
    positions_frac = jnp.dot(positions, H_inv.T)  # (N, 3)
    
    # Apply the minimum image convention in fractional coordinates
    positions_frac = positions_frac - jnp.round(positions_frac)  # Wrap into [-0.5, 0.5]
    
    # Transform back to Cartesian coordinates
    positions_cart = jnp.dot(positions_frac, H.T)  # (N, 3)
    
    return positions_cart

def minimum_image_distance_triclinic(vec, box_vectors):
    """
    Apply the minimum image convention to a distance vector in a non-orthogonal (triclinic) box.
    
    Parameters:
    - vec: (3,) array representing the vector between two particles in Cartesian coordinates.
    - box_vectors: (3, 3) array where each row is a lattice vector of the box.
    
    Returns:
    - The minimum image distance vector in Cartesian coordinates.
    """
    # Construct the transformation matrix from box vectors
    H = box_vectors
    
    # Transform vector to fractional coordinates
    H_inv = jnp.linalg.inv(H)
    vec_frac = jnp.dot(vec, H_inv.T)
    
    # Apply the minimum image convention in fractional coordinates
    vec_frac = vec_frac - jnp.round(vec_frac)
    
    # Transform back to Cartesian coordinates
    vec_cart = jnp.dot(vec_frac, H.T)
    
    return vec_cart



from ase import Atoms
x=False
if x==True:
    # Create Water molecule
    water = Atoms('H2O', positions=[[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]])

    # Create H2O2 molecule
    h2o2 = Atoms('H2O2', positions=[[0.0, 0.0, 0.0],    # Oxygen (O1)
        [1.2, 0.0, 0.0],    # Hydrogen (H1)
        [0.0, 1.2, 0.0],    # Oxygen (O2)
        [0.0, 1.0, 1.2],])

    # Save positions to files in XYZ format
    h2o_pos=water.get_positions()
    h2o2_pos=h2o2.get_positions()
    h2o_ang=water.get_angles([[0,1,2]])
    h2o2_ang= h2o2.get_angles([[0,1,2],[3,1,2]])
    h202_dih = h2o2.get_dihedrals([[0,1,2,3],[3,2,0,1]])

    print('\n',h2o_ang,'\n','\n',h2o2_ang)
    print(get_angles_jnp(h2o_pos,[[0,1,2]]),'\n',get_angles_jnp(h2o2_pos,[[0,1,2],[3,1,2]]))

    print(h202_dih,get_dihedrals_jnp(h2o2_pos,[[0,1,2,3],[3,2,0,1]]))

from ase.build import fcc100,add_adsorbate
from ase.geometry import get_angles_derivatives, get_dihedrals_derivatives
y=False
if y==True:
    initial = fcc100('Pd', size=(5, 5, 1), vacuum=10.0)
    dimer = fcc100('Pd', size=(2, 1, 1), vacuum=10.0)
    add_adsorbate(initial, dimer, 1.611, position=(-2,2))

    initial_pos =initial.get_positions()

    initial_ang_mic= initial.get_angles([[7,8,26],[8,25,26]],mic=False)
    print('Angles ASE: ',initial_ang_mic)
    biase_angles_mic = get_angles_jnp(initial_pos,[[7,8,26],[8,25,26]],mic=False)
    print('Angles biASE: ',biase_angles_mic)

    initial_ang_mic= initial.get_angles([[7,8,26],[8,25,26]],mic=True)
    initial_dih = initial.get_dihedrals([[7,8,25,26],[5,8,25,26]],mic=True)
    initial_cell=initial.get_cell()
    initial_pbc=initial.get_pbc()
    print('Angles ASE with mic: ',initial_ang_mic)
    #print(initial_pbc)
    initial_pbc =[True,True,True]
    biase_angles_mic = get_angles_jnp(initial_pos,[[7,8,26],[8,25,26]],mic=True,pbc_input=initial_pbc,cell_input=initial_cell)
    print('Angles biASE with mic: ',biase_angles_mic)
    pbc=False
    if pbc==False:
        diff_angle_fwd=jacfwd(partial(get_angles_jnp,indices=[[7,8,26],[8,25,26]],mic=False))
        diff_angle_rev=jacrev(partial(get_angles_jnp,indices=[[7,8,26],[8,25,26]],mic=False))

        biase_fwd_angle_diff= diff_angle_fwd(initial_pos)
        biase_rev_angle_diff = diff_angle_rev(initial_pos)

        v0=[initial_pos[7]-initial_pos[8]]
        v1=[initial_pos[26]-initial_pos[8]]

        ase_angle_diff=[]
        no_pbc =[False, False,False]
        ase_angle_diff.append(get_angles_derivatives(v0,v1,pbc=no_pbc,cell=initial_cell))

        v0=[initial_pos[8]-initial_pos[25]]
        v1=[initial_pos[26]-initial_pos[25]]
    
        ase_angle_diff.append(get_angles_derivatives(v0,v1,pbc=no_pbc,cell=initial_cell))

        biase_fwd_angle_diff = filter_non_zero_rows(np.asarray(biase_fwd_angle_diff))
        biase_rev_angle_diff = filter_non_zero_rows(np.asarray(biase_rev_angle_diff))
        print('Diff Angles ASE: \n',np.asarray(ase_angle_diff))
        print('Diff Angles biASE: \nfwd: ',biase_fwd_angle_diff,'\nrev: ',biase_rev_angle_diff)

    diff_angle_fwd_pbc=jacfwd(partial(get_angles_jnp,indices=[[7,8,26],[8,25,26]],mic=True,pbc_input=initial_pbc,cell_input=initial_cell))
    diff_angle_rev_pbc=jacrev(partial(get_angles_jnp,indices=[[7,8,26],[8,25,26]],mic=True,pbc_input=initial_pbc,cell_input=initial_cell))

    biase_fwd_angle_diff_pbc = diff_angle_fwd_pbc(initial_pos)
    biase_rev_angle_diff_pbc = diff_angle_rev_pbc(initial_pos)

    

    v0=[initial_pos[7]-initial_pos[8]]
    v1=[initial_pos[26]-initial_pos[8]]

    ase_angle_diff_pbc=[]

    ase_angle_diff_pbc.append(get_angles_derivatives(v0,v1,pbc=initial_pbc,cell=initial_cell).tolist())

    v0=[initial_pos[8]-initial_pos[25]]
    v1=[initial_pos[26]-initial_pos[25]]
   
    ase_angle_diff_pbc.append(get_angles_derivatives(v0,v1,pbc=initial_pbc,cell=initial_cell).tolist())

    biase_fwd_angle_diff_pbc = filter_non_zero_rows(np.asarray(biase_fwd_angle_diff_pbc))
    biase_rev_angle_diff_pbc = filter_non_zero_rows(np.asarray(biase_rev_angle_diff_pbc))

    print('Diff Angles ASE with mic: \n',np.asarray(ase_angle_diff_pbc))
    print('Diff Angles biASE with mic: \nfwd: ',biase_fwd_angle_diff_pbc,'\nrev: ',biase_rev_angle_diff_pbc)


initial = fcc100('Pd', size=(2, 1, 1), vacuum=10.0)
dimer = fcc100('Pd', size=(1, 1, 1), vacuum=10.0)
add_adsorbate(initial, dimer, 1.611, position=(-2,2))

initial_pos =initial.get_positions()

print(get_distances(initial_pos[0],initial_pos[1]))

print(get_distances_jnp(initial_pos[0],initial_pos[1]))
print(get_distances(initial_pos))

print(get_distances_jnp(initial_pos))

diff_dis_fwd_pbc=jacfwd(partial(get_distances_jnp,initial_pos[0]))
diff_dis_rev_pbc=jacrev(partial(get_distances_jnp,initial_pos[0]))

print(diff_dis_fwd_pbc(initial_pos[1]))
diff_dis_fwd_pbc=jacfwd(partial(get_distances_jnp,initial_pos[1]))
diff_dis_rev_pbc=jacrev(partial(get_distances_jnp,initial_pos[1]))

print(diff_dis_fwd_pbc(initial_pos[0]))
v0=[initial_pos[1]-initial_pos[0]]
print(v0)
print(get_distances_derivatives(v0))
'''
initial = fcc100('Pd', size=(3, 3, 1), vacuum=10.0)
dimer = fcc100('Pd', size=(2, 1, 1), vacuum=10.0)
add_adsorbate(initial, dimer, 1.611, position=(2,2))

initial_pos =initial.get_positions()
initial_pos[-1][0] +=2
initial.set_positions(initial_pos)

initial_ang= initial.get_angles([[7,8,9],[8,9,10]],mic=True)
initial_dih = initial.get_dihedrals([[7,8,9,10],[5,8,9,10]],mic=True)
initial_cell=initial.get_cell()
initial_pbc=initial.get_pbc()
print(initial_ang,initial_dih)
print(np.asarray(initial_cell),initial_pbc)
from ase.geometry import is_orthorhombic
print(is_orthorhombic(initial_cell))
print(get_angles_jnp(initial_pos,[[7,8,9],[8,9,10]],mic=True,pbc_input=initial_pbc,cell_input=initial_cell),get_dihedrals_jnp(initial_pos,[[7,8,9,10],[5,8,9,10]],mic=True,pbc_input=initial_pbc,cell_input=initial_cell))
print(initial_pos[-1][0])
pbc=True
if pbc==False:
    diff_angle_fwd=jacfwd(partial(get_angles_jnp,indices=[[7,8,9],[8,9,10]],mic=False))
    diff_angle_rev=jacrev(partial(get_angles_jnp,indices=[[7,8,9],[8,9,10]],mic=False))

    print(diff_angle_fwd(initial_pos))
    print(diff_angle_rev(initial_pos))

diff_angle_fwd_pbc=jacfwd(partial(get_angles_jnp,indices=[[7,8,9],[8,9,10]],mic=True,pbc_input=initial_pbc,cell_input=initial_cell))
diff_angle_rev_pbc=jacrev(partial(get_angles_jnp,indices=[[8,9,10],[7,8,9],[8,9,10]],mic=True,pbc_input=initial_pbc,cell_input=initial_cell))
print(diff_angle_fwd_pbc(initial_pos))
print(diff_angle_rev_pbc(initial_pos))


from ase.geometry import get_angles_derivatives, get_dihedrals_derivatives


v0=[initial_pos[7]-initial_pos[8]]
v1=[initial_pos[9]-initial_pos[8]]
print(v0,v1)
print(get_angles_derivatives(v0,v1,pbc=initial_pbc,cell=initial_cell))

v0=[initial_pos[8]-initial_pos[9]]
v1=[initial_pos[10]-initial_pos[9]]
print(v0,v1)
print(get_angles_derivatives(v0,v1,pbc=initial_pbc,cell=initial_cell))

print(get_dihedrals_jnp(initial_pos,[[0,1,9,10]],mic=True,pbc_input=initial_pbc,cell_input=initial_cell))
diff_dihedrals_fwd_pbc=jacfwd(partial(get_dihedrals_jnp,indices=[[0,1,9,10],[0,1,9,10]],mic=True,pbc_input=initial_pbc,cell_input=initial_cell))
v0=[initial_pos[1]-initial_pos[0]]
v1=[initial_pos[9]-initial_pos[1]]
v2=[initial_pos[10]-initial_pos[9]]
print(diff_dihedrals_fwd_pbc(initial_pos))
print(get_dihedrals_derivatives(v0,v1,v2,pbc=initial_pbc,cell=initial_cell))


initial = fcc100('Pd', size=(2, 2, 1), vacuum=10.0)
dimer = fcc100('Pd', size=(1, 1, 1), vacuum=10.0)
add_adsorbate(initial, dimer, 1.611, position=(2,2))
initial_cell=initial.get_cell()
initial_pbc=initial.get_pbc()
initial_pos =initial.get_positions()
diff_angle_fwd_pbc=jacfwd(partial(get_angles_jnp,indices=[[0,1,4],[0,1,4]],mic=True,pbc_input=initial_pbc,cell_input=initial_cell))
diff_angle_rev_pbc=jacrev(partial(get_angles_jnp,indices=[[0,1,4],[0,1,4]],mic=True,pbc_input=initial_pbc,cell_input=initial_cell))
print(diff_angle_fwd_pbc(initial_pos))
print(diff_angle_rev_pbc(initial_pos))
v0=[initial_pos[0]-initial_pos[1]]
v1=[initial_pos[4]-initial_pos[1]]
print(v0,v1)
print(get_angles_derivatives(v0,v1,pbc=initial_pbc,cell=initial_cell))'''
