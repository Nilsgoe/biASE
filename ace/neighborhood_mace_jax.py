from typing import Optional, Tuple

import matscipy.neighbours
import numpy as np


def get_neighborhood(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,  # [3, 3]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pbc is None:
        pbc = (False, False, False)

    if cell is None or np.all(cell == 0.0):
        cell = np.identity(3, dtype=float)

    assert len(pbc) == 3 and all(isinstance(i, (bool, np.bool_)) for i in pbc)
    assert cell.shape == (3, 3)

    # Note (mario): I swapped senders and receivers here
    # j = senders, i = receivers instead of the other way around
    # such that the receivers are always in the central cell.
    
    # This is important to propagate message passing towards the center which can be useful in some cases.
    receivers, senders, senders_unit_shifts = matscipy.neighbours.neighbour_list(
        quantities="ijS",
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=cutoff,
    )
    N = int(np.max(receivers)) +1 
    # Sort senders by receivers (ensures correct ordering without explicit looping)
    sort_indices = np.argsort(receivers)
    sorted_receivers = receivers[sort_indices]
    sorted_senders = senders[sort_indices]

    '''# Find the indices where each atom starts in the sorted array
    split_indices = np.where(np.diff(sorted_receivers, prepend=-1, append=N))[0]

    # Use jnp.split to efficiently partition the sender list into groups per atom
    neighbor_list = np.split(sorted_senders, split_indices[:-1])'''
     # Find the start indices for each unique receiver atom
    unique_atoms, start_indices = np.unique(sorted_receivers, return_index=True)

    # Append the total length to properly split at the last atom
    start_indices = np.append(start_indices, len(senders))

    # Use `jnp.split` to efficiently partition the sender list
    neighbor_list = [sorted_senders[start:end] for start, end in zip(start_indices[:-1], start_indices[1:])]
    
    #atm it gives it in form of a list there might be a better solution but it is okay for now
    return neighbor_list # senders, receivers, senders_unit_shifts

if __name__ == "__main__":
    positions = np.array([
            [0.0452, -2.02, 0.0452],
            [1.0145, 0.034, 0.0232],
            [0.0111, 1.041, -0.010],
             [-0.0111, -0.041, 0.510],
        ])
    print(get_neighborhood(positions, 5.0))