import numpy as np
import copy
from pathlib import Path
import pickle

from magsymtb.name_conventions import orbital_map, representations_all_file_name

def build_orbital_vectors(parsed_config):
    """
    The orbital vector is a binary array where 1 indicates an active orbital
    and 0 indicates an inactive orbital. This represents which orbitals are
    included in the tight-binding basis for each atom.

    Args:
        parsed_config: Dictionary containing atom types and their orbitals

    Returns: Dictionary mapping atom position names to their orbital vectors (78-dim binary arrays)
    """
    atom_orbital_vectors = {}
    for atom in parsed_config['Wyckoff_positions']:
        position_name = atom["position_name"]
        orbitals = atom['orbitals']

        orbital_vector = np.zeros(78)
        for orbital in orbitals:
            if orbital in orbital_map:
                orbital_vector[orbital_map[orbital]] = 1
            else:
                print(f"Warning: Orbital '{orbital}' for atom '{position_name}' not recognized")

        atom_orbital_vectors[position_name] = orbital_vector

    return atom_orbital_vectors



def run_complete_orbitals(parsed_config):
    """
    Main entry point for completing orbitals based on symmetry constraints.
    Replaces the old stdin/stdout JSON script.
    """
    conf_file_dir = Path(parsed_config.get('config_file_path', 'N/A')).parent
    representations_all_file_name_path = conf_file_dir / representations_all_file_name

    with open(representations_all_file_name_path, "rb") as fptr:
        magnetic_space_group_representations = pickle.load(fptr)

    repr_s, repr_p, repr_d, repr_f = magnetic_space_group_representations["repr_s_p_d_f"]

    repr_s_np = np.array(repr_s)
    repr_p_np = np.array(repr_p)
    repr_d_np = np.array(repr_d)
    repr_f_np = np.array(repr_f)

    num_operations, _, _ = repr_s_np.shape

    # ==============================================================================
    # Build combined orbital representation matrix
    # ==============================================================================
    # IndSPDF: Array defining the dimensionality of each orbital shell
    # Structure: [1s, 2s, 2p, 3s, 3p, 3d, 4s, 4p, 4d, 4f, ...]
    # Values: dimension of each shell (s=1, p=3, d=5, f=7)
    orbital_nums_spdf = np.array([
        1,           # 1s
        1, 3,        # 2s, 2p
        1, 3, 5,     # 3s, 3p, 3d
        1, 3, 5, 7,  # 4s, 4p, 4d, 4f
        1, 3, 5, 7,  # 5s, 5p, 5d, 5f
        1, 3, 5, 7,  # 6s, 6p, 6d, 6f
        1, 3, 5, 7,  # 7s, 7p, 7d, 7f
    ])

    orbital_max_dim = np.sum(orbital_nums_spdf)
    spdf_combined = np.zeros((num_operations, orbital_max_dim, orbital_max_dim))

    # Fill the diagonal blocks of spdf_combined
    for j in range(num_operations):
        current_idx = 0
        # Iterate through orbital_nums_spdf to place each block
        for block_size in orbital_nums_spdf:
            if block_size == 1:
                spdf_combined[j, current_idx:current_idx + 1, current_idx:current_idx + 1] = repr_s_np[j]
            elif block_size == 3:
                spdf_combined[j, current_idx:current_idx + 3, current_idx:current_idx + 3] = repr_p_np[j]
            elif block_size == 5:
                spdf_combined[j, current_idx:current_idx + 5, current_idx:current_idx + 5] = repr_d_np[j]
            elif block_size == 7:
                spdf_combined[j, current_idx:current_idx + 7, current_idx:current_idx + 7] = repr_f_np[j]
            current_idx += block_size

    # Find which orbitals are coupled by symmetry
    non_zero_spdf_combined = np.sum(np.abs(spdf_combined), axis=0) > 1e-6

    # Build initial orbital vectors from user input
    # Create orbital vectors for each atom based on user-specified orbitals
    atom_orbital_vectors = build_orbital_vectors(parsed_config)# Binary vectors with 1 for active orbitals

    # ==============================================================================
    # Complete orbital sets using symmetry coupling
    # ==============================================================================
    # Update atom_orbital_vectors based on symmetry coupling
    # If user specifies orbital A, and symmetry couples A to B, then B must also be included
    updated_atom_orbital_vectors = {}
    added_orbitals_dict = {}# Dictionary to store which orbitals were added for each atom

    for atom_name, orbital_vector in atom_orbital_vectors.items():
        # Start with a copy of the original vector
        updated_vector = copy.deepcopy(orbital_vector)
        # Find indices where the orbital vector has 1 (active orbitals)
        active_orbital_indices = np.where(orbital_vector == 1)[0]
        # For each active orbital
        for orbital_idx in active_orbital_indices:
            # Find all orbitals coupled to this one by symmetry
            # Look at column orbital_idx in non_zero_spdf_combined
            coupled_orbital_indices = np.where(non_zero_spdf_combined[:, orbital_idx])[0]
            # Set all coupled positions to 1 (add them to the basis)
            updated_vector[coupled_orbital_indices] = 1

        updated_atom_orbital_vectors[atom_name] = updated_vector

        # Report which orbitals were added
        added_indices = np.where((updated_vector == 1) & (orbital_vector == 0))[0]
        if len(added_indices) > 0:
            # Get orbital names for the added indices
            added_orbitals = [k for k, v in orbital_map.items() if v in added_indices]
            added_orbitals_dict[atom_name] = added_orbitals
        else:
            added_orbitals_dict[atom_name] = []
    # Replace the original vectors with updated (completed) ones
    atom_orbital_vectors = updated_atom_orbital_vectors

    # ==============================================================================
    # Extract symmetry representations for active orbitals only
    # ==============================================================================
    # For each atom, extract the submatrices of symmetry representations
    # that act only on its active orbitals (reduces dimension from 78x78 to n×n)
    repr_on_active_orbitals = {}

    for atom_name, orbital_vector in atom_orbital_vectors.items():
        # Find indices of active orbitals for this atom
        active_indices = np.where(orbital_vector == 1)[0]

        if len(active_indices) > 0:
            # Create index arrays for extracting submatrices
            # idx will select rows and columns corresponding to active orbitals
            idx = np.ix_(active_indices, active_indices)
            # Extract the symmetry matrices for just these orbitals
            repr_matrices_for_atom = []
            for sym_op in range(num_operations):
                # Extract the submatrix from spdf_combined for this symmetry operation
                # This gives the representation acting on this atom's orbital subspace
                submatrix = spdf_combined[sym_op][idx]
                repr_matrices_for_atom.append(submatrix)
            repr_on_active_orbitals[atom_name] = np.array(repr_matrices_for_atom)
        else:
            repr_on_active_orbitals[atom_name] = np.array([])

    # ==============================================================================
    # Package results
    # ==============================================================================
    output_data = {
        # Updated orbital vectors (after symmetry completion)
        "updated_orbital_vectors": {name: vec.tolist() for name, vec in atom_orbital_vectors.items()},

        # List of orbitals that were added by symmetry for each atom
        "added_orbitals": added_orbitals_dict,

        # Symmetry representation matrices acting on each atom's active orbital subspace
        "representations_on_active_orbitals": {name: matrices.tolist() for name, matrices in repr_on_active_orbitals.items()}
    }

    return output_data