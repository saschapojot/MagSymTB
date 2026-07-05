import subprocess
import sys
import json
import numpy as np
from datetime import datetime
from copy import deepcopy
from pathlib import Path
import sympy as sp
import pickle
import base64
import scipy.linalg
import argparse

from magsymtb.name_conventions import orbital_map, processed_input_pkl_file_name, type_linear, type_hermitian, H_latex_file_name, \
    H_html_file_name, H_pkl_file_name, hopping_parameters_template_file_name,representations_all_file_name,relations_file_name

from magsymtb.parse_files.parse_conf import parse_configuration
from magsymtb.parse_files.sanity_check import run_sanity_check
from magsymtb.symmetry.generate_magnetic_space_group_representations import generate_representations
from magsymtb.symmetry.complete_orbitals import run_complete_orbitals

sp.init_printing(use_unicode=False, wrap_line=False)

err_representation=12
err_orbital_completion=13
save_err_code = 30
# this script computes for magnetic space group system

def load_and_validate_config(confFileName):
    """
    Parses the configuration, prints the summary, and runs sanity checks.
    Args:
        confFileName: conf file

    Returns:

    """
    # 1. Parse configuration file directly
    parsed_config = parse_configuration(confFileName)
    # 2. Display parsed configuration
    print("=" * 60)
    print("COMPLETE PARSED CONFIGURATION")
    print("=" * 60)
    print(f"{'System Name:':<25} {parsed_config.get('name', 'N/A')}")
    print(f"{'Config File:':<25} {parsed_config.get('config_file_path', 'N/A')}")
    print(f"{'Dimension:':<25} {parsed_config.get('dim', 'N/A')}")

    directions_to_study = parsed_config.get('directions_to_study')
    directions_str = ", ".join(directions_to_study) if directions_to_study else "None"
    print(f"{'Directions to study:':<25} {directions_str}")
    print(f"{'Spin considered:':<25} {parsed_config.get('spin', 'N/A')}")
    print(f"{'Truncation Radius:':<25} {parsed_config.get('truncation_radius', 'N/A')}")


    print("-" * 60)
    print(f"{'Space Group number:':<25} {parsed_config.get('space_group', 'N/A')}")
    print(f"{'H-M Name:':<25} {parsed_config.get('space_group_name_H_M', 'N/A')}")
    print(f"{'Cell Setting:':<25} {parsed_config.get('cell_setting', 'N/A')}")

    origin = parsed_config.get('space_group_origin')
    origin_str = f"{origin}" if origin else "N/A"
    print(f"{'Space Group Origin:':<25} {origin_str}")

    print("-" * 60)
    print("Magnetic Space Group Information:")
    print(f"{'BNS Number:':<25} {parsed_config.get('space_group_magn_number_BNS', 'N/A')}")
    print(f"{'UNI Name:':<25} {parsed_config.get('space_group_magn_name_UNI', 'N/A')}")
    print(f"{'BNS Name:':<25} {parsed_config.get('space_group_magn_name_BNS', 'N/A')}")
    print(f"{'OG Number:':<25} {parsed_config.get('space_group_magn_number_OG', 'N/A')}")
    print(f"{'OG Name:':<25} {parsed_config.get('space_group_magn_name_OG', 'N/A')}")
    print(f"{'Litvin PG Number:':<25} {parsed_config.get('point_group_number_Litvin', 'N/A')}")
    print(f"{'UNI PG Name:':<25} {parsed_config.get('point_group_name_UNI', 'N/A')}")

    print("\nLattice Basis Vectors:")
    basis = parsed_config.get('lattice_basis')
    if basis and isinstance(basis, list):
        for i, vec in enumerate(basis):
            print(f"  Vector {i + 1}: {vec}")

    print("-" * 60)
    print(f"{'Wyckoff position number:':<25} {parsed_config.get('Wyckoff_position_num', 'N/A')}")
    print("\nAtom/Orbital Definitions:")
    atom_types = parsed_config.get('Wyckoff_position_types', {})
    if atom_types:
        for atom, data in atom_types.items():
            orbitals = data.get('orbitals', []) if isinstance(data, dict) else data
            print(f"  {atom:<5} : {', '.join(orbitals)}")

    print("\nAtom Position Coefficients:")
    wyckoff_positions = parsed_config.get('Wyckoff_positions', [])
    if wyckoff_positions:
        wyckoff_positions.sort(key=lambda x: x.get('position_name', ''))
        for pos in wyckoff_positions:
            position_name = pos.get('position_name', 'Unknown')
            coords = pos.get('fractional_coordinates')
            if coords:
                print(f"  {position_name:<5} : [{coords[0]}, {coords[1]}, {coords[2]}]")
    print("=" * 60)
    # 3. Run sanity checks
    print("\n" + "=" * 60)
    print("RUNNING SANITY CHECK")
    print("=" * 60)
    run_sanity_check(parsed_config)
    print("Sanity check passed!")

    return parsed_config

def run_general_computation(confFileName):
    """
    execution of computation process
    Args:
        confFileName: conf file name

    Returns:

    """
    # 1. Load, print, and validate the configuration
    parsed_config = load_and_validate_config(confFileName)

    conf_file_dir = Path(parsed_config.get('config_file_path', 'N/A')).parent
    directions_to_study = parsed_config.get('directions_to_study')

    #2. Generate magnetic space group representations
    print("\n" + "=" * 60)
    print("COMPUTING SPACE GROUP REPRESENTATIONS")
    print("=" * 60)
    try:
        # Call the function directly instead of using subprocess
        magnetic_space_group_representations = generate_representations(parsed_config)
        print("Space group representations generated successfully!")

        # You can verify it loaded correctly
        representations_all_file_name_path = conf_file_dir / representations_all_file_name
        print(f"Successfully saved and loaded representations at {representations_all_file_name_path}")

    except Exception as e:
        print("Space group representations generation failed!")
        print(f"Error: {e}")
        sys.exit(err_representation)

    # 3. Complete Orbitals Under Symmetry
    print("\n" + "=" * 60)
    print("COMPLETING ORBITALS UNDER SYMMETRY")
    print("=" * 60)
    try:
        # Call the function directly
        orbital_completion_data = run_complete_orbitals(parsed_config)
        print("Orbital completion successful!")

        # Display which orbitals were added by symmetry
        print("\n" + "-" * 40)
        print("ORBITALS ADDED BY SYMMETRY:")
        print("-" * 40)
        added_orbitals = orbital_completion_data["added_orbitals"]
        if any(added_orbitals.values()):
            for atom_type, orbitals in added_orbitals.items():
                if orbitals:
                    print(f"  {atom_type}: {', '.join(orbitals)}")
        else:
            print("  No additional orbitals needed - input was already complete")

        # Display final active orbitals for each atom
        print("\n" + "-" * 40)
        print("FINAL ACTIVE ORBITALS PER ATOM:")
        print("-" * 40)
        updated_vectors = orbital_completion_data["updated_orbital_vectors"]
        orbital_map_reverse = {v: k for k, v in orbital_map.items()}  # Reverse lookup

        for atom_type, vector in updated_vectors.items():
            # Find indices where orbital is active (value = 1)
            active_indices = [i for i, val in enumerate(vector) if val == 1]
            # Convert indices back to orbital names
            active_orbital_names = [orbital_map_reverse.get(idx, f"unknown_{idx}") for idx in active_indices]
            print(f"  {atom_type} ({len(active_orbital_names)} orbitals): {', '.join(active_orbital_names)}")

        # Display symmetry representation information
        print("\n" + "-" * 40)
        print("SYMMETRY REPRESENTATIONS ON ACTIVE ORBITALS:")
        print("-" * 40)
        representations = orbital_completion_data["representations_on_active_orbitals"]
        for atom_type, repr_matrices in representations.items():
            if repr_matrices:
                repr_array = np.array(repr_matrices)
                print(f"  {atom_type}: {repr_array.shape[0]} operations, {repr_array.shape[1]}×{repr_array.shape[2]} matrices")

        # Update parsed_config with completed orbitals
        for atom_pos in parsed_config['Wyckoff_positions']:
            position_name = atom_pos['position_name']
            print(f"Updating orbitals for position_name={position_name}")
            # Get the updated orbital vector for this atom
            if position_name in updated_vectors:
                vector = updated_vectors[position_name]
                active_indices = [i for i, val in enumerate(vector) if val == 1]
                active_orbital_names = [orbital_map_reverse.get(idx) for idx in active_indices]

                # Update the specific position entry
                atom_pos['orbitals'] = active_orbital_names
                # Update the Wyckoff_position_types dictionary for this position_name
                parsed_config['Wyckoff_position_types'][position_name] = {'orbitals': active_orbital_names}

        # Store completion results for later use
        orbital_completion_results = {
            "status": "completed",
            "added_orbitals": added_orbitals,
            "orbital_vectors": updated_vectors,
            "representations_on_active_orbitals": representations,
        }

        print("\n" + "=" * 60)
        print("ORBITAL COMPLETION FINISHED")
        print("=" * 60)

    except Exception as e:
        print("Orbital completion failed!")
        print(f"Error: {e}")
        sys.exit(err_orbital_completion)

    # 4.  Save preprocessing data to pickle file
    print("\n" + "=" * 80)
    print("SAVING PREPROCESSING DATA")
    print("=" * 80)
    # Prepare comprehensive preprocessing data package
    origin_cart = np.array([0, 0, 0])  # origin for .cif file
    repr_s, repr_p, repr_d, repr_f = magnetic_space_group_representations["repr_s_p_d_f"]
    repr_s_np = np.array(repr_s)
    repr_p_np = np.array(repr_p)
    repr_d_np = np.array(repr_d)
    repr_f_np = np.array(repr_f)
    magnetic_space_group_matrices_spatial_cartesian = np.array(magnetic_space_group_representations["magnetic_space_group_matrices_spatial_cartesian"])

    # magnetic space group are represented in 3 parts, magnetic_space_group_cart_spatial is the spatial part
    # spinor_mat_representation is the spinor part,
    # delta_vec indicates time reversal
    # spatial part
    magnetic_space_group_cart_spatial = [np.array(item) for item in magnetic_space_group_matrices_spatial_cartesian]
    print(f"directions_to_study={directions_to_study}")
    search_dim = parsed_config['dim'] # =len(directions_to_study)

    # spinor part
    spinor_mat_representation = np.array(magnetic_space_group_representations["spinor_mat_representation"])
    # indicating time reversal
    delta_vec = np.array(magnetic_space_group_representations["delta_vec"])

    preprocessing_data = {
        # Core configuration
        'parsed_config': parsed_config,
        "name": parsed_config["name"],
        # magnetic Space group representations
        'magnetic_space_group_representations': magnetic_space_group_representations,
        'directions_to_study': directions_to_study,
        "dim": search_dim,
        # NumPy arrays for efficient computation
        'magnetic_space_group_cart_spatial': magnetic_space_group_cart_spatial,  # List of np.ndarray
        "spinor_mat_representation": spinor_mat_representation,
        "delta_vec": delta_vec,
        'origin_cart': origin_cart,  # np.ndarray (3,)
        # Orbital representation matrices
        'repr_s_np': repr_s_np,  # np.ndarray (num_ops, 1, 1)
        'repr_p_np': repr_p_np,  # np.ndarray (num_ops, 3, 3)
        'repr_d_np': repr_d_np,  # np.ndarray (num_ops, 5, 5)
        'repr_f_np': repr_f_np,  # np.ndarray (num_ops, 7, 7)
        # Orbital completion results
        'orbital_completion_results': orbital_completion_results,
        # Orbital mapping dictionary
        'orbital_map': orbital_map,
        # Metadata
        'creation_date': datetime.now().isoformat(),
        'script_version': '1.0',
        'description': 'Preprocessing data for tight-binding model construction'
    }

    # Determine output file path
    preprocessed_pickle_file = str(conf_file_dir / processed_input_pkl_file_name)

    # Save to pickle file
    try:
        with open(preprocessed_pickle_file, 'wb') as f:
            pickle.dump(preprocessing_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Calculate file size
        file_size = Path(preprocessed_pickle_file).stat().st_size
        if file_size < 1024:
            size_str = f"{file_size} bytes"
        elif file_size < 1024 ** 2:
            size_str = f"{file_size / 1024:.2f} KB"
        else:
            size_str = f"{file_size / (1024 ** 2):.2f} MB"

        print(f"✓ Preprocessing data saved successfully!")
        print(f"  File: {preprocessed_pickle_file}")
        print(f"  Size: {size_str}")
        print(f"\nSaved data includes:")
        print(f"  - parsed_config: Configuration dictionary")
        print(f"  - magnetic_space_group_representations: Full representation data")
        print(f"  - magnetic_space_group_cart_spatial: {len(magnetic_space_group_cart_spatial)} operations")
        print(f"  - origin_cart: magnetic Space group origin")
        print(f"  - repr_s/p/d/f_np: Orbital representation matrices")
        print(f"  - orbital_completion_results: Symmetry-completed orbitals")
        print(f"  - orbital_map: 78-dimensional orbital mapping")

    except Exception as e:
        print(f"✗ Failed to save preprocessing data!")
        print(f"  Error: {e}")
        sys.exit(save_err_code)

    print("=" * 80)

def main():

    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Run general magnetic symmetry tight-binding computation."
    )

    # Add the positional argument for the configuration file
    parser.add_argument(
        "conf_file",
        type=str,
        help="Path to the input .conf file"
    )

    # Parse the arguments from the terminal
    args = parser.parse_args()

    # Pass the file path to your main computation function
    run_general_computation(args.conf_file)

if __name__ == "__main__":
    main()