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

sp.init_printing(use_unicode=False, wrap_line=False)

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