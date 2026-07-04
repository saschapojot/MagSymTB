import argparse
import os
import sys

from magsymtb.parse_files.parse_cif import (
    subroutine_generate_conf_file,
    subroutine_generate_all_symmetry_transformation_matrices
)

def init_configuration(cif_file_name, tol=1e-3):
    """Core logic to process the CIF file."""
    if not os.path.exists(cif_file_name):
        print(f"Error: file not found: {cif_file_name}", file=sys.stderr)
        sys.exit(4)

    print(f"Processing CIF file: {cif_file_name}")

    # Call the functions directly instead of using subprocess
    subroutine_generate_conf_file(cif_file_name, tol)
    subroutine_generate_all_symmetry_transformation_matrices(cif_file_name)

    print("Initialization complete.")


def main():
    """CLI Entry point for tb-init"""
    parser = argparse.ArgumentParser(
        description="Parse a .cif file and generate a .conf file and symmetry matrices."
    )

    # Positional argument for the file
    parser.add_argument(
        "cif_file",
        type=str,
        help="Path to the input .cif file"
    )

    # Optional argument for tolerance
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="Numerical tolerance for parsing (default: 1e-3)"
    )

    args = parser.parse_args()

    # Execute the logic
    init_configuration(args.cif_file, args.tol)



if __name__ == "__main__":
    main()