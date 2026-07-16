import argparse
import sympy as sp

from magsymtb.plot_energy_band.block_diagonalization import subroutine_eigen_problem_for_energy_band_plot


def main():
    sp.init_printing(use_unicode=False, wrap_line=False)

    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Run diagonalization for plotting energy bands."
    )

    # Required positional argument
    # parser.add_argument(
    #     "confFileName",
    #     type=str,
    #     help="Path to the input .conf file (e.g., /path/to/mc.conf)"
    # )

    # Optional arguments to replace the hardcoded values
    parser.add_argument(
        "-p", "--num_processes",
        type=int,
        default=12,
        help="Number of processes to use (default: 12)"
    )

    parser.add_argument(
        "-n", "--interpolate_point_num",
        type=int,
        default=200,
        help="Number of interpolation points (default: 200)"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_false",
        dest="verbose",
        help="Disable verbose output (default is verbose=True)"
    )

    # Parse the arguments passed from the CLI
    args = parser.parse_args()

    # Run the diagonalization subroutine using the parsed arguments
    out_pickle_file_name = subroutine_eigen_problem_for_energy_band_plot(
        args.num_processes,
        args.interpolate_point_num,
        args.verbose
    )

    print(f"out_pickle_file_name={out_pickle_file_name}")

# This ensures the script can still be run directly using `python run_diagonalization_band_plotting.py ...`
if __name__ == "__main__":
    main()