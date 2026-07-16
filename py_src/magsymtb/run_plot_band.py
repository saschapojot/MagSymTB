import sys
import argparse
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from magsymtb.name_conventions import plotting_band_data_pkl_file_name

def load_plotting_band_data():
    current_dir = Path.cwd()
    directory = current_dir
    data_for_plotting_file_name = str(directory / plotting_band_data_pkl_file_name)
    try:
        with open(data_for_plotting_file_name, 'rb') as f:
            data_for_plotting = pickle.load(f)
        print(f"Successfully loaded data from {data_for_plotting_file_name}")
        return data_for_plotting
    except FileNotFoundError:
        print(f"Error: File not found at {data_for_plotting_file_name}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the pickle file: {e}")
        return None


def main():
    sp.init_printing(use_unicode=False, wrap_line=False)

    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Plot energy bands from diagonalization data.")

    # parser.add_argument(
    #     "confFileName",
    #     type=str,
    #     help="Path to the input .conf file (e.g., /path/to/mc.conf)"
    # )

    # Added optional arguments for y-axis limits (previously hardcoded)
    parser.add_argument(
        "--ymin",
        type=float,
        default=-0.5,
        help="Minimum y-axis value for energy (default: -0.5)"
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=0.5,
        help="Maximum y-axis value for energy (default: 0.5)"
    )

    args = parser.parse_args()
    data_non_existent_err_code = 21

    # Load data
    data_for_plotting = load_plotting_band_data()
    if data_for_plotting is None:
        print("Error: Could not load data. Exiting.")
        sys.exit(data_non_existent_err_code)

    name = data_for_plotting["name"]
    all_distances = data_for_plotting["all_distances"]
    high_symmetry_indices = data_for_plotting["high_symmetry_indices"]
    high_symmetry_labels = data_for_plotting["high_symmetry_labels"]
    all_eigenvalues = data_for_plotting["all_eigenvalues"]

    # --- Plotting bands ---
    plt.figure(figsize=(6, 5))
    num_bands = all_eigenvalues.shape[1]

    for i in range(0, num_bands):
        plt.plot(all_distances, all_eigenvalues[:, i], color='blue', linewidth=1.5)

    for index in high_symmetry_indices:
        if index < len(all_distances):
            plt.axvline(x=all_distances[index], color='black', linestyle='--', linewidth=0.8)

    valid_indices = [i for i in high_symmetry_indices if i < len(all_distances)]
    tick_locations = [all_distances[i] for i in valid_indices]

    plt.xticks(tick_locations, high_symmetry_labels, fontsize=14)

    plt.xlim(all_distances[0], all_distances[-1])
    plt.ylim(args.ymin, args.ymax)

    plt.title(f"{name}", fontsize=24)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save the plot
    current_dir = Path.cwd()
    base_directory = current_dir
    out_pic_file_name = str(base_directory / "band.png")
    plt.savefig(out_pic_file_name, bbox_inches='tight')
    print(f"Band structure plot successfully saved to: {out_pic_file_name}")

if __name__ == "__main__":
    main()