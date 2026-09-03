import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, Circle
import matplotlib.lines as mlines
import matplotlib as mpl

from magsymtb.name_conventions import tree_pkl_file_name
from magsymtb.tree_save_load.load_save_tree import load_tree_structures


# This script plots:
# 1. A pure 2D system with ALL constraint trees on a single plot (all dotted blue lines).
# 2. A separate plot with ONLY the lattice, unit cell atoms, and truncation circles (no tree hopping edges).


# ==============================================================================
#  Helper Functions
# ==============================================================================

def load_tree_data(tree_pkl_path):
    """
    Loads the tree structure pickle file.
    """
    path = Path(tree_pkl_path)
    file_path = path / tree_pkl_file_name

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Tree pickle file not found at: '{file_path.resolve()}'"
        )
    print(f"Loading tree structures from: {file_path.resolve()}")
    return load_tree_structures(str(file_path))


def assign_line_types(root_vertex):
    """
    Traverses the constraint tree and assigns line_type to hoppings.
    """

    def _traverse_recursive(node, current_line_type):
        node.hopping.line_type = current_line_type
        for child in node.children:
            next_line_type = current_line_type
            if child.type == "hermitian":
                next_line_type = 1 - current_line_type
            _traverse_recursive(child, next_line_type)

    if root_vertex.is_root:
        _traverse_recursive(root_vertex, 0)


def get_lattice_vectors_from_tree(root_vertex):
    """
    Traverses a single constraint tree starting from 'root_vertex' to extract
    [n0, n1, n2] vectors for the atoms in every hopping of every node.
    """
    extracted_vectors = []

    def _traverse_recursive(node):
        hop = node.hopping
        to_cell = [hop.to_atom.n0, hop.to_atom.n1, hop.to_atom.n2]
        from_cell = [hop.from_atom.n0, hop.from_atom.n1, hop.from_atom.n2]
        extracted_vectors.append(to_cell)
        extracted_vectors.append(from_cell)
        for child in node.children:
            _traverse_recursive(child)

    _traverse_recursive(root_vertex)
    return extracted_vectors


def get_extreme_vectors(lattice_set):
    """
    Finds the vectors containing the maximum and minimum n0 and n1.
    """
    if not lattice_set:
        return {
            "n0_max": {"value": 0, "vector": (0, 0, 0)},
            "n0_min": {"value": 0, "vector": (0, 0, 0)},
            "n1_max": {"value": 0, "vector": (0, 0, 0)},
            "n1_min": {"value": 0, "vector": (0, 0, 0)},
        }

    vec_max_n0 = max(lattice_set, key=lambda v: v[0])
    vec_min_n0 = min(lattice_set, key=lambda v: v[0])
    vec_max_n1 = max(lattice_set, key=lambda v: v[1])
    vec_min_n1 = min(lattice_set, key=lambda v: v[1])

    return {
        "n0_max": {"value": vec_max_n0[0], "vector": vec_max_n0},
        "n0_min": {"value": vec_min_n0[0], "vector": vec_min_n0},
        "n1_max": {"value": vec_max_n1[1], "vector": vec_max_n1},
        "n1_min": {"value": vec_min_n1[1], "vector": vec_min_n1},
    }


def expand_vector_bounds(vector):
    """
    Adjusts a vector's elements: adds 1 if positive, subtracts 1 if negative.
    """
    adjusted = []
    for val in vector:
        if val > 0:
            adjusted.append(val + 1)
        elif val < 0:
            adjusted.append(val - 1)
        else:
            adjusted.append(val)
    return tuple(adjusted)


def get_real_coords(atom_idx_obj, a0, a1, a2):
    """
    Computes the real-space 3D coordinates (x, y, z) for an atomIndex object.
    """
    n0, n1, n2 = atom_idx_obj.n0, atom_idx_obj.n1, atom_idx_obj.n2
    f0, f1, f2 = atom_idx_obj.frac_coord
    pos_vec = (n0 + f0) * a0 + (n1 + f1) * a1 + (n2 + f2) * a2
    return pos_vec[0], pos_vec[1], pos_vec[2]


def draw_basis_vectors(ax, origin, a0, a1):
    """
    Draws the lattice basis vectors a0 and a1 starting from a specific origin point.
    """
    ox, oy = origin

    ax.add_patch(FancyArrowPatch(
        (ox, oy), (ox + a0[0], oy + a0[1]),
        arrowstyle='-|>', mutation_scale=15, color='black', linewidth=2,
        shrinkA=0, shrinkB=0, zorder=20
    ))
    ax.text(ox + a0[0], oy + a0[1], r"$\mathbf{a}_0$", fontsize=14, fontweight='bold', zorder=20)

    ax.add_patch(FancyArrowPatch(
        (ox, oy), (ox + a1[0], oy + a1[1]),
        arrowstyle='-|>', mutation_scale=15, color='black', linewidth=2,
        shrinkA=0, shrinkB=0, zorder=20
    ))
    ax.text(ox + a1[0], oy + a1[1], r"$\mathbf{a}_1$", fontsize=14, fontweight='bold', zorder=20)


def draw_self_hopping_loop(ax, atom_x, atom_y, atom_z, color, linestyle='dotted', radius=0.4):
    """
    Draws a circle with an arrow head on it to represent self-hopping.
    """
    offset_angle_deg = 45
    offset_angle_rad = np.deg2rad(offset_angle_deg)

    circle_center_x = atom_x + radius * np.cos(offset_angle_rad)
    circle_center_y = atom_y + radius * np.sin(offset_angle_rad)

    circle = Circle((circle_center_x, circle_center_y), radius,
                    color=color, fill=False, linestyle=linestyle, linewidth=1.5, zorder=12)
    ax.add_patch(circle)

    arrow_angle_rad = offset_angle_rad
    arrow_x = circle_center_x + radius * np.cos(arrow_angle_rad)
    arrow_y = circle_center_y + radius * np.sin(arrow_angle_rad)

    tangent_angle = arrow_angle_rad + np.pi / 2
    dx = np.cos(tangent_angle) * 0.001
    dy = np.sin(tangent_angle) * 0.001

    arrow = FancyArrowPatch((arrow_x, arrow_y), (arrow_x + dx, arrow_y + dy),
                            arrowstyle='-|>',
                            mutation_scale=10,
                            color=color,
                            zorder=12)
    ax.add_patch(arrow)


def draw_tree_arrows(root_vertex, ax, a0, a1, a2, tree_color='blue', tolerance=1e-5):
    """
    Draws hopping arrows for a given tree root.
    ALL edges in this tree use the specified 'tree_color' and DOTTED line style.
    """

    def _traverse_draw(node):
        hop = node.hopping
        start_x, start_y, start_z = get_real_coords(hop.from_atom, a0, a1, a2)
        end_x, end_y, end_z = get_real_coords(hop.to_atom, a0, a1, a2)

        # All edges are dotted
        arrow_style = 'dotted'

        # Draw Hopping using tree_color with dotted line style
        if abs(start_x - end_x) < tolerance and abs(start_y - end_y) < tolerance and abs(start_z - end_z) < tolerance:
            draw_self_hopping_loop(ax, start_x, start_y, start_z, tree_color, arrow_style)
        else:
            arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                                    arrowstyle='-|>',
                                    mutation_scale=18,
                                    color=tree_color,
                                    linestyle=arrow_style,
                                    linewidth=1.8,
                                    zorder=12)
            ax.add_patch(arrow)

        for child in node.children:
            _traverse_draw(child)

    _traverse_draw(root_vertex)


# ==============================================================================
#  Plotting Functions
# ==============================================================================

def plot_all_trees(all_roots_sorted, parsed_config, unit_cell_atoms, output_dir, output_dir_svg, grid_params,
                   tree_color='blue'):
    """
    Plots ALL trees on a single picture, using the same uniform color (e.g., blue) and dotted lines for all edges.
    """
    n0_range = grid_params['n0_range']
    n1_range = grid_params['n1_range']
    n0_min, n0_max = grid_params['n0_min'], grid_params['n0_max']
    n1_min, n1_max = grid_params['n1_min'], grid_params['n1_max']

    lattice_basis = np.array(parsed_config["lattice_basis"])
    a0, a1, a2 = lattice_basis
    truncation_radius = parsed_config["truncation_radius"]

    # Create Single Canvas
    fig, ax = plt.subplots(figsize=(14, 14))

    # --- 1. Draw Grid Lines ---
    grid_segments = []

    for n0_idx in n0_range:
        p_start = n0_idx * a0 + n1_min * a1
        p_end = n0_idx * a0 + n1_max * a1
        grid_segments.append([p_start[:2], p_end[:2]])

    for n1_idx in n1_range:
        p_start = n0_min * a0 + n1_idx * a1
        p_end = n0_max * a0 + n1_idx * a1
        grid_segments.append([p_start[:2], p_end[:2]])

    lc = LineCollection(grid_segments, colors='grey', linewidths=1.0, alpha=0.6)
    ax.add_collection(lc)

    # --- 2. Highlight Unit Cell [0,0] ---
    c0 = 0 * a0 + 0 * a1
    c1 = 1 * a0 + 0 * a1
    c2 = 1 * a0 + 1 * a1
    c3 = 0 * a0 + 1 * a1
    highlight_segments = [
        [c0[:2], c1[:2]], [c1[:2], c2[:2]],
        [c2[:2], c3[:2]], [c3[:2], c0[:2]]
    ]
    lc_highlight = LineCollection(highlight_segments, colors='black',
                                  linewidths=2.5, alpha=1.0, zorder=5)
    ax.add_collection(lc_highlight)

    # --- 3. Draw Truncation Circles for ALL Atoms in Unit Cell [0,0] ---
    for atom in unit_cell_atoms:
        f0, f1, f2 = atom.frac_coord
        pos = (0 + f0) * a0 + (0 + f1) * a1 + f2 * a2
        cx, cy = pos[0], pos[1]

        # Draw Pink Dashed Truncation Radius Circle & Center Point
        circle = Circle((cx, cy), truncation_radius, color='pink', fill=False,
                        linestyle='--', linewidth=3, zorder=8)
        ax.add_patch(circle)
        ax.scatter([cx], [cy], c='pink', s=5, zorder=15)

    # --- 4. Draw Atoms ---
    unique_position_names = set(atom.position_name for atom in unit_cell_atoms)
    sorted_position_names = sorted(list(unique_position_names))
    num_unique_positions = len(sorted_position_names)
    hsv_colors = plt.cm.hsv(np.linspace(0, 1, num_unique_positions, endpoint=False))
    name_to_color = {name: hsv_colors[i] for i, name in enumerate(sorted_position_names)}

    xs, ys, plot_colors = [], [], []

    for n0 in n0_range[:-1]:
        for n1 in n1_range[:-1]:
            for atom in unit_cell_atoms:
                f0, f1, f2 = atom.frac_coord
                pos = (n0 + f0) * a0 + (n1 + f1) * a1 + f2 * a2
                xs.append(pos[0])
                ys.append(pos[1])
                plot_colors.append(name_to_color[atom.position_name])

    ax.scatter(xs, ys, c=plot_colors, s=40, edgecolors='black', zorder=10)

    # Add Legend for Wyckoff positions
    legend_elements = []
    for name, color in name_to_color.items():
        legend_elements.append(mlines.Line2D([], [], color=color, marker='o',
                                             linestyle='None', markersize=10, label=name,
                                             markeredgecolor='black'))
    ax.legend(handles=legend_elements, loc='upper right', title="Wyckoff Pos")

    # --- 5. Draw ALL Trees with the SAME color and DOTTED lines ---
    num_trees = len(all_roots_sorted)
    for root_vertex in all_roots_sorted:
        draw_tree_arrows(
            root_vertex=root_vertex,
            ax=ax,
            a0=a0,
            a1=a1,
            a2=a2,
            tree_color=tree_color
        )

    # --- 6. Axes Limits, Labels & Basis Vectors ---
    corner1 = n0_min * a0 + n1_min * a1
    corner2 = n0_max * a0 + n1_min * a1
    corner3 = n0_max * a0 + n1_max * a1
    corner4 = n0_min * a0 + n1_max * a1
    all_x = [corner1[0], corner2[0], corner3[0], corner4[0]]
    all_y = [corner1[1], corner2[1], corner3[1], corner4[1]]

    padding = 0.5
    ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
    ax.set_ylim(min(all_y) - padding, max(all_y) + padding)
    ax.set_aspect('equal')
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    basis_origin = (corner1[0], corner1[1])
    draw_basis_vectors(ax, basis_origin, a0, a1)

    ax.set_title(f"Combined Constraint Trees ({num_trees} Trees)", fontsize=16, pad=15)

    # Save Output Files
    filename_png = "lattice_grid_all_trees.png"
    filename_svg = "lattice_grid_all_trees.svg"
    output_file_png = os.path.join(output_dir, filename_png)
    output_file_svg = os.path.join(output_dir_svg, filename_svg)

    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_file_svg, bbox_inches='tight')
    print(f"Combined plot saved to:\n  - PNG: {output_file_png}\n  - SVG: {output_file_svg}")
    plt.close(fig)


def plot_lattice_only(parsed_config, unit_cell_atoms, output_dir, output_dir_svg, grid_params):
    """
    Plots ONLY the lattice, atoms, unit cell highlight, basis vectors, and truncation circles for each atom in [0,0].
    No hopping edges or arrows are drawn.
    """
    n0_range = grid_params['n0_range']
    n1_range = grid_params['n1_range']
    n0_min, n0_max = grid_params['n0_min'], grid_params['n0_max']
    n1_min, n1_max = grid_params['n1_min'], grid_params['n1_max']

    lattice_basis = np.array(parsed_config["lattice_basis"])
    a0, a1, a2 = lattice_basis
    truncation_radius = parsed_config["truncation_radius"]

    # Create Single Canvas
    fig, ax = plt.subplots(figsize=(14, 14))

    # --- 1. Draw Grid Lines ---
    grid_segments = []

    for n0_idx in n0_range:
        p_start = n0_idx * a0 + n1_min * a1
        p_end = n0_idx * a0 + n1_max * a1
        grid_segments.append([p_start[:2], p_end[:2]])

    for n1_idx in n1_range:
        p_start = n0_min * a0 + n1_idx * a1
        p_end = n0_max * a0 + n1_idx * a1
        grid_segments.append([p_start[:2], p_end[:2]])

    lc = LineCollection(grid_segments, colors='grey', linewidths=1.0, alpha=0.6)
    ax.add_collection(lc)

    # --- 2. Highlight Unit Cell [0,0] ---
    c0 = 0 * a0 + 0 * a1
    c1 = 1 * a0 + 0 * a1
    c2 = 1 * a0 + 1 * a1
    c3 = 0 * a0 + 1 * a1
    highlight_segments = [
        [c0[:2], c1[:2]], [c1[:2], c2[:2]],
        [c2[:2], c3[:2]], [c3[:2], c0[:2]]
    ]
    lc_highlight = LineCollection(highlight_segments, colors='black',
                                  linewidths=2.5, alpha=1.0, zorder=5)
    ax.add_collection(lc_highlight)

    # --- 3. Draw Truncation Circles for ALL Atoms in Unit Cell [0,0] ---
    for atom in unit_cell_atoms:
        f0, f1, f2 = atom.frac_coord
        pos = (0 + f0) * a0 + (0 + f1) * a1 + f2 * a2
        cx, cy = pos[0], pos[1]

        # Draw Pink Dashed Truncation Radius Circle & Center Point
        circle = Circle((cx, cy), truncation_radius, color='pink', fill=False,
                        linestyle='--', linewidth=3, zorder=8)
        ax.add_patch(circle)
        ax.scatter([cx], [cy], c='pink', s=5, zorder=15)

    # --- 4. Draw Atoms ---
    unique_position_names = set(atom.position_name for atom in unit_cell_atoms)
    sorted_position_names = sorted(list(unique_position_names))
    num_unique_positions = len(sorted_position_names)
    hsv_colors = plt.cm.hsv(np.linspace(0, 1, num_unique_positions, endpoint=False))
    name_to_color = {name: hsv_colors[i] for i, name in enumerate(sorted_position_names)}

    xs, ys, plot_colors = [], [], []

    for n0 in n0_range[:-1]:
        for n1 in n1_range[:-1]:
            for atom in unit_cell_atoms:
                f0, f1, f2 = atom.frac_coord
                pos = (n0 + f0) * a0 + (n1 + f1) * a1 + f2 * a2
                xs.append(pos[0])
                ys.append(pos[1])
                plot_colors.append(name_to_color[atom.position_name])

    ax.scatter(xs, ys, c=plot_colors, s=40, edgecolors='black', zorder=10)

    # Add Legend for Wyckoff positions
    legend_elements = []
    for name, color in name_to_color.items():
        legend_elements.append(mlines.Line2D([], [], color=color, marker='o',
                                             linestyle='None', markersize=10, label=name,
                                             markeredgecolor='black'))
    ax.legend(handles=legend_elements, loc='upper right', title="Wyckoff Pos")

    # --- 5. Axes Limits, Labels & Basis Vectors ---
    corner1 = n0_min * a0 + n1_min * a1
    corner2 = n0_max * a0 + n1_min * a1
    corner3 = n0_max * a0 + n1_max * a1
    corner4 = n0_min * a0 + n1_max * a1
    all_x = [corner1[0], corner2[0], corner3[0], corner4[0]]
    all_y = [corner1[1], corner2[1], corner3[1], corner4[1]]

    padding = 0.5
    ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
    ax.set_ylim(min(all_y) - padding, max(all_y) + padding)
    ax.set_aspect('equal')
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    basis_origin = (corner1[0], corner1[1])
    draw_basis_vectors(ax, basis_origin, a0, a1)

    ax.set_title("Lattice Grid & Truncation Circles (No Hopping Edges)", fontsize=16, pad=15)

    # Save Output Files
    filename_png = "lattice_grid_only.png"
    filename_svg = "lattice_grid_only.svg"
    output_file_png = os.path.join(output_dir, filename_png)
    output_file_svg = os.path.join(output_dir_svg, filename_svg)

    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_file_svg, bbox_inches='tight')
    print(f"Lattice-only plot saved to:\n  - PNG: {output_file_png}\n  - SVG: {output_file_svg}")
    plt.close(fig)


# ==============================================================================
#  Main Execution
# ==============================================================================

def main():
    # 1. Load tree data
    tree_pkl_path = "./"
    tree_data_package = load_tree_data(tree_pkl_path)
    all_roots_sorted = tree_data_package["roots"]
    metadata = tree_data_package["metadata"]

    # 2. Pre-process all trees to assign line types
    for root in all_roots_sorted:
        assign_line_types(root)

    # 3. CALCULATE GLOBAL GRID RANGES
    lattices_all = set()
    for one_root in all_roots_sorted:
        extracted_vectors = get_lattice_vectors_from_tree(one_root)
        for vec in extracted_vectors:
            lattices_all.add(tuple(vec))

    targets = get_extreme_vectors(lattices_all)
    print("-" * 50)
    print(f"Largest positive n0:  {targets['n0_max']['value']} | Vector: {targets['n0_max']['vector']}")
    print(f"Smallest negative n0: {targets['n0_min']['value']} | Vector: {targets['n0_min']['vector']}")
    print("-" * 50)
    print(f"Largest positive n1:  {targets['n1_max']['value']} | Vector: {targets['n1_max']['vector']}")
    print(f"Smallest negative n1: {targets['n1_min']['value']} | Vector: {targets['n1_min']['vector']}")
    print("-" * 50)

    vec_n0_max_expanded = expand_vector_bounds(targets['n0_max']['vector'])
    vec_n0_min_expanded = expand_vector_bounds(targets['n0_min']['vector'])
    vec_n1_max_expanded = expand_vector_bounds(targets['n1_max']['vector'])
    vec_n1_min_expanded = expand_vector_bounds(targets['n1_min']['vector'])

    max_n0_val = vec_n0_max_expanded[0]
    min_n0_val = vec_n0_min_expanded[0]
    max_n1_val = vec_n1_max_expanded[1]
    min_n1_val = vec_n1_min_expanded[1]

    grid_params = {
        'n0_range': list(range(min_n0_val, max_n0_val + 1)),
        'n1_range': list(range(min_n1_val, max_n1_val + 1)),
        'n0_min': min_n0_val,
        'n0_max': max_n0_val,
        'n1_min': min_n1_val,
        'n1_max': max_n1_val
    }
    print("\n" + "=" * 50)
    print("GLOBAL GRID GENERATION RANGES")
    print("=" * 50)
    print(f"n0 Range ({min_n0_val} to {max_n0_val}): {grid_params['n0_range']}")
    print(f"n1 Range ({min_n1_val} to {max_n1_val}): {grid_params['n1_range']}")
    print("=" * 50)

    # 4. Setup Output Directory
    parsed_config = metadata["parsed_config"]
    config_file_path = parsed_config["config_file_path"]
    config_dir = Path(config_file_path).parent
    output_dir = str(config_dir) + "/tree_visualization_2d/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_dir_svg = output_dir + "/svg"
    Path(output_dir_svg).mkdir(parents=True, exist_ok=True)
    unit_cell_atoms = metadata["unit_cell_atoms"]

    # 5. Generate Combined Tree Plot
    print(f"\nGenerating combined plot for all {len(all_roots_sorted)} trees (all in blue, dotted lines)...")
    plot_all_trees(
        all_roots_sorted=all_roots_sorted,
        parsed_config=parsed_config,
        unit_cell_atoms=unit_cell_atoms,
        output_dir=output_dir,
        output_dir_svg=output_dir_svg,
        grid_params=grid_params,
        tree_color='blue'
    )

    # 6. Generate Lattice-Only Plot (No hopping edges)
    print("\nGenerating lattice-only plot (no hopping edges)...")
    plot_lattice_only(
        parsed_config=parsed_config,
        unit_cell_atoms=unit_cell_atoms,
        output_dir=output_dir,
        output_dir_svg=output_dir_svg,
        grid_params=grid_params
    )

    print("\nDone.")


if __name__ == "__main__":
    main()