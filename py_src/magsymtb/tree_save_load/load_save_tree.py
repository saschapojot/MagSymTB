"""
load_save_tree.py
-----------------
Module for saving and loading constraint tree structures (vertex objects)
for the MagSymTB tight-binding computational framework.
"""

import pickle
from pathlib import Path
from datetime import datetime
from magsymtb.classes.class_defs  import vertex, hopping, atomIndex


def save_tree_structures(roots_list, file_path, metadata=None):
    """
    Saves constraint tree structures (roots and descendants) to a pickle (.pkl) file.

    Args:
        roots_list (list): List of root vertex objects.
        file_path (str or Path): Target path for the output pickle file.
        metadata (dict, optional): System metadata (system name, config, parameters, etc.).

    Returns:
        str: Absolute path of the saved file.

    Raises:
        ValueError: If any element in roots_list is not a root vertex.
    """
    file_path = Path(file_path)
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate that every vertex in roots_list is a root node
    for idx, v in enumerate(roots_list):
        if not getattr(v, 'is_root', False):
            raise ValueError(
                f"Element at index {idx} in roots_list is not a root vertex: {v}"
            )

    data_package = {
        'roots': roots_list,
        'num_roots': len(roots_list),
        'saved_at': datetime.now().isoformat(),
        'metadata': metadata or {}
    }

    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data_package, f, protocol=pickle.HIGHEST_PROTOCOL)

        file_size = file_path.stat().st_size
        size_str = (
            f"{file_size / 1024:.2f} KB"
            if file_size < 1024**2
            else f"{file_size / (1024**2):.2f} MB"
        )

        print("✓ Tree structures saved successfully!")
        print(f"  File:  {file_path.resolve()}")
        print(f"  Roots: {len(roots_list)}")
        print(f"  Size:  {size_str}")
        return str(file_path.resolve())

    except Exception as e:
        print(f"✗ Failed to save tree structures to {file_path}!")
        print(f"  Error: {e}")
        raise e


def load_tree_structures(file_path):
    """
    Loads constraint tree structures from a pickle (.pkl) file.
    Args:
        file_path: Path to the saved tree structure pickle file.

    Returns:
        dict: Loaded dictionary containing:
            - 'roots': List of root vertex objects
            - 'num_roots': Number of root trees
            - 'saved_at': ISO timestamp string
            - 'metadata': Metadata dictionary

    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Tree structure pickle file not found: {file_path}")

    try:
        with open(file_path, 'rb') as f:
            data_package = pickle.load(f)

        roots = data_package.get('roots', [])
        num_roots = data_package.get('num_roots', len(roots))
        saved_at = data_package.get('saved_at', 'Unknown')

        print(f"✓ Tree structures loaded successfully from: {file_path.resolve()}")
        print(f"  Roots count: {num_roots}")
        print(f"  Saved at:    {saved_at}")
        return data_package

    except Exception as e:
        print(f"✗ Failed to load tree structures from {file_path}!")
        print(f"  Error: {e}")
        raise e