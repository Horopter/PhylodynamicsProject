"""
Project configuration file.

Defines device-independent paths and project settings.
All scripts should import from this module to ensure consistent paths.

Author: Santosh Desai <santoshdesai12@hotmail.com>
"""

import os
import sys
from pathlib import Path

# Get the project root directory (parent of this file's directory)
# This file should be at the root of the project
PROJECT_ROOT = Path(__file__).parent.resolve()


def get_project_root():
    """
    Get the project root directory.
    
    Returns:
    --------
    Path : Path to the project root directory
    """
    return PROJECT_ROOT


def get_project_root():
    """
    Get the project root directory.
    
    Returns:
    --------
    Path : Path to the project root directory
    """
    return PROJECT_ROOT


def setup_project_paths(script_file: str = None):
    """
    Set up Python path to include project root and necessary directories.
    
    This function should be called at the start of any script that needs
    to import project modules. It adds the project root and analysis directory
    to sys.path.
    
    Parameters:
    -----------
    script_file : str, optional
        Path to the script file (usually __file__). If provided, will also
        add the analysis directory to the path for cross-module imports.
    
    Returns:
    --------
    Path : Path to the project root
    """
    # Always add project root
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    
    # If script_file is provided and it's in the analysis directory,
    # add analysis parent
    if script_file:
        script_path = Path(script_file)
        if 'analysis' in script_path.parts:
            analysis_parent = script_path.parent.parent
            if str(analysis_parent) not in sys.path:
                sys.path.insert(0, str(analysis_parent))
    
    return PROJECT_ROOT

# Data directories (input data)
DATA_DIR = PROJECT_ROOT / "data"
PHYLODYNAMICSDL_DIR = DATA_DIR / "phylodynamicsDL"
TREES_DIR = PHYLODYNAMICSDL_DIR / "output_trees"
PARAMS_FILE = PHYLODYNAMICSDL_DIR / "all_params.csv"
TREE_ENCODINGS_FILE = PHYLODYNAMICSDL_DIR / "tree_encodings.csv"

# Output directories (output data)
OUTPUT_DIR = PROJECT_ROOT / "output"
MLE_OUTPUT_DIR = OUTPUT_DIR / "mle_results"
PHYLODEEP_OUTPUT_DIR = OUTPUT_DIR / "parameter_estimates"
BAYESIAN_OUTPUT_DIR = OUTPUT_DIR / "bayesian_results"
COMPARISON_OUTPUT_DIR = OUTPUT_DIR / "comparison_results"

# Default parameters
DEFAULT_SAMPLING_PROBA = 0.5
MIN_TIP_SIZE = 50  # Minimum tip size for PhyloDeep

# Create output directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
MLE_OUTPUT_DIR.mkdir(exist_ok=True)
PHYLODEEP_OUTPUT_DIR.mkdir(exist_ok=True)
BAYESIAN_OUTPUT_DIR.mkdir(exist_ok=True)
COMPARISON_OUTPUT_DIR.mkdir(exist_ok=True)


def get_tree_file(tree_idx):
    """
    Get the path to a tree file by index.
    
    Parameters:
    -----------
    tree_idx : int
        Tree index
    
    Returns:
    --------
    Path : Path to the tree file
    """
    return TREES_DIR / f"tree_{tree_idx}.nwk"


def get_params_file(tree_idx):
    """
    Get the path to a parameters file by index.
    
    Parameters:
    -----------
    tree_idx : int
        Tree index
    
    Returns:
    --------
    Path : Path to the parameters file
    """
    return TREES_DIR / f"params_{tree_idx}.csv"


def verify_paths():
    """
    Verify that required paths exist.
    
    Returns:
    --------
    dict : Dictionary with verification results
    """
    results = {
        "project_root": PROJECT_ROOT.exists(),
        "phylodynamicsdl_dir": PHYLODYNAMICSDL_DIR.exists(),
        "trees_dir": TREES_DIR.exists(),
        "params_file": PARAMS_FILE.exists(),
    }
    
    if TREES_DIR.exists():
        nwk_files = list(TREES_DIR.glob("*.nwk"))
        csv_files = list(TREES_DIR.glob("*.csv"))
        results["num_tree_files"] = len(nwk_files)
        results["num_param_files"] = len(csv_files)
    
    return results


if __name__ == "__main__":
    # Print configuration when run directly
    print("=" * 80)
    print("Project Configuration")
    print("=" * 80)
    print(f"\nPROJECT_ROOT: {PROJECT_ROOT}")
    print(f"\nInput Data Directories:")
    print(f"  DATA_DIR: {DATA_DIR}")
    print(f"  PHYLODYNAMICSDL_DIR: {PHYLODYNAMICSDL_DIR}")
    print(f"  TREES_DIR: {TREES_DIR}")
    print(f"  PARAMS_FILE: {PARAMS_FILE}")
    print(f"\nOutput Data Directories:")
    print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"  MLE_OUTPUT_DIR: {MLE_OUTPUT_DIR}")
    print(f"  PHYLODEEP_OUTPUT_DIR: {PHYLODEEP_OUTPUT_DIR}")
    print(f"  BAYESIAN_OUTPUT_DIR: {BAYESIAN_OUTPUT_DIR}")
    print(f"  COMPARISON_OUTPUT_DIR: {COMPARISON_OUTPUT_DIR}")
    print(f"\nDefault Parameters:")
    print(f"  DEFAULT_SAMPLING_PROBA: {DEFAULT_SAMPLING_PROBA}")
    print(f"  MIN_TIP_SIZE: {MIN_TIP_SIZE}")
    
    print(f"\n{'=' * 80}")
    print("Path Verification")
    print("=" * 80)
    verification = verify_paths()
    for key, value in verification.items():
        status = "[OK]" if value else "[FAIL]"
        print(f"  {status} {key}: {value}")

