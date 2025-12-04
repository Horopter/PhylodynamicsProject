"""
Tree generation script for birth-death phylogenetic trees.

Generates trees using the generate_bd binary (must be in PATH or specified).
Uses device-independent paths from config.py.

This is the refactored version of the original implementation.
See original/ directory for the historical code from PhylodynamicsDL repository.
"""

import numpy as np
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup project paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import setup_project_paths, PROJECT_ROOT, TREES_DIR
setup_project_paths(__file__)

# Configuration
n_trees = 20000
max_workers = 8
timeout_seconds = 5

# Generate random birth-death parameters
vals = np.random.uniform(0.0, 1.0, size=(n_trees, 2)).round(3)

# Find generate_bd binary (check common locations or use from PATH)
generate_bd = None
possible_paths = [
    Path.home() / ".local" / "bin" / "generate_bd",
    Path("/usr/local/bin/generate_bd"),
    Path("/usr/bin/generate_bd"),
]

for path in possible_paths:
    if path.exists():
        generate_bd = path
        break

if generate_bd is None:
    # Try to find in PATH
    import shutil
    generate_bd_path = shutil.which("generate_bd")
    if generate_bd_path:
        generate_bd = Path(generate_bd_path)
    else:
        raise FileNotFoundError(
            "generate_bd binary not found. Please install it or add to PATH."
        )


def generate_tree(i, la, psi, out_dir=None, timeout_seconds=5):
    if out_dir is None:
        out_dir = TREES_DIR
    out_dir.mkdir(exist_ok=True, parents=True)
    tree_path = out_dir / f"tree_{i}.nwk"
    log_path = out_dir / f"params_{i}.csv"

    if tree_path.exists():
        return None

    try:
        subprocess.run(
            [
                str(generate_bd),
                "--min_tips",
                "10",
                "--max_tips",
                "500",
                "--la",
                str(la),
                "--psi",
                str(psi),
                "--p",
                str(0.5),
                "--nwk",
                tree_path,
                "--log",
                log_path,
            ],
            check=True,
            timeout=timeout_seconds,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return tree_path, log_path
    except subprocess.TimeoutExpired:
        # print(f"Timeout encountered! Skipping tree {i}.")
        return None


if __name__ == "__main__":
    print(f"Generating {n_trees} trees...")
    print(f"Output directory: {TREES_DIR}")
    print(f"Using generate_bd: {generate_bd}")

    output_paths = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for i, row in enumerate(vals):
            if (TREES_DIR / f"tree_{i}.nwk").exists():
                continue
            else:
                la, psi = row
                futures.append(ex.submit(generate_tree, i, la, psi, out_dir=TREES_DIR))
    for fut in tqdm(
        as_completed(futures),
        total=len(futures),
        desc="Simulating Trees.",
        unit="trees",
    ):
        pass
