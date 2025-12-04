# Original Tree Generation Code

This directory contains the original tree generation code from the [PhylodynamicsDL repository](https://github.com/mcanearm/PhylodynamicsDL) that was used to generate the trees in `data/phylodynamicsDL/output_trees/`.

## Files

- **`generate_trees.py`** - Original Python script for generating birth-death trees
  - Uses `generate_bd` binary (from treesimulator package)
- Generates 20,000 trees with random birth-death parameters
- Parameters: λ and ψ uniformly distributed [0.0, 1.0]
- Tree size: 10-500 tips
- Sampling probability: 0.5
- Parallel execution with 8 workers

- **`generate_trees.R`** - Alternative R implementation
  - Uses `phylopomp` R package
  - Moran model simulation approach

- **`ORIGINAL_README.md`** - Original repository README documenting the project goals

## Current Implementation

The refactored version with device-independent paths and improved structure is located at:
- `../generate_trees.py` - Current implementation

## Usage

These files are preserved for:
- **Historical reference** - Documenting how the data was originally generated
- **Reproducibility** - Understanding the exact parameters and process used
- **Reference** - Comparing original vs. refactored implementations

## Repository Reference

Source: [PhylodynamicsDL](https://github.com/mcanearm/PhylodynamicsDL)

