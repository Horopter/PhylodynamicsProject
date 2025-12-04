# PhyloDeep vs Maximum Likelihood Estimation: Statistical Efficiency Comparison

A comprehensive comparison of deep learning (PhyloDeep) and classical statistical methods (MLE, Bayesian MCMC) for estimating birth-death parameters from phylogenetic trees.

## Overview

This project evaluates the **statistical efficiency** of PhyloDeep models compared to Maximum Likelihood Estimation (MLE) for birth-death phylogenetic models. We compare three methods:

1. **Maximum Likelihood Estimation (MLE)** - Classical method using Stadler (2010) likelihood
2. **PhyloDeep** - Deep learning CNN model trained on simulated trees
3. **Bayesian MCMC** - Bayesian inference with PyMC for uncertainty quantification

All methods work with **tree data only** - no nucleotide sequences required.

## Key Features

- ✅ **Device-independent paths** - Works on any system via `config.py`
- ✅ **Comprehensive comparison** - Bias, variance, and statistical efficiency metrics
- ✅ **Multiple estimation methods** - MLE, PhyloDeep, and Bayesian MCMC
- ✅ **Tree-only analysis** - No sequence data needed
- ✅ **Automated visualization** - Generates comparison plots and reports

## Installation

### Prerequisites

- Python 3.12+ (Python 3.8-3.11 for PhyloDeep compatibility)
- `uv` package manager (recommended) or `pip`

### Quick Setup

```bash
# Clone the repository
git clone <repository-url>
cd PhyloDeepPOMP

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -e .

# Install optional Bayesian dependencies
pip install -e ".[bayesian]"
```

### Dependencies

**Core dependencies** (required):
- `phylodeep>=0.9` - Deep learning models for phylodynamics
- `scipy>=1.11.0` - Numerical optimization for MLE
- `ete3` - Tree manipulation (installed with phylodeep)
- `pandas`, `numpy` - Data processing
- `matplotlib`, `seaborn` - Visualization

**Optional dependencies** (for Bayesian analysis):
- `pymc>=5.0.0` - Bayesian MCMC sampling
- `arviz>=0.15.0` - Bayesian analysis and visualization

## Project Structure

```
PhyloDeepPOMP/
├── config.py                    # Device-independent path configuration
├── data/                        # Input data directory
│   └── phylodynamicsDL/
│       ├── output_trees/       # Input: phylogenetic trees (.nwk)
│       └── all_params.csv     # Input: true parameters
├── output/                      # Output data directory
│   ├── comparison_results/     # Comparison reports and plots
│   ├── mle_results/            # MLE estimates
│   ├── parameter_estimates/    # PhyloDeep estimates
│   └── bayesian_results/       # Bayesian estimates
├── src/                         # Source code directory
│   ├── analysis/               # Parameter estimation methods
│   │   ├── mle/                    # MLE estimation
│   │   │   ├── mle_birth_death.py      # MLE implementation (Stadler 2010)
│   │   │   └── mle_analysis.py         # MLE batch processing script
│   │   ├── bayesian/                # Bayesian MCMC estimation
│   │   │   ├── bayesian_birth_death.py # Bayesian MCMC implementation
│   │   │   └── bayesian_analysis.py    # Bayesian batch processing script
│   │   ├── phylodeep/               # PhyloDeep deep learning
│   │   │   └── phylodeep_analysis.py  # PhyloDeep batch processing script
│   │   ├── utils/                   # Shared utilities
│   │   │   └── batch_processor.py     # Unified batch processing
│   │   ├── run_all_analyses.py    # Launcher: runs all analyses
│   │   └── compare_results.py      # Comparison script
│   └── simulation/              # Tree simulation/generation
│       ├── generate_trees.py   # Tree generation script (refactored)
│       └── original/            # Original code from PhylodynamicsDL repo
│           ├── generate_trees.py
│           ├── generate_trees.R
│           └── README.md
├── MODEL_EXPLANATION.md        # Detailed model documentation
└── README.md                   # This file
```

## Quick Start

### 1. Verify Setup

```bash
# Check configuration
python config.py
```

### 2. Run Analysis

```bash
# Run all analyses (MLE, PhyloDeep, Bayesian)
python src/analysis/run_all_analyses.py

# Or run individual methods:
python src/analysis/mle/mle_analysis.py
python src/analysis/phylodeep/phylodeep_analysis.py
python src/analysis/bayesian/bayesian_analysis.py

# Then compare results
python src/analysis/compare_results.py
```

This will:
- Load trees from `data/phylodynamicsDL/output_trees/`
- Run all three estimation methods on each tree
- Compute bias, variance, and statistical efficiency
- Generate comparison plots and CSV reports
- Save results to `output/comparison_results/`

### 3. View Results

Results are saved in `output/comparison_results/`:
- `comparison_results.csv` - All estimates and metrics
- `efficiency_by_tip_size.csv` - Statistical efficiency by tree size
- `bias_variance_summary.csv` - Bias and variance summary
- `*.png` - Visualization plots

## Usage

### Running Individual Methods

#### MLE Estimation

```python
from src.analysis.mle import BirthDeathMLE

estimator = BirthDeathMLE("tree_file.nwk", sampling_prob=0.5)
result = estimator.estimate_bd()

print(f"λ (birth rate): {result['lambda']:.4f}")
print(f"μ (death rate): {result['mu']:.4f}")
print(f"R₀: {result['R_naught']:.4f}")
```

#### PhyloDeep Estimation

```python
from phylodeep import BD, FULL
from phylodeep.paramdeep import paramdeep

params = paramdeep(
    "tree_file.nwk",
    sampling_prob=0.5,
    model=BD,
    vector_representation=FULL,
    ci_computation=False
)
```

#### Bayesian MCMC Estimation

```python
from src.analysis.bayesian import BayesianBirthDeath

bayesian = BayesianBirthDeath("tree_file.nwk", sampling_prob=0.5)
result = bayesian.estimate_bd(
    draws=2000,      # MCMC samples
    tune=1000,       # Burn-in
    chains=2         # Number of chains
)

print(f"λ: {result['lambda_mean']:.4f} [{result['lambda_q2_5']:.4f}, {result['lambda_q97_5']:.4f}]")
```

### Configuration

Edit `config.py` to customize:
- Input/output directories
- Default sampling probability
- Minimum tree size

Or modify settings in `src/analysis/run_all_analyses.py` or individual analysis scripts:
```python
TARGET_TIP_SIZES = [50, 100, 200, 500]  # Specific sizes
# or
TARGET_TIP_SIZES = None  # All trees >= MIN_TIP_SIZE
```

## Methods

### Maximum Likelihood Estimation (MLE)

**Implementation**: Stadler (2010) "Sampling-through-time in birth-death trees"

- Uses numerical optimization (L-BFGS-B) to maximize likelihood
- Likelihood formula: `log L = (n-1)*log(λ) - (λ+μ)*T_total + n*log(ρ)`
- Provides point estimates with log-likelihood values
- Fast and statistically efficient

### PhyloDeep

**Implementation**: Pre-trained CNN models from PhyloDeep library

- Deep learning models trained on simulated birth-death trees
- Uses FULL tree representation (most accurate)
- Fast inference (seconds per tree)
- Provides point estimates

### Bayesian MCMC

**Implementation**: PyMC with Stadler (2010) likelihood

- Full posterior distributions, not just point estimates
- Provides credible intervals (95% CI)
- Can incorporate prior knowledge
- Slower than MLE/PhyloDeep (minutes per tree)

## Statistical Efficiency

**Definition**: `Efficiency = Var(MLE) / Var(PhyloDeep)`

- **Efficiency > 1**: PhyloDeep is more efficient (lower variance)
- **Efficiency < 1**: MLE is more efficient (lower variance)
- **Efficiency = 1**: Equal efficiency

The analysis computes efficiency by tree size to understand how tree characteristics affect estimation quality.

## Output Files

### Comparison Results

- **`comparison_results.csv`**: All estimates for each tree
  - Columns: `tree_idx`, `tip_size`, `true_lambda`, `true_mu`
  - MLE: `mle_lambda`, `mle_mu`, `mle_R0`, etc.
  - PhyloDeep: `phylodeep_lambda`, `phylodeep_mu`, etc.
  - Bayesian: `bayesian_lambda_mean`, `bayesian_lambda_q2_5`, etc.

- **`efficiency_by_tip_size.csv`**: Statistical efficiency metrics
  - Efficiency ratios for λ and μ
  - Bias and variance by method and tree size

- **`bias_variance_summary.csv`**: Summary statistics
  - Mean bias, variance, RMSE by method and tree size

### Visualizations

- **`scatter_mle_vs_phylodeep.png`**: Scatter plots comparing estimates
- **`efficiency_vs_tip_size.png`**: Efficiency trends by tree size
- **`bias_distribution.png`**: Bias distributions by method

## Data Requirements

### Input Data

- **Trees**: Newick format (`.nwk`) in `data/phylodynamicsDL/output_trees/`
- **Parameters**: CSV file with true parameters in `data/phylodynamicsDL/all_params.csv`
  - Required columns: `idx`, `tips`, `la` (λ), `psi` (μ)

### Tree Requirements

- ✅ Rooted trees with branch lengths
- ✅ Minimum 50 tips (for PhyloDeep)
- ✅ Newick format
- ❌ No nucleotide sequences needed

## Generating Trees

To generate new trees:

```bash
python src/simulation/generate_trees.py
```

**Note**: Requires `generate_bd` binary in PATH. The script uses device-independent paths from `config.py`.

### Original Implementation

The original tree generation code from the [PhylodynamicsDL repository](https://github.com/mcanearm/PhylodynamicsDL) is preserved in `src/simulation/original/` for historical reference and reproducibility.

## Documentation

- **`MODEL_EXPLANATION.md`**: Detailed explanation of PhyloDeep and Bayesian models
- **Code comments**: Inline documentation in all Python files

## Citation

If you use this code, please cite:

- **Stadler (2010)**: "Sampling-through-time in birth-death trees" - Journal of Theoretical Biology
- **PhyloDeep**: Original PhyloDeep paper (see PhyloDeep documentation)

## Troubleshooting

### PhyloDeep Import Errors

If you get `ModuleNotFoundError: No module named 'phylodeep'`:
```bash
pip install phylodeep
```

**Note**: PhyloDeep requires Python 3.8-3.11. If using Python 3.12+, you may need to use a virtual environment with Python 3.11.

### Bayesian Methods Not Available

If Bayesian methods fail:
```bash
pip install pymc arviz
```

### Tree File Not Found

Verify trees exist:
```bash
python config.py  # Check path verification
ls data/phylodynamicsDL/output_trees/*.nwk | head
```

### Memory Issues

For large datasets, process trees in batches by setting `TARGET_TIP_SIZES` in `run_analysis.py`.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with clear commit messages
4. Submit a pull request

## License

[Add your license here]

## Contact

[Add contact information here]
