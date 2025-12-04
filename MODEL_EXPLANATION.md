# PhyloDeep and Bayesian Model Estimation

This document explains how `phylodeep_estimation_generation.py` and the Bayesian models work for estimating birth-death parameters from phylogenetic trees.

## Overview

Both methods estimate the same parameters (λ, μ, R₀, infectious period) from phylogenetic trees, but use different approaches:
- **PhyloDeep**: Deep learning CNN model trained on simulated trees
- **Bayesian MCMC**: Bayesian inference using Stadler (2010) likelihood

**Important**: Both methods work with **tree data only** - no nucleotide sequences required.

## PhyloDeep Estimation (`phylodeep_estimation_generation.py`)

### What It Does

This script uses PhyloDeep's pre-trained deep learning models to estimate birth-death parameters from phylogenetic trees.

### How It Works

1. **Tree Encoding**: 
   - Converts phylogenetic trees into feature representations
   - Uses `FULL` representation (most accurate) - encodes the full tree structure
   - Alternative: `SUMSTATS` (summary statistics) - faster but less accurate

2. **Model Prediction**:
   - Loads pre-trained CNN models (trained on simulated birth-death trees)
   - Feeds encoded trees through the neural network
   - Outputs parameter estimates

3. **Model Selection** (optional):
   - Can select between BD, BDEI, BDSS models
   - For this project, we use BD only since all trees are birth-death

### Key Features

- **Input**: Tree files (`.nwk` format) from `data/phylodynamicsDL/output_trees/`
- **Output**: Parameter estimates (λ, μ, R₀, infectious period)
- **Speed**: Fast - processes trees in seconds
- **Data Requirements**: Tree topology and branch lengths only
- **Minimum Tree Size**: 50 tips (PhyloDeep requirement)

### Code Structure

```python
# Load trees grouped by tip size
for tip_size in TARGET_TIP_SIZES:
    tree_indices = df_params[df_params['tips'] == tip_size]['idx'].values
    
    for tree_idx in tree_indices:
        tree_file = TREES_DIR / f"tree_{tree_idx}.nwk"
        
        # Estimate parameters using PhyloDeep
        params = paramdeep(
            tree_file_str,
            SAMPLING_PROBA,
            model=BD,
            vector_representation=FULL,
            ci_computation=False
        )
```

### Output Files

- `all_estimates.csv`: All PhyloDeep estimates
- `estimates_n{size}.csv`: Estimates grouped by tip size
- `summary_statistics.csv`: Mean, std, median for each parameter by tip size

**Note**: Output files are saved to `output/parameter_estimates/`.

## Bayesian MCMC Estimation (`bayesian_birth_death.py`)

### What It Does

I implemented this to get posterior distributions for birth-death parameters using Bayesian MCMC sampling. The key advantage over MLE is full uncertainty quantification - not just point estimates but credible intervals.

### How It Works

1. **Tree Statistics Extraction**:
   - Extracts tree topology (number of tips, branching events)
   - Extracts branch lengths (total branch length, tree height)
   - Same statistics as MLE for consistency

2. **Likelihood Function**:
   - Uses Stadler (2010) log-likelihood
   - Formula: `(n-1)*log(λ) - (λ+μ)*T_total + n*log(ρ)`
   - Only needs tree structure - no sequences

3. **MCMC Sampling**:
   - Sets up priors for λ and μ
   - Uses PyMC to sample from posterior distribution
   - Returns posterior means, medians, and credible intervals

### Key Features

- **Input**: Tree files (`.nwk` format)
- **Output**: Posterior distributions with credible intervals
- **Speed**: Slower than PhyloDeep - minutes per tree (depends on draws)
- **Data Requirements**: Tree topology and branch lengths only
- **Uncertainty**: Full posterior distributions, not just point estimates

### Code Structure

```python
# Initialize Bayesian estimator
bayesian = BayesianBirthDeath(tree_file, sampling_prob=0.5)

# Run MCMC
result = bayesian.estimate_bd(
    draws=2000,      # Number of MCMC samples
    tune=1000,       # Burn-in samples
    chains=2         # Number of chains
)

# Result includes:
# - lambda_mean, lambda_std, lambda_q2_5, lambda_q97_5
# - mu_mean, mu_std, mu_q2_5, mu_q97_5
# - R_naught_mean, R_naught_q2_5, R_naught_q97_5
```

### Priors

- **Uniform priors**: λ ~ Uniform(0.01, 10.0), μ ~ Uniform(0.01, λ-ε)
- **Exponential priors**: Alternative option
- **Constraint**: μ < λ (ensures positive net diversification rate)

### Output Format

Returns dictionary with:
- **Posterior means**: Expected values
- **Posterior medians**: Robust estimates
- **Standard deviations**: Uncertainty measure
- **Credible intervals**: 2.5% and 97.5% quantiles (95% CI)

## Comparison: PhyloDeep vs Bayesian

| Feature | PhyloDeep | Bayesian MCMC |
|---------|-----------|---------------|
| **Method** | Deep learning CNN | MCMC sampling |
| **Speed** | Fast (seconds) | Slower (minutes) |
| **Output** | Point estimates | Posterior distributions |
| **Uncertainty** | Limited (bootstrap CIs) | Full posterior (credible intervals) |
| **Training** | Pre-trained on simulations | No training needed |
| **Data** | Tree topology + branch lengths | Tree topology + branch lengths |
| **Sequences** | Not required | Not required |

## Data Requirements

Both methods require:
- ✅ Phylogenetic trees in Newick format (`.nwk`)
- ✅ Trees with branch lengths (time information)
- ✅ Rooted trees
- ✅ Minimum 50 tips (for PhyloDeep)

Both methods do **NOT** require:
- ❌ Nucleotide sequences
- ❌ Sequence alignments
- ❌ Substitution model parameters
- ❌ Molecular evolution data

## Stadler (2010) Likelihood

The Bayesian method uses the Stadler (2010) likelihood, which models the birth-death process that generated the tree. It uses:

- **n**: Number of tips (tree topology)
- **T_total**: Total branch length (sum of all branch lengths)
- **ρ**: Sampling probability

The likelihood formula:
```
log L = (n-1) * log(λ) - (λ + μ) * T_total + n * log(ρ)
```

This models the probability of observing the tree given birth rate λ and death rate μ, accounting for incomplete sampling (ρ).

## Usage

### PhyloDeep Estimation

```bash
python src/analysis/phylodeep/phylodeep_analysis.py
```

Processes all trees in `data/phylodynamicsDL/output_trees/` and saves results to `output/parameter_estimates/`.

### Bayesian Estimation

```python
from src.analysis.bayesian import BayesianBirthDeath

bayesian = BayesianBirthDeath("tree_file.nwk", sampling_prob=0.5)
result = bayesian.estimate_bd(draws=2000, tune=1000, chains=2)
```

Or use in the comparison script:
```bash
python src/analysis/run_all_analyses.py
python src/analysis/compare_results.py
```

**Note**: The analysis folder is organized into subfolders:
- `mle/` - MLE estimation (class + script)
- `bayesian/` - Bayesian MCMC (class + script)
- `phylodeep/` - PhyloDeep DL (script only)
- `utils/` - Shared batch processor

This runs both PhyloDeep and Bayesian (and MLE) on the same trees for comparison.

## Why Both Methods?

- **PhyloDeep**: Fast, good for large-scale analysis, pre-trained models
- **Bayesian**: Provides uncertainty quantification, can incorporate prior knowledge, full posterior distributions

Together, they allow us to:
1. Compare point estimates (PhyloDeep vs MLE)
2. Assess uncertainty (Bayesian credible intervals)
3. Evaluate statistical efficiency (variance comparison)
4. Understand how tree size affects estimation quality

## Implementation Notes

- Both methods extract the same tree statistics for consistency
- Both use tree-only data (no sequences)
- Bayesian uses the same Stadler likelihood as MLE
- PhyloDeep uses pre-trained neural networks
- Results can be directly compared since they estimate the same parameters

