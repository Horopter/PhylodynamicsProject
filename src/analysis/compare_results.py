"""
Compare Results: Compares results from MLE, PhyloDeep, and Bayesian analyses.

This script:
- Loads results from output/mle_results/, output/parameter_estimates/, output/bayesian_results/
- Merges results on tree_idx and tip_size
- Computes bias, variance, and statistical efficiency
- Generates comparison visualizations
- Saves comparison reports to output/comparison_results/
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Setup environment
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import setup_project_paths
setup_project_paths(__file__)

# Project configuration
from config import (
    PROJECT_ROOT,
    MLE_OUTPUT_DIR,
    PHYLODEEP_OUTPUT_DIR,
    COMPARISON_OUTPUT_DIR,
)

print("=" * 80)
print("Compare Results: MLE vs PhyloDeep vs Bayesian")
print("=" * 80)

# Delete and recreate output directory for clean results
import shutil
if COMPARISON_OUTPUT_DIR.exists():
    shutil.rmtree(COMPARISON_OUTPUT_DIR)
    print(f"Cleaned existing comparison output directory: {COMPARISON_OUTPUT_DIR}")
os.makedirs(COMPARISON_OUTPUT_DIR, exist_ok=True)
print(f"Created comparison output directory: {COMPARISON_OUTPUT_DIR}")

# Load results from each method
print("\n[STEP 1] Loading results from all methods...")
print("-" * 80)

# Load MLE results
mle_file = MLE_OUTPUT_DIR / "all_mle_results.csv"
if mle_file.exists():
    df_mle = pd.read_csv(str(mle_file))
    print(f"✓ Loaded MLE results: {len(df_mle)} trees")
    # Rename columns to have mle_ prefix
    mle_cols = {col: f'mle_{col}' if col not in ['tree_idx', 'tip_size', 'true_lambda', 'true_mu'] 
                else col for col in df_mle.columns}
    df_mle = df_mle.rename(columns=mle_cols)
else:
    print(f"✗ MLE results not found: {mle_file}")
    print("  Run mle_analysis.py first")
    df_mle = None

# Load PhyloDeep results
phylodeep_file = (
    PHYLODEEP_OUTPUT_DIR / "all_phylodeep_results.csv"
)
if phylodeep_file.exists():
    df_phylodeep = pd.read_csv(str(phylodeep_file))
    print(f"✓ Loaded PhyloDeep results: {len(df_phylodeep)} trees")
    # Rename columns to have phylodeep_ prefix
    phylodeep_cols = {
        col: f'phylodeep_{col}'
        if col not in ['tree_idx', 'tip_size', 'true_lambda', 'true_mu']
        else col
        for col in df_phylodeep.columns
    }
    df_phylodeep = df_phylodeep.rename(columns=phylodeep_cols)
else:
    print(f"✗ PhyloDeep results not found: {phylodeep_file}")
    print("  Run phylodeep_analysis.py first")
    df_phylodeep = None

# Load Bayesian results
bayesian_file = PROJECT_ROOT / "output" / "bayesian_results" / "all_bayesian_results.csv"
if bayesian_file.exists():
    df_bayesian = pd.read_csv(str(bayesian_file))
    print(f"✓ Loaded Bayesian results: {len(df_bayesian)} trees")
    # Rename columns to have bayesian_ prefix
    bayesian_cols = {
        col: f'bayesian_{col}'
        if col not in ['tree_idx', 'tip_size', 'true_lambda', 'true_mu']
        else col
        for col in df_bayesian.columns
    }
    df_bayesian = df_bayesian.rename(columns=bayesian_cols)
else:
    print(f"⚠ Bayesian results not found: {bayesian_file}")
    print("  Run bayesian_analysis.py first (optional)")
    df_bayesian = None

# Check if we have at least two methods to compare
if df_mle is None and df_phylodeep is None:
    print("\n✗ No results found! Run at least one analysis script first.")
    sys.exit(1)

# Merge results
print("\n[STEP 2] Merging results...")
print("-" * 80)

# Start with MLE or PhyloDeep as base
if df_mle is not None:
    df_comparison = df_mle.copy()
    base_method = "MLE"
elif df_phylodeep is not None:
    df_comparison = df_phylodeep.copy()
    base_method = "PhyloDeep"

# Merge PhyloDeep
if df_phylodeep is not None and df_mle is not None:
    df_comparison = pd.merge(
        df_comparison,
        df_phylodeep[[
            'tree_idx', 'tip_size', 'phylodeep_lambda',
            'phylodeep_mu', 'phylodeep_R0', 'phylodeep_infectious_period'
        ]],
        on=['tree_idx', 'tip_size'],
        how='outer'
    )
    print(f"✓ Merged PhyloDeep results")

# Merge Bayesian
if df_bayesian is not None:
    bayesian_cols_to_merge = (
        ['tree_idx', 'tip_size'] +
        [col for col in df_bayesian.columns if col.startswith('bayesian_')]
    )
    df_comparison = pd.merge(
        df_comparison,
        df_bayesian[bayesian_cols_to_merge],
        on=['tree_idx', 'tip_size'],
        how='outer'
    )
    print(f"✓ Merged Bayesian results")

print(f"Total trees in comparison: {len(df_comparison)}")

# Save merged results
merged_file = COMPARISON_OUTPUT_DIR / "all_comparison_results.csv"
df_comparison.to_csv(str(merged_file), index=False)
print(f"Saved merged results to: {merged_file}")

# Compute comparison statistics
print("\n[STEP 3] Computing comparison statistics...")
print("-" * 80)

comparison_stats = []
tip_sizes = sorted(df_comparison['tip_size'].dropna().unique())

for tip_size in tip_sizes:
    df_size = df_comparison[df_comparison['tip_size'] == tip_size].copy()
    
    if len(df_size) == 0:
        continue
    
    # MLE vs PhyloDeep comparison
    if 'mle_lambda' in df_size.columns and 'phylodeep_lambda' in df_size.columns:
        mle_lambda_valid = df_size['mle_lambda'].dropna()
        phylodeep_lambda_valid = df_size['phylodeep_lambda'].dropna()
        mle_mu_valid = df_size['mle_mu'].dropna()
        phylodeep_mu_valid = df_size['phylodeep_mu'].dropna()
        
        # Lambda statistics
        if len(mle_lambda_valid) > 0 and len(phylodeep_lambda_valid) > 0:
            mle_lambda_mean = mle_lambda_valid.mean()
            mle_lambda_var = mle_lambda_valid.var()
            phylodeep_lambda_mean = phylodeep_lambda_valid.mean()
            phylodeep_lambda_var = phylodeep_lambda_valid.var()
            
            # Bias
            mle_lambda_bias = None
            phylodeep_lambda_bias = None
            if df_size['true_lambda'].notna().any():
                true_lambda = df_size['true_lambda'].dropna().iloc[0]
                mle_lambda_bias = mle_lambda_mean - true_lambda
                phylodeep_lambda_bias = phylodeep_lambda_mean - true_lambda
            
            # Efficiency
            efficiency_lambda = (
                mle_lambda_var / phylodeep_lambda_var
                if phylodeep_lambda_var > 0
                else np.nan
            )
            
            comparison_stats.append({
                'tip_size': tip_size,
                'parameter': 'lambda',
                'method': 'MLE',
                'mean': mle_lambda_mean,
                'variance': mle_lambda_var,
                'bias': mle_lambda_bias,
                'n_estimates': len(mle_lambda_valid)
            })
            
            comparison_stats.append({
                'tip_size': tip_size,
                'parameter': 'lambda',
                'method': 'PhyloDeep',
                'mean': phylodeep_lambda_mean,
                'variance': phylodeep_lambda_var,
                'bias': phylodeep_lambda_bias,
                'n_estimates': len(phylodeep_lambda_valid)
            })
            
            comparison_stats.append({
                'tip_size': tip_size,
                'parameter': 'lambda',
                'method': 'Efficiency',
                'efficiency_ratio': efficiency_lambda,
                'n_estimates': min(len(mle_lambda_valid), len(phylodeep_lambda_valid))
            })
        
        # Mu statistics
        if len(mle_mu_valid) > 0 and len(phylodeep_mu_valid) > 0:
            mle_mu_mean = mle_mu_valid.mean()
            mle_mu_var = mle_mu_valid.var()
            phylodeep_mu_mean = phylodeep_mu_valid.mean()
            phylodeep_mu_var = phylodeep_mu_valid.var()
            
            # Bias
            mle_mu_bias = None
            phylodeep_mu_bias = None
            if df_size['true_mu'].notna().any():
                true_mu = df_size['true_mu'].dropna().iloc[0]
                mle_mu_bias = mle_mu_mean - true_mu
                phylodeep_mu_bias = phylodeep_mu_mean - true_mu
            
            # Efficiency
            efficiency_mu = mle_mu_var / phylodeep_mu_var if phylodeep_mu_var > 0 else np.nan
            
            comparison_stats.append({
                'tip_size': tip_size,
                'parameter': 'mu',
                'method': 'MLE',
                'mean': mle_mu_mean,
                'variance': mle_mu_var,
                'bias': mle_mu_bias,
                'n_estimates': len(mle_mu_valid)
            })
            
            comparison_stats.append({
                'tip_size': tip_size,
                'parameter': 'mu',
                'method': 'PhyloDeep',
                'mean': phylodeep_mu_mean,
                'variance': phylodeep_mu_var,
                'bias': phylodeep_mu_bias,
                'n_estimates': len(phylodeep_mu_valid)
            })
            
            comparison_stats.append({
                'tip_size': tip_size,
                'parameter': 'mu',
                'method': 'Efficiency',
                'efficiency_ratio': efficiency_mu,
                'n_estimates': min(len(mle_mu_valid), len(phylodeep_mu_valid))
            })

# Save statistics
if comparison_stats:
    df_stats = pd.DataFrame(comparison_stats)
    stats_file = COMPARISON_OUTPUT_DIR / "comparison_statistics.csv"
    df_stats.to_csv(str(stats_file), index=False)
    print(f"Saved comparison statistics to: {stats_file}")
    
    # Print summary
    print("\n" + "-" * 80)
    print("COMPARISON SUMMARY STATISTICS")
    print("-" * 80)
    
    for tip_size in tip_sizes[:5]:  # Show first 5 tip sizes
        df_size_stats = df_stats[df_stats['tip_size'] == tip_size]
        if len(df_size_stats) == 0:
            continue
        
        print(f"\nTip Size: n={tip_size}")
        print("-" * 60)
        
        for param in ['lambda', 'mu']:
            param_stats = df_size_stats[df_size_stats['parameter'] == param]
            if len(param_stats) == 0:
                continue
            
            print(f"\n  Parameter: {param}")
            for method in ['MLE', 'PhyloDeep']:
                method_stats = param_stats[param_stats['method'] == method]
                if len(method_stats) > 0:
                    s = method_stats.iloc[0]
                    print(f"    {method}:")
                    print(f"      Mean: {s['mean']:.6f}")
                    print(f"      Variance: {s['variance']:.6f}")
                    if pd.notna(s['bias']):
                        print(f"      Bias: {s['bias']:.6f}")
                    print(f"      N estimates: {s['n_estimates']}")
            
            # Efficiency
            eff_stats = param_stats[param_stats['method'] == 'Efficiency']
            if len(eff_stats) > 0:
                eff = eff_stats.iloc[0]['efficiency_ratio']
                print(f"    Statistical Efficiency (Var(MLE)/Var(PhyloDeep)): {eff:.4f}")
                if eff > 1:
                    print(f"      → PhyloDeep is {eff:.2f}x more efficient (lower variance)")
                elif eff < 1:
                    print(f"      → MLE is {1/eff:.2f}x more efficient (lower variance)")

# Create visualizations
print("\n[STEP 4] Creating visualizations...")
print("-" * 80)

try:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MLE vs PhyloDeep Parameter Estimates Comparison', fontsize=16)
    
    # Lambda comparison
    if 'mle_lambda' in df_comparison.columns and 'phylodeep_lambda' in df_comparison.columns:
        ax1 = axes[0, 0]
        mle_lambda = df_comparison['mle_lambda'].dropna()
        phylodeep_lambda = df_comparison['phylodeep_lambda'].dropna()
        if len(mle_lambda) > 0 and len(phylodeep_lambda) > 0:
            # Merge on tree_idx for scatter plot
            merged_lambda = pd.merge(
                df_comparison[['tree_idx', 'mle_lambda']].dropna(),
                df_comparison[['tree_idx', 'phylodeep_lambda']].dropna(),
                on='tree_idx'
            )
            if len(merged_lambda) > 0:
                ax1.scatter(
                    merged_lambda['mle_lambda'],
                    merged_lambda['phylodeep_lambda'],
                    alpha=0.5, s=20
                )
                min_val = min(
                    merged_lambda['mle_lambda'].min(),
                    merged_lambda['phylodeep_lambda'].min()
                )
                max_val = max(
                    merged_lambda['mle_lambda'].max(),
                    merged_lambda['phylodeep_lambda'].max()
                )
                ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
                ax1.set_xlabel('MLE λ')
                ax1.set_ylabel('PhyloDeep λ')
                ax1.set_title('Lambda (Birth Rate) Estimates')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
    
    # Mu comparison
    if 'mle_mu' in df_comparison.columns and 'phylodeep_mu' in df_comparison.columns:
        ax2 = axes[0, 1]
        merged_mu = pd.merge(
            df_comparison[['tree_idx', 'mle_mu']].dropna(),
            df_comparison[['tree_idx', 'phylodeep_mu']].dropna(),
            on='tree_idx'
        )
        if len(merged_mu) > 0:
            ax2.scatter(merged_mu['mle_mu'], merged_mu['phylodeep_mu'], alpha=0.5, s=20)
            min_val = min(merged_mu['mle_mu'].min(), merged_mu['phylodeep_mu'].min())
            max_val = max(merged_mu['mle_mu'].max(), merged_mu['phylodeep_mu'].max())
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
            ax2.set_xlabel('MLE μ')
            ax2.set_ylabel('PhyloDeep μ')
            ax2.set_title('Mu (Death Rate) Estimates')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
    
    # Efficiency by tip size
    ax3 = axes[1, 0]
    if len(comparison_stats) > 0:
        df_stats_plot = pd.DataFrame(comparison_stats)
        efficiency_data = df_stats_plot[df_stats_plot['method'] == 'Efficiency'].copy()
        if len(efficiency_data) > 0:
            lambda_eff = efficiency_data[efficiency_data['parameter'] == 'lambda']
            mu_eff = efficiency_data[efficiency_data['parameter'] == 'mu']
            
            if len(lambda_eff) > 0:
                ax3.plot(lambda_eff['tip_size'], lambda_eff['efficiency_ratio'], 
                        'o-', label='λ efficiency', markersize=8)
            if len(mu_eff) > 0:
                ax3.plot(mu_eff['tip_size'], mu_eff['efficiency_ratio'], 
                        's-', label='μ efficiency', markersize=8)
            ax3.axhline(y=1, color='r', linestyle='--', label='Equal efficiency')
            ax3.set_xlabel('Tree Size (number of tips)')
            ax3.set_ylabel('Statistical Efficiency\n(Var(MLE) / Var(PhyloDeep))')
            ax3.set_title('Statistical Efficiency vs Tree Size')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    
    # Bias comparison
    ax4 = axes[1, 1]
    if 'true_lambda' in df_comparison.columns and df_comparison['true_lambda'].notna().any():
        df_with_bias = df_comparison[df_comparison['true_lambda'].notna()].copy()
        if 'mle_lambda' in df_with_bias.columns and 'phylodeep_lambda' in df_with_bias.columns:
            df_with_bias['mle_lambda_bias'] = (
                df_with_bias['mle_lambda'] - df_with_bias['true_lambda']
            )
            df_with_bias['phylodeep_lambda_bias'] = (
                df_with_bias['phylodeep_lambda'] - df_with_bias['true_lambda']
            )
            
            mle_bias = df_with_bias['mle_lambda_bias'].dropna()
            phylodeep_bias = df_with_bias['phylodeep_lambda_bias'].dropna()
            
            if len(mle_bias) > 0 and len(phylodeep_bias) > 0:
                ax4.hist(mle_bias, bins=30, alpha=0.5, label='MLE bias', density=True)
                ax4.hist(phylodeep_bias, bins=30, alpha=0.5, label='PhyloDeep bias', density=True)
                ax4.axvline(x=0, color='r', linestyle='--', label='No bias')
                ax4.set_xlabel('Bias (Estimated - True)')
                ax4.set_ylabel('Density')
                ax4.set_title('Bias Distribution: Lambda')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'True parameter values\nnot available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Bias Comparison')
    
    plt.tight_layout()
    plot_file = COMPARISON_OUTPUT_DIR / "comparison_plots.png"
    plt.savefig(str(plot_file), dpi=300, bbox_inches='tight')
    print(f"Saved comparison plots to: {plot_file}")
    plt.close()
    
except Exception as e:
    print(f"Warning: Could not create plots: {str(e)}")

print("\n" + "=" * 80)
print("COMPARISON COMPLETE")
print("=" * 80)
print(f"\nResults saved in: {COMPARISON_OUTPUT_DIR}/")
print("  - all_comparison_results.csv: Merged results from all methods")
print("  - comparison_statistics.csv: Summary statistics (bias, variance, efficiency)")
print("  - comparison_plots.png: Visualization plots")
print("\n" + "=" * 80)

