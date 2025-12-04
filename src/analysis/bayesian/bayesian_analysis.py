"""
Bayesian Analysis: Runs Bayesian MCMC estimation on phylogenetic trees.

Uses the unified batch processor with the Bayesian estimator class.
"""

# Setup environment first (adds paths)
from utils.common import setup_analysis_environment
setup_analysis_environment(__file__)

# Now we can import from config and other modules
from config import BAYESIAN_OUTPUT_DIR, DEFAULT_SAMPLING_PROBA
from utils.batch_processor import process_trees_batch

# Bayesian imports
try:
    from bayesian.bayesian_birth_death import BayesianBirthDeath, HAS_PYMC
    if not HAS_PYMC:
        print(
            "ERROR: PyMC is not installed. "
            "Install with: pip install -r requirements-bayesian.txt"
        )
        sys.exit(1)
    HAS_BAYESIAN = True
except ImportError as e:
    HAS_BAYESIAN = False
    print(
        f"ERROR: Bayesian methods not available. "
        f"Import error: {str(e)}\n"
        f"Install PyMC: pip install -r requirements-bayesian.txt"
    )
    sys.exit(1)
except Exception as e:
    HAS_BAYESIAN = False
    print(
        f"ERROR: Failed to import Bayesian module: {str(e)}\n"
        f"Install PyMC: pip install -r requirements-bayesian.txt"
    )
    sys.exit(1)


def bayesian_estimator(tree_file: str, sampling_prob: float, draws=1000, tune=500, chains=4):
    """Estimate parameters using Bayesian MCMC."""
    try:
        estimator = BayesianBirthDeath(tree_file, sampling_prob=sampling_prob)
        result = estimator.estimate_bd(draws=draws, tune=tune, chains=chains)
        
        if result.get('success', False):
            return {
                'success': True,
                'lambda_mean': result.get('lambda_mean'),
                'lambda_median': result.get('lambda_median'),
                'lambda_std': result.get('lambda_std'),
                'lambda_q2_5': result.get('lambda_q2_5'),
                'lambda_q97_5': result.get('lambda_q97_5'),
                'mu_mean': result.get('mu_mean'),
                'mu_median': result.get('mu_median'),
                'mu_std': result.get('mu_std'),
                'mu_q2_5': result.get('mu_q2_5'),
                'mu_q97_5': result.get('mu_q97_5'),
                'R0_mean': result.get('R_naught_mean'),
                'R0_median': result.get('R_naught_median'),
                'R0_std': result.get('R_naught_std'),
                'R0_q2_5': result.get('R_naught_q2_5'),
                'R0_q97_5': result.get('R_naught_q97_5'),
                'infectious_period_mean': result.get('Infectious_period_mean'),
                'infectious_period_std': result.get('Infectious_period_std'),
                # MCMC diagnostics
                'lambda_ess': result.get('lambda_ess'),
                'mu_ess': result.get('mu_ess'),
                'lambda_rhat': result.get('lambda_rhat'),
                'mu_rhat': result.get('mu_rhat'),
                'n_divergences': result.get('n_divergences'),
            }
        else:
            error_msg = result.get('message', 'Unknown error')
            return {
                'success': False,
                'lambda_mean': None,
                'mu_mean': None,
                'R0_mean': None,
                'error': error_msg[:200] if error_msg else 'Bayesian estimation failed'
            }
    except Exception as e:
        error_msg = str(e)
        if len(error_msg) > 200:
            error_msg = error_msg[:200]
        return {
            'success': False,
            'lambda_mean': None,
            'mu_mean': None,
            'R0_mean': None,
            'error': f'Exception: {error_msg}'
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Bayesian analysis")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,  # Default to parallel (use all CPUs)
        help="Number of parallel jobs (-1 = all CPUs, 1 = sequential, N > 1 = N workers)"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run sequentially instead of in parallel (overrides --n-jobs)"
    )
    args = parser.parse_args()
    
    # Determine n_jobs
    if args.sequential:
        n_jobs = 1
        print("Note: Running in sequential mode (--sequential flag)")
    elif args.n_jobs == -1:
        import multiprocessing
        n_jobs = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
        print(f"Note: Running in parallel mode with {n_jobs} workers")
    else:
        n_jobs = args.n_jobs
        if n_jobs > 1:
            print(f"Note: Running in parallel mode with {n_jobs} workers")
        else:
            print("Note: Running in sequential mode")
    
    print("Note: Bayesian analysis is slower than MLE/PhyloDeep")
    
    process_trees_batch(
        estimator_func=bayesian_estimator,
        output_dir=BAYESIAN_OUTPUT_DIR,
        method_name="Bayesian",
        sampling_prob=DEFAULT_SAMPLING_PROBA,
        draws=1000,
        tune=500,
        chains=4,  # Use 4 chains for robust diagnostics
        progress_interval=1,  # Show progress for each tree (slower method)
        n_jobs=n_jobs,
    )
