"""
MLE Analysis: Runs Maximum Likelihood Estimation on phylogenetic trees.

Uses the unified batch processor with the MLE estimator class.
"""

# Setup environment first (adds paths)
from utils.common import setup_analysis_environment
setup_analysis_environment(__file__)

# Now we can import from config and other modules
from config import MLE_OUTPUT_DIR, DEFAULT_SAMPLING_PROBA
from utils.batch_processor import process_trees_batch
from mle.mle_birth_death import BirthDeathMLE


def mle_estimator(tree_file: str, sampling_prob: float):
    """Estimate parameters using MLE."""
    estimator = BirthDeathMLE(tree_file, sampling_prob=sampling_prob)
    result = estimator.estimate_bd()
    
    if result['success']:
        return {
            'lambda': result['lambda'],
            'mu': result['mu'],
            'R0': result['R_naught'],
            'infectious_period': result['Infectious_period'],
            'log_likelihood': result['log_likelihood'],
            'n_iterations': result['n_iterations'],
        }
    else:
        return {
            'lambda': None,
            'mu': None,
            'R0': None,
            'infectious_period': None,
            'log_likelihood': None,
            'n_iterations': None,
        }


if __name__ == "__main__":
    import argparse
    import multiprocessing
    
    parser = argparse.ArgumentParser(description="Run MLE analysis")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (1 = sequential, N > 1 = parallel with N workers)"
    )
    args = parser.parse_args()
    
    process_trees_batch(
        estimator_func=mle_estimator,
        output_dir=MLE_OUTPUT_DIR,
        method_name="MLE",
        sampling_prob=DEFAULT_SAMPLING_PROBA,
        n_jobs=args.n_jobs,
    )
