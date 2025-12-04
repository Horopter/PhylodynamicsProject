"""
Unified batch processing module for running parameter estimation on multiple
    trees.

This module provides shared functionality for batch processing trees across
all estimation methods (MLE, PhyloDeep, Bayesian).

Author: Santosh Desai <santoshdesai12@hotmail.com>
"""

import os
import sys
import time
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# Setup environment (warnings and paths)
from .common import setup_analysis_environment
setup_analysis_environment(__file__)

from config import (
    TREES_DIR,
    PARAMS_FILE,
    DEFAULT_SAMPLING_PROBA,
    MIN_TIP_SIZE
)


def load_tree_metadata(target_tip_sizes: Optional[List[int]] = None):
    """
    Load tree metadata and determine which trees to analyze.
    
    Parameters:
    -----------
    target_tip_sizes : list of int, optional
        Specific tip sizes to analyze. If None, analyzes all trees >=
             MIN_TIP_SIZE.
    
    Returns:
    --------
    df_params : pd.DataFrame
        Parameters dataframe
    target_tip_sizes : list
        List of tip sizes to process
    """
    df_params = pd.read_csv(str(PARAMS_FILE))
    
    if target_tip_sizes is None:
        df_filtered = df_params[df_params['tips'] >= MIN_TIP_SIZE].copy()
        target_tip_sizes = sorted(df_filtered['tips'].unique())
    else:
        # Filter to only available sizes
        available_sizes = {}
        for size in target_tip_sizes:
            count = len(df_params[df_params['tips'] == size])
            available_sizes[size] = count
        target_tip_sizes = [
            s for s in target_tip_sizes if available_sizes.get(s, 0) > 0
        ]
    
    return df_params, target_tip_sizes


def _process_single_tree(
    tree_info: tuple,
    estimator_func_name: str,
    estimator_module: str,
    sampling_prob: float,
    estimator_kwargs: dict
) -> Dict[str, Any]:
    """
    Process a single tree (used for parallel execution).
    
    This function is called in a separate process, so we need to re-import
    the estimator function and set up the environment properly.
    """
    import importlib
    import sys
    import os
    from pathlib import Path
    
    try:
        # Setup paths - need to add project root, src, and analysis directories
        batch_processor_file = Path(__file__)
        project_root = batch_processor_file.parent.parent.parent.parent
        src_dir = project_root / "src"
        analysis_dir = src_dir / "analysis"
        
        # Add to Python path if not already there
        paths_to_add = [
            str(project_root),
            str(src_dir),
            str(analysis_dir),
        ]
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)
        
        # Set up environment variables
        os.environ['PYTHONPATH'] = ':'.join(paths_to_add +
            [os.environ.get('PYTHONPATH', '')])
        
        # Import the estimator function
        module = importlib.import_module(estimator_module)
        estimator_func = getattr(module, estimator_func_name)
        
        tree_idx, tip_size, tree_file_str, true_lambda, true_mu = tree_info
        
        result = {
            'tree_idx': tree_idx,
            'tip_size': tip_size,
            'true_lambda': true_lambda,
            'true_mu': true_mu,
        }
        
        # Call estimator function
        estimate_result = estimator_func(
            tree_file_str, sampling_prob, **estimator_kwargs
        )
        
        if estimate_result and estimate_result.get('success', True):
            result.update(estimate_result)
        else:
            result.update(estimate_result or {})
            
    except Exception as e:
        # Return error info with full traceback for debugging
        import traceback
        error_msg = str(e)
        if len(error_msg) > 200:
            error_msg = error_msg[:200]
        result = {
            'tree_idx': tree_info[0] if tree_info else None,
            'tip_size': tree_info[1] if tree_info and len(tree_info) > 1 else
                None,
            'true_lambda': tree_info[3] if tree_info and len(tree_info) > 3
                else None,
            'true_mu': tree_info[4] if tree_info and len(tree_info) > 4 else
                None,
            'error': error_msg
        }
    
    return result


def process_trees_batch(
    estimator_func: Callable[[str, float], Dict[str, Any]],
    output_dir: Path,
    method_name: str,
    target_tip_sizes: Optional[List[int]] = None,
    sampling_prob: float = DEFAULT_SAMPLING_PROBA,
    progress_interval: int = 10,
    n_jobs: int = 1,
    **estimator_kwargs
):
    """
    Process trees in batch using the provided estimator function.
    
    Parameters:
    -----------
    estimator_func : callable
        Function that takes (tree_file, sampling_prob) and returns result dict
    output_dir : Path
        Directory to save results
    method_name : str
        Name of the method (for output files and logging)
    target_tip_sizes : list of int, optional
        Specific tip sizes to analyze
    sampling_prob : float
        Sampling probability
    progress_interval : int
        Print progress every N trees
    n_jobs : int
        Number of parallel jobs to use (1 = sequential, -1 = all CPUs)
    **estimator_kwargs
        Additional keyword arguments to pass to estimator_func
    
    Returns:
    --------
    df_all : pd.DataFrame
        Combined results from all trees
    """
    print("=" * 80)
    print(f"{method_name} Analysis")
    print("=" * 80)
    
    # Load metadata
    print("\n[STEP 1] Loading tree metadata...")
    print("-" * 80)
    df_params, target_tip_sizes = load_tree_metadata(target_tip_sizes)
    print(f"Loaded {len(df_params)} tree records")
    print(
        f"Tip size range: {df_params['tips'].min()} - "
        f"{df_params['tips'].max()}"
    )
    
    if not target_tip_sizes:
        print("\nNo trees found for analysis!")
        return pd.DataFrame()
    
    tip_sizes_str = (
        f"{target_tip_sizes[:10]}"
        f"{'...' if len(target_tip_sizes) > 10 else ''}"
    )
    print(f"\nAnalyzing trees with tip sizes: {tip_sizes_str}")
    
    # Delete and recreate output directory for clean results
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Cleaned existing output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    print(f"\n[STEP 2] Running {method_name} on trees...")
    print("-" * 80)
    print(f"Using sampling probability: {sampling_prob}")
    
    # Determine number of workers
    # Default behavior: use parallel processing if n_jobs > 1 or n_jobs == -1
    if n_jobs == -1:
        import multiprocessing
        n_jobs = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
        use_parallel = True
    elif n_jobs > 1:
        use_parallel = True
    else:
        n_jobs = 1
        use_parallel = False
    
    if use_parallel:
        print(f"Using parallel processing with {n_jobs} workers")
    else:
        print("Using sequential processing")
    
    # Get estimator function info for parallel execution
    estimator_func_name = estimator_func.__name__
    estimator_module = estimator_func.__module__
    
    # If the module is __main__, we need to use the actual module name
    # This happens when scripts are run directly
    if estimator_module == '__main__':
        # Try to infer from the function's file location
        func_file = Path(estimator_func.__code__.co_filename)
        if 'mle' in str(func_file):
            estimator_module = 'mle.mle_analysis'
        elif 'bayesian' in str(func_file):
            estimator_module = 'bayesian.bayesian_analysis'
        elif 'phylodeep' in str(func_file):
            estimator_module = 'phylodeep_method.phylodeep_analysis'
    
    start_time = time.time()
    all_results = []
    
    for tip_size in target_tip_sizes:
        print(f"\n{'='*80}")
        print(f"Analyzing trees with n={tip_size} tips")
        print(f"{'='*80}")
        
        tree_indices = df_params[df_params['tips'] == tip_size]['idx'].values
        num_trees = len(tree_indices)
        print(f"Found {num_trees} trees with {tip_size} tips")
        
        # Prepare tree information for processing
        tree_tasks = []
        for tree_idx in tree_indices:
            tree_file = TREES_DIR / f"tree_{tree_idx}.nwk"
            
            if not tree_file.exists():
                continue
            
            tree_file_str = str(tree_file)
            
            # Get true parameters if available
            true_params_row = df_params[df_params['idx'] == tree_idx]
            true_lambda = (
                true_params_row['la'].iloc[0]
                if 'la' in true_params_row.columns
                else None
            )
            true_mu = (
                true_params_row['psi'].iloc[0]
                if 'psi' in true_params_row.columns
                else None
            )
            
            tree_tasks.append((
                tree_idx, tip_size, tree_file_str, true_lambda, true_mu
            ))
        
        results_for_size = []
        successful = 0
        failed = 0
        
        if use_parallel and len(tree_tasks) > 1:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = {
                    executor.submit(
                        _process_single_tree,
                        task,
                        estimator_func_name,
                        estimator_module,
                        sampling_prob,
                        estimator_kwargs
                    ): task[0]  # tree_idx
                    for task in tree_tasks
                }
                
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    tree_idx = futures[future]
                    
                    if completed % progress_interval == 0:
                        print(
                            f"  Processed {completed}/{len(tree_tasks)} "
                            f"trees (idx={tree_idx})..."
                        )
                    
                    try:
                        # Add timeout to prevent hanging
                        # 5 minute timeout per tree
                        result = future.result(timeout=300)
                        if result.get('error'):
                            failed += 1
                        elif (result.get('lambda') is not None or
                              result.get('lambda_mean') is not None):
                            successful += 1
                        else:
                            failed += 1
                        results_for_size.append(result)
                    except TimeoutError:
                        failed += 1
                        tree_idx = futures[future]
                        print(
                            f"    [WARN] Tree {tree_idx} timed out after 5 "
                            "minutes"
                        )
                        results_for_size.append({
                            'tree_idx': tree_idx,
                            'tip_size': tip_size,
                            'true_lambda': true_lambda,
                            'true_mu': true_mu,
                            'error': 'Timeout: exceeded 5 minute limit'
                        })
                    except Exception as e:
                        failed += 1
                        tree_idx = futures[future]
                        error_msg = str(e)
                        if len(error_msg) > 200:
                            error_msg = error_msg[:200]
                        print(
                            f"    [WARN] Tree {tree_idx} failed: "
                            f"{error_msg[:50]}"
                        )
                        results_for_size.append({
                            'tree_idx': tree_idx,
                            'tip_size': tip_size,
                            'true_lambda': true_lambda,
                            'true_mu': true_mu,
                            'error': error_msg
                        })
        else:
            # Sequential processing
            for i, (tree_idx, tip_size, tree_file_str, true_lambda,
                true_mu) in enumerate(tree_tasks):
                result = {
                    'tree_idx': tree_idx,
                    'tip_size': tip_size,
                    'true_lambda': true_lambda,
                    'true_mu': true_mu,
                }
                
                try:
                    if (i + 1) % progress_interval == 0:
                        print(
                            f"  Processing tree {i+1}/{len(tree_tasks)} "
                            f"(idx={tree_idx})..."
                        )
                    
                    # Call estimator function
                    estimate_result = estimator_func(
                        tree_file_str, sampling_prob, **estimator_kwargs
                    )
                    
                    if estimate_result and estimate_result.get('success',
                        True):
                        result.update(estimate_result)
                        successful += 1
                    else:
                        result.update(estimate_result or {})
                        failed += 1
                except Exception as e:
                    failed += 1
                    if (i + 1) % progress_interval == 0:
                        print(f"    Error: {str(e)[:50]}")
                
                results_for_size.append(result)
        
        print(f"\nSummary for n={tip_size}:")
        print(f"  Successful: {successful}/{num_trees}")
        print(f"  Failed: {failed}/{num_trees}")
        
        # Save results for this tip size
        if results_for_size:
            df_size = pd.DataFrame(results_for_size)
            all_results.append(df_size)

            output_file = (
                output_dir / f"{method_name.lower()}_results_n{tip_size}.csv"
            )
            df_size.to_csv(str(output_file), index=False)
            print(f"  Saved results to: {output_file}")
    
    # Combine and save all results
    if all_results:
        print("\n" + "=" * 80)
        print("[STEP 3] Saving combined results...")
        print("=" * 80)
        
        df_all = pd.concat(all_results, ignore_index=True)
        combined_file = output_dir / f"all_{method_name.lower()}_results.csv"
        df_all.to_csv(str(combined_file), index=False)
        print(f"Saved combined results to: {combined_file}")
        print(f"Total trees analyzed: {len(df_all)}")
        
        # Count successful estimates (check for lambda column)
        success_col = (
            'lambda' if 'lambda' in df_all.columns
            else list(df_all.columns)[4]
            if len(df_all.columns) > 4
            else None
        )
        if success_col:
            successful_count = df_all[success_col].notna().sum()
            print(f"Successful estimates: {successful_count}")
    else:
        df_all = pd.DataFrame()
    
    # Calculate total time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print("\n" + "=" * 80)
    print(f"{method_name.upper()} ANALYSIS COMPLETE")
    print("=" * 80)
    print(
        f"Total time: {hours:02d}:{minutes:02d}:{seconds:02d} "
        f"({elapsed_time:.2f} seconds)"
    )
    print(f"Results saved in: {output_dir}/")
    print("=" * 80)
    
    return df_all

