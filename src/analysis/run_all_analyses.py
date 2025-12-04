"""
Launcher script: Runs all three analysis methods (MLE, PhyloDeep, Bayesian).

This script can run analyses either sequentially or in parallel.
Each analysis saves results to its respective output directory.
After all analyses complete, you can run compare_results.py to compare them.

Author: Santosh Desai <santoshdesai12@hotmail.com>
"""

import sys
import subprocess
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse


def run_analysis_script(script_path: Path, method_name: str):
    """Run a single analysis script and return success status."""
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False
        )
        return True, None
    except subprocess.CalledProcessError as e:
        return False, f"Exit code {e.returncode}"
    except Exception as e:
        return False, str(e)


def main(parallel: bool = True):
    """Main function to run all analyses."""
    # Get the directory containing this script
    script_dir = Path(__file__).parent

    print("=" * 80)
    print("Running All Analyses: MLE, PhyloDeep, and Bayesian")
    print("=" * 80)

    if parallel:
        print("\nRunning analyses in PARALLEL mode")
    else:
        print("\nRunning analyses in SEQUENTIAL mode")

    start_time = time.time()

    # Define all analysis scripts
    scripts = {
        "MLE": script_dir / "mle" / "mle_analysis.py",
        "PhyloDeep": script_dir / "phylodeep_method" / "phylodeep_analysis.py",
        "Bayesian": script_dir / "bayesian" / "bayesian_analysis.py",
    }

    results = {}

    if parallel:
        # Run all analyses in parallel
        print("\n" + "=" * 80)
        print("Running all analyses in parallel...")
        print("=" * 80)

        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(run_analysis_script, script, name): name
                for name, script in scripts.items()
            }

            for future in as_completed(futures):
                method_name = futures[future]
                success, error = future.result()
                results[method_name] = (success, error)

                if success:
                    print(
                        f"\n[OK] {method_name} analysis completed successfully"
                    )
                else:
                    if method_name == "Bayesian":
                        print(
                            f"\n[WARN] {method_name} analysis failed: {error}"
                        )
                        print(
                            "   (This is optional - may require PyMC "
                            "installation)"
                        )
                    else:
                        print(
                            f"\n[FAIL] {method_name} analysis failed: {error}"
                        )
    else:
        # Run analyses sequentially
        for i, (method_name, script_path) in enumerate(scripts.items(), 1):
            print("\n" + "=" * 80)
            print(f"STEP {i}: Running {method_name} Analysis")
            print("=" * 80)

            success, error = run_analysis_script(script_path, method_name)
            results[method_name] = (success, error)

            if success:
                print(f"\n[OK] {method_name} analysis completed successfully")
            else:
                if method_name == "Bayesian":
                    print(
                        f"\n[WARN] {method_name} analysis failed: {error}"
                    )
                    print(
                        "   (This is optional - may require PyMC "
                        "installation)"
                    )
                else:
                    print(f"\n[FAIL] {method_name} analysis failed: {error}")

    # Calculate total time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    print("\n" + "=" * 80)
    print("ALL ANALYSES COMPLETE")
    print("=" * 80)
    print(
        f"Total time: {hours:02d}:{minutes:02d}:{seconds:02d} "
        f"({elapsed_time:.2f} seconds)"
    )

    # Summary
    successful = sum(1 for success, _ in results.values() if success)
    print(
        f"\nSummary: {successful}/{len(results)} analyses completed "
        "successfully"
    )

    print("\nNext step: Run compare_results.py to compare the results")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MLE, PhyloDeep, and Bayesian analyses"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run analyses sequentially instead of in parallel"
    )
    args = parser.parse_args()

    main(parallel=not args.sequential)
