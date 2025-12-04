"""
Daemon process for running all analyses and comparison.

This script runs as a background daemon that:
1. Triggers MLE, Bayesian, and PhyloDeep analyses in parallel
2. Waits for all analyses to complete
3. Automatically runs comparison when done
4. Logs all output to files
5. Can run even when laptop lid is closed

Usage:
    # Run in background (daemon mode)
    python daemon_runner.py start
    
    # Check status
    python daemon_runner.py status
    
    # Stop daemon
    python daemon_runner.py stop
    
    # View logs
    tail -f output/daemon/daemon.log

Author: Santosh Desai <santoshdesai12@hotmail.com>
"""

import os
import sys
import time
import signal
import subprocess
import argparse
import logging
import shutil
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil

# Setup environment
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import setup_project_paths, OUTPUT_DIR, get_project_root
setup_project_paths(__file__)

# Daemon configuration
DAEMON_DIR = OUTPUT_DIR / "daemon"
DAEMON_PID_FILE = DAEMON_DIR / "daemon.pid"
DAEMON_LOG_FILE = DAEMON_DIR / "daemon.log"
DAEMON_STDOUT = DAEMON_DIR / "daemon_stdout.log"
DAEMON_STDERR = DAEMON_DIR / "daemon_stderr.log"

# Individual analysis log files
MLE_LOG_FILE = DAEMON_DIR / "mle_analysis.log"
PHYLODEEP_LOG_FILE = DAEMON_DIR / "phylodeep_analysis.log"
BAYESIAN_LOG_FILE = DAEMON_DIR / "bayesian_analysis.log"
COMPARISON_LOG_FILE = DAEMON_DIR / "comparison.log"

# Create daemon directory
DAEMON_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging():
    """Setup logging for daemon."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(DAEMON_LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def get_daemon_pid():
    """Get the PID of the running daemon, if any."""
    if DAEMON_PID_FILE.exists():
        try:
            with open(DAEMON_PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            # Check if process is still running
            if psutil.pid_exists(pid):
                try:
                    proc = psutil.Process(pid)
                    # Check if it's actually our daemon
                    if 'daemon_runner.py' in ' '.join(proc.cmdline()):
                        return pid
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            # PID file exists but process is dead - remove stale file
            DAEMON_PID_FILE.unlink()
        except (ValueError, FileNotFoundError):
            pass
    return None


def write_pid_file(pid):
    """Write PID to file."""
    with open(DAEMON_PID_FILE, 'w') as f:
        f.write(str(pid))


def stop_daemon():
    """Stop the running daemon."""
    logger = setup_logging()
    pid = get_daemon_pid()
    
    if pid is None:
        logger.info("No daemon process found")
        # Clean up PID file if it exists (stale file)
        if DAEMON_PID_FILE.exists():
            try:
                DAEMON_PID_FILE.unlink()
                logger.info("Removed stale PID file")
            except Exception as e:
                logger.warning(f"Could not remove stale PID file: {e}")
        return False
    
    try:
        proc = psutil.Process(pid)
        logger.info(f"Stopping daemon (PID: {pid})...")
        proc.terminate()
        
        # Wait for graceful shutdown
        try:
            proc.wait(timeout=10)
        except psutil.TimeoutExpired:
            logger.warning("Daemon didn't terminate gracefully, forcing...")
            proc.kill()
            proc.wait(timeout=5)  # Wait a bit more after kill
        
        # Always delete PID file after stopping process
        if DAEMON_PID_FILE.exists():
            DAEMON_PID_FILE.unlink()
            logger.info("Removed PID file")
        
        logger.info("Daemon stopped successfully")
        return True
    except psutil.NoSuchProcess:
        logger.info("Daemon process not found (may have already stopped)")
        # Always delete PID file if process doesn't exist
        if DAEMON_PID_FILE.exists():
            try:
                DAEMON_PID_FILE.unlink()
                logger.info("Removed stale PID file")
            except Exception as e:
                logger.warning(f"Could not remove stale PID file: {e}")
        return False
    except Exception as e:
        logger.error(f"Error stopping daemon: {e}")
        # Even on error, try to clean up PID file
        if DAEMON_PID_FILE.exists():
            try:
                DAEMON_PID_FILE.unlink()
                logger.info("Removed PID file after error")
            except Exception as cleanup_error:
                logger.warning(
                    f"Could not remove PID file after error: {cleanup_error}"
                )
        return False


def daemon_status():
    """Check daemon status."""
    logger = setup_logging()
    pid = get_daemon_pid()
    
    if pid is None:
        logger.info("Daemon is not running")
        return False
    
    try:
        proc = psutil.Process(pid)
        logger.info(f"Daemon is running (PID: {pid})")
        logger.info(f"Started: {datetime.fromtimestamp(proc.create_time())}")
        logger.info(
            f"CPU: {proc.cpu_percent():.1f}%, "
            f"Memory: {proc.memory_info().rss / 1024 / 1024:.1f} MB"
        )
        return True
    except psutil.NoSuchProcess:
        logger.info("Daemon is not running (stale PID file)")
        if DAEMON_PID_FILE.exists():
            DAEMON_PID_FILE.unlink()
        return False


def run_analysis_script(script_path: Path, method_name: str, logger):
    """Run a single analysis script with its own log file."""
    logger.info(f"Starting {method_name} analysis...")
    start_time = time.time()
    
    # Get the appropriate log file for this analysis
    log_file_map = {
        "MLE": MLE_LOG_FILE,
        "PhyloDeep": PHYLODEEP_LOG_FILE,
        "Bayesian": BAYESIAN_LOG_FILE
    }
    analysis_log_file = log_file_map.get(method_name, DAEMON_STDOUT)
    
    # Ensure log file directory exists
    analysis_log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Scripts need to run from src/analysis/ directory for relative imports
    # Also need project root in PYTHONPATH for config imports
    analysis_dir = Path(__file__).parent
    project_root = analysis_dir.parent.parent
    
    # Make script path relative to analysis_dir
    script_path_abs = Path(script_path).resolve()
    analysis_dir_abs = analysis_dir.resolve()
    script_path_rel = script_path_abs.relative_to(analysis_dir_abs)
    
    # Set up environment with PYTHONPATH
    # Need: project_root (for config), analysis_dir (for utils), and
    # src_dir (for mle/bayesian/phylodeep packages)
    src_dir = analysis_dir.parent
    env = os.environ.copy()
    pythonpath = env.get('PYTHONPATH', '')
    if pythonpath:
        env['PYTHONPATH'] = (
            f"{project_root}:{analysis_dir}:{src_dir}:{pythonpath}"
        )
    else:
        env['PYTHONPATH'] = f"{project_root}:{analysis_dir}:{src_dir}"
    
    # Try to use venv Python if available, otherwise use sys.executable
    # Check Python version compatibility (project requires <3.12)
    python_exe = sys.executable
    venv_python = project_root / ".venv" / "bin" / "python3"
    if venv_python.exists():
        # Check Python version
        try:
            import subprocess
            version_result = subprocess.run(
                [str(venv_python), '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            version_str = version_result.stdout.strip()
            # Extract version number (e.g., "Python 3.13.7" -> 3.13)
            import re
            version_match = re.search(r'(\d+)\.(\d+)', version_str)
            if version_match:
                major, minor = (
                    int(version_match.group(1)),
                    int(version_match.group(2))
                )
                if major == 3 and minor >= 12:
                    logger.warning(
                        f"[WARN] Venv Python {version_str} is incompatible "
                        f"(requires <3.12). Using sys.executable instead."
                    )
                else:
                    python_exe = str(venv_python)
            else:
                python_exe = str(venv_python)
        except Exception:
            # If version check fails, use venv Python anyway
            python_exe = str(venv_python)
    
    try:
        # Open log file for writing
        with open(analysis_log_file, 'w') as log_f:
            # Write header to log file
            log_f.write("=" * 80 + "\n")
            log_f.write(f"{method_name} Analysis Log\n")
            log_f.write(
                f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            log_f.write("=" * 80 + "\n\n")
            log_f.flush()
            
            # Run the script, redirecting both stdout and stderr to log file
            result = subprocess.run(
                [python_exe, str(script_path_rel)],
                cwd=
                    str(analysis_dir),
                        # Run from src/analysis/ for relative imports
                env=env,  # Include PYTHONPATH for config imports
                stdout=log_f,  # Redirect stdout to log file
                stderr=subprocess.STDOUT,  # Redirect stderr to same file
                text=True
            )
        
        elapsed = time.time() - start_time
        
        # Append completion info to log file
        with open(analysis_log_file, 'a') as log_f:
            log_f.write("\n" + "=" * 80 + "\n")
            log_f.write(
                f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            log_f.write(f"Exit code: {result.returncode}\n")
            log_f.write(f"Elapsed time: {elapsed:.2f} seconds\n")
            log_f.write("=" * 80 + "\n")
        
        if result.returncode == 0:
            logger.info(
                f"[OK] {method_name} analysis completed successfully "
                f"({elapsed:.1f}s, log: {analysis_log_file.name})"
            )
            return True, None, elapsed
        else:
            # Read last part of log for error message
            try:
                with open(analysis_log_file, 'r') as log_f:
                    log_lines = log_f.readlines()
                    error_msg = ''.join(log_lines[-20:])  # Last 20 lines
                    if len(error_msg) > 500:
                        error_msg = error_msg[:500]
            except Exception:
                error_msg = f"Exit code {result.returncode}"
            logger.error(
                f"[FAIL] {method_name} analysis failed "
                f"(log: {analysis_log_file.name})"
            )
            return False, error_msg, elapsed
            
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        with open(analysis_log_file, 'a') as log_f:
            log_f.write(f"\nERROR: Timeout after {elapsed:.2f} seconds\n")
        logger.error(
            f"[FAIL] {method_name} timed out "
            f"(log: {analysis_log_file.name})"
        )
        return False, "Timeout", elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        with open(analysis_log_file, 'a') as log_f:
            log_f.write(f"\nERROR: {str(e)}\n")
        logger.error(
            f"[FAIL] {method_name} analysis error: {str(e)} "
            f"(log: {analysis_log_file.name})"
        )
        return False, str(e), elapsed


def run_comparison(logger):
    """Run the comparison script."""
    logger.info("=" * 80)
    logger.info("Starting comparison of results...")
    logger.info("=" * 80)
    
    script_dir = Path(__file__).parent
    compare_script = script_dir / "compare_results.py"
    # Comparison script also needs to run from src/analysis/ for imports
    analysis_dir = Path(__file__).parent
    project_root = analysis_dir.parent.parent
    
    # Make script path relative to analysis_dir
    compare_script_abs = compare_script.resolve()
    analysis_dir_abs = analysis_dir.resolve()
    compare_script_rel = compare_script_abs.relative_to(analysis_dir_abs)
    
    # Set up environment with PYTHONPATH
    # Need: project_root (for config), analysis_dir (for utils), and
    # src (for mle/bayesian/phylodeep packages)
    src_dir = analysis_dir.parent
    env = os.environ.copy()
    pythonpath = env.get('PYTHONPATH', '')
    if pythonpath:
        env['PYTHONPATH'] = (
            f"{project_root}:{analysis_dir}:{src_dir}:{pythonpath}"
        )
    else:
        env['PYTHONPATH'] = f"{project_root}:{analysis_dir}:{src_dir}"
    
    # Try to use venv Python if available, but check version compatibility
    python_exe = sys.executable
    venv_python = project_root / ".venv" / "bin" / "python3"
    if venv_python.exists():
        # Check Python version (project requires <3.12)
        try:
            import subprocess
            version_result = subprocess.run(
                [str(venv_python), '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            version_str = version_result.stdout.strip()
            import re
            version_match = re.search(r'(\d+)\.(\d+)', version_str)
            if version_match:
                major, minor = (
                    int(version_match.group(1)),
                    int(version_match.group(2))
                )
                if major == 3 and minor >= 12:
                    logger.warning(
                        f"[WARN] Venv Python {version_str} is incompatible "
                        f"(requires <3.12). Using sys.executable instead."
                    )
                else:
                    python_exe = str(venv_python)
            else:
                python_exe = str(venv_python)
        except Exception:
            python_exe = str(venv_python)
    
    # Ensure log file directory exists
    COMPARISON_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    try:
        # Open log file for writing
        with open(COMPARISON_LOG_FILE, 'w') as log_f:
            # Write header to log file
            log_f.write("=" * 80 + "\n")
            log_f.write("Comparison Analysis Log\n")
            log_f.write(
                f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            log_f.write("=" * 80 + "\n\n")
            log_f.flush()
            
            # Run the script, redirecting both stdout and stderr to log file
            result = subprocess.run(
                [python_exe, str(compare_script_rel)],
                cwd=
                    str(analysis_dir),
                        # Run from src/analysis/ for relative imports
                env=env,  # Include PYTHONPATH for config imports
                stdout=log_f,  # Redirect stdout to log file
                stderr=subprocess.STDOUT,  # Redirect stderr to same file
                text=True
            )
        
        elapsed = time.time() - start_time
        
        # Append completion info to log file
        with open(COMPARISON_LOG_FILE, 'a') as log_f:
            log_f.write("\n" + "=" * 80 + "\n")
            log_f.write(
                f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            log_f.write(f"Exit code: {result.returncode}\n")
            log_f.write(f"Elapsed time: {elapsed:.2f} seconds\n")
            log_f.write("=" * 80 + "\n")
        
        if result.returncode == 0:
            logger.info(
                f"[OK] Comparison completed successfully "
                f"({elapsed:.1f}s, log: {COMPARISON_LOG_FILE.name})"
            )
            return True
        else:
            logger.error(
                f"[FAIL] Comparison failed "
                f"(log: {COMPARISON_LOG_FILE.name})"
            )
            return False
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        with open(COMPARISON_LOG_FILE, 'a') as log_f:
            log_f.write(f"\nERROR: Exit code {e.returncode}\n")
            if e.stderr:
                log_f.write(f"Error: {e.stderr[:500]}\n")
        logger.error(
            f"[FAIL] Comparison failed: Exit code {e.returncode} "
            f"(log: {COMPARISON_LOG_FILE.name})"
        )
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        with open(COMPARISON_LOG_FILE, 'a') as log_f:
            log_f.write(f"\nERROR: {str(e)}\n")
        logger.error(
            f"[FAIL] Comparison error: {str(e)} "
            f"(log: {COMPARISON_LOG_FILE.name})"
        )
        return False


def cleanup_output_and_logs(project_root, logger):
    """
    Clean up output directories and log files before starting a new run.
    
    This ensures each run starts with a clean slate:
    - Deletes and recreates all output subdirectories
    - Clears all log files
    """
    from config import (
        OUTPUT_DIR,
        MLE_OUTPUT_DIR,
        PHYLODEEP_OUTPUT_DIR,
        BAYESIAN_OUTPUT_DIR,
        COMPARISON_OUTPUT_DIR
    )
    
    logger.info("=" * 80)
    logger.info("CLEANING UP OUTPUT AND LOGS")
    logger.info("=" * 80)
    
    # Clean up output directories
    output_dirs = [
        MLE_OUTPUT_DIR,
        PHYLODEEP_OUTPUT_DIR,
        BAYESIAN_OUTPUT_DIR,
        COMPARISON_OUTPUT_DIR
    ]
    
    for output_dir in output_dirs:
        if output_dir.exists():
            # Count files before deletion
            file_count = sum(1 for _ in output_dir.rglob('*') if _.is_file())
            dir_count = sum(1 for _ in output_dir.rglob('*') if _.is_dir())
            
            try:
                # Use shutil.rmtree for complete removal
                # (handles all files and subdirs)
                shutil.rmtree(output_dir)
                logger.info(
                    f"[OK] Deleted: {output_dir} "
                    f"({file_count} files, {dir_count} dirs)"
                )
            except PermissionError as e:
                logger.warning(
                    f"[WARN] Permission error deleting {output_dir}: {e}"
                )
                # Try to delete files individually
                try:
                    for item in output_dir.rglob('*'):
                        try:
                            if item.is_file():
                                item.unlink()
                            elif item.is_dir():
                                item.rmdir()
                        except Exception:
                            pass
                    # Try to remove the directory
                    output_dir.rmdir()
                    logger.info(f"[OK] Deleted (with retries): {output_dir}")
                except Exception as e2:
                    logger.error(
                        f"[FAIL] Could not delete {output_dir} even "
                        f"with retries: {e2}"
                    )
            except Exception as e:
                logger.warning(f"[WARN] Could not delete {output_dir}: {e}")
                # Force delete with ignore_errors
                try:
                    shutil.rmtree(output_dir, ignore_errors=True)
                    logger.info(f"[OK] Force deleted: {output_dir}")
                except Exception as e2:
                    logger.error(
                        f"[FAIL] Could not force delete {output_dir}: {e2}"
                    )
        
        # Recreate directory (fresh and empty)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            # Verify it's empty
            remaining = list(output_dir.iterdir())
            if remaining:
                logger.warning(
                    f"[WARN] {output_dir} is not empty after creation! "
                    f"({len(remaining)} items)"
                )
                # Try to remove remaining items
                for item in remaining:
                    try:
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                    except Exception:
                        pass
            else:
                logger.info(f"[OK] Created (empty): {output_dir}")
        except Exception as e:
            logger.error(f"[FAIL] Could not create {output_dir}: {e}")
    
    # Clean up ALL log files in daemon directory (remove unnecessary ones too)
    logger.info("\nCleaning up log files...")
    
    # First, remove all .log files in the daemon directory
    if DAEMON_DIR.exists():
        log_files_found = list(DAEMON_DIR.glob("*.log"))
        for log_file in log_files_found:
            try:
                log_file.unlink()
                logger.info(f"[OK] Removed: {log_file.name}")
            except Exception as e:
                logger.warning(f"[WARN] Could not remove {log_file.name}: {e}")
    
    # Define expected log files
    expected_log_files = [
        DAEMON_LOG_FILE,
        DAEMON_STDOUT,
        DAEMON_STDERR,
        MLE_LOG_FILE,
        PHYLODEEP_LOG_FILE,
        BAYESIAN_LOG_FILE,
        COMPARISON_LOG_FILE
    ]
    
    # Recreate expected log files (empty)
    for log_file in expected_log_files:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.touch()
            logger.info(f"[OK] Created: {log_file.name}")
        except Exception as e:
            logger.warning(f"[WARN] Could not create {log_file.name}: {e}")
    
    logger.info("=" * 80)
    logger.info("CLEANUP COMPLETE")
    logger.info("=" * 80)


def daemon_main():
    """Main daemon function."""
    # Get project root for cleanup
    project_root = get_project_root()
    
    # Setup initial logging (will be cleared and re-setup after cleanup)
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("DAEMON STARTED")
    logger.info("=" * 80)
    logger.info(f"PID: {os.getpid()}")
    logger.info(f"Log file: {DAEMON_LOG_FILE}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Clean up output directories and log files BEFORE starting
    cleanup_output_and_logs(project_root, logger)
    
    # Re-setup logging after clearing log files
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("DAEMON STARTED (after cleanup)")
    logger.info("=" * 80)
    
    # Write PID file
    write_pid_file(os.getpid())
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        if DAEMON_PID_FILE.exists():
            DAEMON_PID_FILE.unlink()
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        script_dir = Path(__file__).parent
        
        # Define all analysis scripts
        scripts = {
            "MLE": script_dir / "mle" / "mle_analysis.py",
            "PhyloDeep": script_dir / "phylodeep_method" /
                "phylodeep_analysis.py",
            "Bayesian": script_dir / "bayesian" / "bayesian_analysis.py",
        }
        
        # Run all analyses in parallel
        logger.info("\n" + "=" * 80)
        logger.info("Starting all analyses in parallel...")
        logger.info("=" * 80)
        
        analysis_start = time.time()
        results = {}
        
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    run_analysis_script, script, name, logger
                ): name
                for name, script in scripts.items()
            }
            
            for future in as_completed(futures):
                method_name = futures[future]
                success, error, elapsed = future.result()
                results[method_name] = (success, error, elapsed)
                
                if success:
                    logger.info(f"[OK] {method_name} completed")
                else:
                    if method_name == "Bayesian":
                        logger.warning(
                            f"[WARN] {method_name} failed (optional): {error}"
                        )
                    else:
                        logger.error(f"[FAIL] {method_name} failed: {error}")
        
        analysis_elapsed = time.time() - analysis_start
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("ANALYSIS SUMMARY")
        logger.info("=" * 80)
        successful = sum(1 for success, _, _ in results.values() if success)
        logger.info(f"Completed: {successful}/{len(results)} analyses")
        logger.info(f"Total analysis time: {analysis_elapsed:.1f}s")
        
        for method_name, (success, error, elapsed) in results.items():
            status = "[OK]" if success else "[FAIL]"
            logger.info(f"  {status} {method_name}: {elapsed:.1f}s")
        
        # Run comparison if at least one analysis succeeded
        if successful > 0:
            logger.info("\n" + "=" * 80)
            logger.info("Running comparison...")
            logger.info("=" * 80)
            
            comparison_success = run_comparison(logger)
            
            if comparison_success:
                logger.info("\n" + "=" * 80)
                logger.info("DAEMON COMPLETED SUCCESSFULLY")
                logger.info("=" * 80)
            else:
                logger.warning("\n" + "=" * 80)
                logger.warning("DAEMON COMPLETED WITH WARNINGS")
                logger.warning("=" * 80)
        else:
            logger.error("\n" + "=" * 80)
            logger.error(
                "No analyses completed successfully - skipping comparison"
            )
            logger.error("=" * 80)
        
    except KeyboardInterrupt:
        logger.info("Daemon interrupted by user")
    except Exception as e:
        logger.error(f"Daemon error: {e}", exc_info=True)
    finally:
        if DAEMON_PID_FILE.exists():
            DAEMON_PID_FILE.unlink()
        logger.info("Daemon stopped")


def start_daemon():
    """Start the daemon in background."""
    logger = setup_logging()
    
    # Check if already running
    if get_daemon_pid() is not None:
        logger.error("Daemon is already running!")
        daemon_status()
        return False
    
    logger.info("Starting daemon in background...")
    
    # Redirect stdout/stderr to files
    stdout_fd = open(DAEMON_STDOUT, 'a')
    stderr_fd = open(DAEMON_STDERR, 'a')
    
    # Start daemon process
    # Use nohup-like behavior for persistence
    try:
        # Fork process
        pid = os.fork()
        
        if pid > 0:
            # Parent process
            logger.info(f"Daemon started with PID: {pid}")
            logger.info(f"Logs: {DAEMON_LOG_FILE}")
            logger.info(f"Stdout: {DAEMON_STDOUT}")
            logger.info(f"Stderr: {DAEMON_STDERR}")
            logger.info("Daemon will continue running even if terminal closes")
            stdout_fd.close()
            stderr_fd.close()
            return True
        else:
            # Child process - become daemon
            os.setsid()  # Create new session
            
            # Second fork
            pid = os.fork()
            if pid > 0:
                os._exit(0)  # Exit parent
            
            # Daemon process
            # Change to analysis directory to preserve relative imports
            analysis_dir = Path(__file__).parent
            os.chdir(str(analysis_dir))
            os.umask(0)  # Reset file mode
            
            # Redirect standard file descriptors
            sys.stdout.flush()
            sys.stderr.flush()
            
            # Close file descriptors (except our log files)
            for fd in range(3):
                try:
                    os.close(fd)
                except OSError:
                    pass
            
            # Redirect stdin, stdout, stderr
            os.open('/dev/null', os.O_RDWR)  # stdin
            os.dup2(stdout_fd.fileno(), 1)  # stdout
            os.dup2(stderr_fd.fileno(), 2)  # stderr
            
            stdout_fd.close()
            stderr_fd.close()
            
            # Run main daemon function
            daemon_main()
            os._exit(0)
            
    except OSError as e:
        logger.error(f"Failed to fork daemon: {e}")
        stdout_fd.close()
        stderr_fd.close()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Daemon for running analyses and comparison"
    )
    parser.add_argument(
        'command',
        choices=['start', 'stop', 'status', 'restart'],
        help='Daemon command'
    )
    
    args = parser.parse_args()
    
    if args.command == 'start':
        start_daemon()
    elif args.command == 'stop':
        stop_daemon()
    elif args.command == 'status':
        daemon_status()
    elif args.command == 'restart':
        stop_daemon()
        time.sleep(2)
        start_daemon()


if __name__ == "__main__":
    main()

