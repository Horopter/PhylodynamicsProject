"""
Common utilities and imports for analysis modules.

Provides shared imports and utilities to reduce redundancy across files.

Author: Santosh Desai <santoshdesai12@hotmail.com>
"""

import warnings
import sys
from pathlib import Path

# Suppress warnings globally for analysis scripts
warnings.filterwarnings('ignore')


def setup_analysis_environment(script_file: str = None):
    """
    Set up the analysis environment: paths and warnings.

    Parameters:
    -----------
    script_file : str, optional
        Path to the script file (usually __file__)

    Returns:
    --------
    Path : Path to the project root
    """
    # Setup project paths - add root first, then import config
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Now we can import config
    from config import setup_project_paths
    return setup_project_paths(script_file)

