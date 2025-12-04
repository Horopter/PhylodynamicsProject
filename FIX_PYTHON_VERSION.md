# Fix Python Version Issue

## Problem
Your virtual environment is using Python 3.13, but the project requires Python <3.12 because:
- `ete3` library doesn't support Python 3.13 (tries to import `cgi` module which was removed)
- Project specification: `requires-python = ">=3.8,<3.12"`

## Solution: Recreate Virtual Environment

### Step 1: Deactivate and Remove Old Venv
```bash
# If venv is currently active
deactivate

# Remove the old venv
rm -rf .venv
```

### Step 2: Create New Venv with Python 3.11
```bash
# Create new venv with Python 3.11
python3.11 -m venv .venv

# Activate the new venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Install core dependencies
pip install -r requirements.txt

# Optionally install Bayesian dependencies
pip install -r requirements-bayesian.txt
```

### Step 4: Verify Installation
```bash
# Check Python version
python --version
# Should show: Python 3.11.x

# Test imports
python -c "import ete3; print('ete3 works!')"
python -c "import phylodeep; print('phylodeep works!')"
```

### Step 5: Restart Daemon
```bash
# Start the daemon
python src/analysis/daemon_runner.py start

# Monitor logs
tail -f output/daemon/daemon.log
```

## Alternative: Use System Python 3.11
If you prefer to use system Python 3.11 directly (without venv):
```bash
# Make sure Python 3.11 has all packages
python3.11 -m pip install -r requirements.txt
```

## Why This Happened
- Python 3.13 removed the `cgi` module
- `ete3` library still tries to import it
- This is an upstream compatibility issue with `ete3`
- The project correctly specifies Python <3.12 requirement

