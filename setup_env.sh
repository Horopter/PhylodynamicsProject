#!/bin/bash
# Setup script for PhyloDeep Density Regression Analysis

echo "="*70
echo "Setting up environment for Density Regression Analysis"
echo "="*70

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Current Python: $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv_phylodeep" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv_phylodeep
fi

# Activate virtual environment
source venv_phylodeep/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install core packages
echo "Installing core packages..."
pip install numpy scipy scikit-learn pandas matplotlib seaborn ete3

# Try to install PhyloDeep
echo "Attempting to install PhyloDeep..."
if pip install phylodeep 2>&1 | grep -q "Successfully installed"; then
    echo "✓ PhyloDeep installed successfully"
else
    echo "⚠ PhyloDeep installation failed (may require Python 3.8-3.11)"
    echo "  See install_instructions.md for alternatives"
fi

# Verify installation
echo ""
echo "="*70
echo "Verification"
echo "="*70
python3 -c "import numpy, scipy, sklearn, pandas; print('✓ Core packages installed')" 2>/dev/null && echo "✓ Core packages OK" || echo "✗ Core packages missing"

python3 -c "import phylodeep; print('✓ PhyloDeep installed')" 2>/dev/null && echo "✓ PhyloDeep OK" || echo "⚠ PhyloDeep not available (will use synthetic data)"

echo ""
echo "="*70
echo "Setup complete!"
echo "="*70
echo "To activate environment: source venv_phylodeep/bin/activate"
echo "To run notebook: jupyter notebook density_regression_analysis.ipynb"

