"""
Bayesian MCMC estimation module.

Contains:
- BayesianBirthDeath: Core Bayesian implementation class
- bayesian_analysis: Batch processing script
- HAS_PYMC: Boolean indicating if PyMC is available

Author: Santosh Desai <santoshdesai12@hotmail.com>
"""

from .bayesian_birth_death import BayesianBirthDeath, HAS_PYMC

__all__ = ['BayesianBirthDeath', 'HAS_PYMC']
