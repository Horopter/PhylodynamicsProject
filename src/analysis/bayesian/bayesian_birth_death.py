"""
Bayesian MCMC estimation for Birth-Death model parameters.

I implemented this to get posterior distributions for birth-death parameters
using MCMC sampling. The key advantage over MLE is that we get full uncertainty
quantification - not just point estimates but credible intervals.

Uses PyMC for the MCMC sampling and the Stadler (2010) likelihood - same as MLE
but with Bayesian inference instead of optimization.
"""

import numpy as np
from ete3 import Tree
import warnings

warnings.filterwarnings('ignore')

# Try to import PyMC - make it optional
try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False
    pm = None
    az = None

# Export HAS_PYMC so other modules can check
__all__ = ['BayesianBirthDeath', 'HAS_PYMC']


class BayesianBirthDeath:
    """
    Bayesian MCMC estimation for Birth-Death model parameters.

    I built this to complement the MLE approach - gives us posterior distributions
    instead of just point estimates. Uses PyMC for MCMC and Stadler (2010) likelihood.
    Works with tree data only - no sequences needed.
    """

    def __init__(self, tree_file, sampling_prob=0.5):
        """
        Initialize Bayesian estimator.

        Parameters:
        -----------
        tree_file : str
            Path to tree file in Newick format
        sampling_prob : float
            Sampling probability (default: 0.5)
        """
        if not HAS_PYMC:
            raise ImportError(
                "PyMC is required for Bayesian estimation. "
                "Install with: pip install pymc arviz"
            )

        self.tree_file = tree_file
        self.tree = Tree(tree_file, format=1)
        self.sampling_prob = sampling_prob

        # Extract the same tree stats as MLE - need topology and branch lengths
        self.n_tips = len(self.tree.get_leaves())
        self.branching_times = self._extract_branching_times()
        self.tree_height = max(self.branching_times) if self.branching_times else 0.0
        self.branch_lengths = [
            node.dist
            for node in self.tree.traverse()
            if node.up is not None and node.dist is not None and node.dist > 0
        ]
        self.total_branch_length = (
            float(np.sum(self.branch_lengths)) if self.branch_lengths else 0.0
        )
        self.mean_branch_length = (
            float(np.mean(self.branch_lengths)) if self.branch_lengths else 0.0
        )

        if self.n_tips <= 1 or self.total_branch_length <= 0:
            raise ValueError("Tree has insufficient information for parameter estimation")

    def _extract_branching_times(self):
        """
        Extract all branching times from the tree - same method as MLE.
        Just getting node ages to compute tree height.
        """
        times = []

        # Get all node distances from root
        for node in self.tree.traverse():
            if node != self.tree:
                dist = node.get_distance(self.tree)
                times.append(dist)

        # Also including root time (0)
        times.append(0.0)

        # Sorting in descending order (most recent first)
        return sorted(set(times), reverse=True)

    def _stadler_loglikelihood(self, lambda_val, mu_val):
        """
        Stadler (2010) log-likelihood - same formula as MLE.

        Important: This only needs tree topology and branch lengths. No nucleotide
        sequences at all. The likelihood models the birth-death process that created
        the tree structure.

        Parameters:
        -----------
        lambda_val : float or array
            Birth rate(s)
        mu_val : float or array
            Death rate(s)

        Returns:
        --------
        float or array
            Log-likelihood value(s)
        """
        # Use PyTensor operations for compatibility with PyMC
        import pytensor.tensor as pt

        n = self.n_tips
        T_total = self.total_branch_length
        rho = self.sampling_prob

        # Stadler likelihood formula - only uses tree structure, no sequences
        # n = number of tips (topology info)
        # T_total = sum of all branch lengths
        # rho = sampling probability
        # Use PyTensor operations instead of NumPy for compatibility
        log_likelihood = (
            (n - 1) * pt.log(lambda_val)      # branching events term
            - (lambda_val + mu_val) * T_total # branch length term
            + n * pt.log(rho)                  # sampling term
        )

        return log_likelihood

    def estimate_bd(self, draws=2000, tune=1000, chains=4,
                    lambda_prior='uniform', mu_prior='uniform',
                    return_inference_data=False):
        """
        Estimate BD parameters using MCMC sampling.

        This gives us posterior distributions instead of just point estimates.
        Can adjust the number of samples and chains based on how much time we have.

        Parameters:
        -----------
        draws : int
            Number of MCMC samples (default: 2000)
        tune : int
            Burn-in samples (default: 1000)
        chains : int
            Number of chains for diagnostics (default: 2)
        lambda_prior : str
            Prior type: 'uniform' or 'exponential' (default: 'uniform')
        mu_prior : str
            Prior type: 'uniform' or 'exponential' (default: 'uniform')
        return_inference_data : bool
            If True, return full ArviZ object for diagnostics (default: False)

        Returns:
        --------
        dict : Results with posterior means, medians, credible intervals
        """
        if not HAS_PYMC:
            return {
                "success": False,
                "message": "PyMC is not installed. Install with: pip install pymc arviz",
            }

        try:
            with pm.Model() as model:
                # Set up priors - uniform seems reasonable for birth-death rates
                if lambda_prior == 'uniform':
                    lambda_param = pm.Uniform('lambda', lower=0.01, upper=10.0)
                elif lambda_prior == 'exponential':
                    lambda_param = pm.Exponential('lambda', lam=1.0)
                else:
                    raise ValueError(f"Unknown lambda_prior: {lambda_prior}")

                if mu_prior == 'uniform':
                    # Need mu < lambda for positive net diversification
                    # Use a simpler approach: set upper bound to a fixed value
                    # and enforce constraint via Potential using simple arithmetic
                    mu_param = pm.Uniform('mu', lower=0.01, upper=9.99)
                    # Constraint: mu must be less than lambda
                    # Use simple arithmetic penalty instead of switch to avoid compatibility issues
                    diff = lambda_param - mu_param
                    # Large penalty if diff is negative or very small (mu >= lambda)
                    # Use pm.math.maximum to avoid switch
                    penalty = -1e10 * pm.math.maximum(0.0, 1e-6 - diff)
                    pm.Potential('mu_lt_lambda', penalty)
                elif mu_prior == 'exponential':
                    mu_param = pm.Exponential('mu', lam=1.0)
                    # Constraint: mu must be less than lambda
                    diff = lambda_param - mu_param
                    penalty = -1e10 * pm.math.maximum(0.0, 1e-6 - diff)
                    pm.Potential('mu_lt_lambda', penalty)
                else:
                    raise ValueError(f"Unknown mu_prior: {mu_prior}")

                # Likelihood - Stadler formula using tree data only
                loglik = pm.Potential(
                    'loglik',
                    self._stadler_loglikelihood(lambda_param, mu_param)
                )

                # Run MCMC
                trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    return_inferencedata=True,
                    progressbar=False
                )

            # Get posterior samples
            posterior = trace.posterior

            lambda_samples = posterior['lambda'].values.flatten()
            mu_samples = posterior['mu'].values.flatten()

            # Compute summary statistics from posterior
            lambda_mean = float(np.mean(lambda_samples))
            lambda_std = float(np.std(lambda_samples))
            lambda_median = float(np.median(lambda_samples))
            lambda_q2_5 = float(np.percentile(lambda_samples, 2.5))
            lambda_q97_5 = float(np.percentile(lambda_samples, 97.5))

            mu_mean = float(np.mean(mu_samples))
            mu_std = float(np.std(mu_samples))
            mu_median = float(np.median(mu_samples))
            mu_q2_5 = float(np.percentile(mu_samples, 2.5))
            mu_q97_5 = float(np.percentile(mu_samples, 97.5))

            # Compute derived quantities from posterior
            R0_samples = lambda_samples / mu_samples
            R0_mean = float(np.mean(R0_samples))
            R0_std = float(np.std(R0_samples))
            R0_median = float(np.median(R0_samples))
            R0_q2_5 = float(np.percentile(R0_samples, 2.5))
            R0_q97_5 = float(np.percentile(R0_samples, 97.5))

            infectious_period_samples = 1.0 / mu_samples
            infectious_period_mean = float(np.mean(infectious_period_samples))
            infectious_period_std = float(np.std(infectious_period_samples))

            # Get MCMC diagnostics using ArviZ
            try:
                # Get ESS (Effective Sample Size) for key parameters
                ess_data = az.ess(trace)
                lambda_ess = float(ess_data['lambda'].values) if 'lambda' in ess_data else None
                mu_ess = float(ess_data['mu'].values) if 'mu' in ess_data else None
                
                # Get R-hat (convergence diagnostic)
                rhat_data = az.rhat(trace)
                lambda_rhat = float(rhat_data['lambda'].values) if 'lambda' in rhat_data else None
                mu_rhat = float(rhat_data['mu'].values) if 'mu' in rhat_data else None
                
                # Count divergences from sample_stats
                n_divergences = None
                try:
                    sample_stats = trace.sample_stats
                    if 'diverging' in sample_stats:
                        diverging = sample_stats['diverging'].values
                        n_divergences = int(np.sum(diverging))
                    else:
                        n_divergences = 0
                except Exception:
                    n_divergences = None
            except Exception as e:
                # If diagnostics fail, set to None
                lambda_ess = None
                mu_ess = None
                lambda_rhat = None
                mu_rhat = None
                n_divergences = None

            result = {
                "success": True,
                "method": "Bayesian MCMC (PyMC)",

                # Lambda posterior stats
                "lambda_mean": lambda_mean,
                "lambda_median": lambda_median,
                "lambda_std": lambda_std,
                "lambda_q2_5": lambda_q2_5,
                "lambda_q97_5": lambda_q97_5,

                # Mu posterior stats
                "mu_mean": mu_mean,
                "mu_median": mu_median,
                "mu_std": mu_std,
                "mu_q2_5": mu_q2_5,
                "mu_q97_5": mu_q97_5,

                # R0 from posterior (R_naught)
                "R_naught_mean": R0_mean,
                "R_naught_median": R0_median,
                "R_naught_std": R0_std,
                "R_naught_q2_5": R0_q2_5,
                "R_naught_q97_5": R0_q97_5,

                # Infectious period
                "Infectious_period_mean": infectious_period_mean,
                "Infectious_period_std": infectious_period_std,

                # MCMC diagnostics
                "lambda_ess": lambda_ess,
                "mu_ess": mu_ess,
                "lambda_rhat": lambda_rhat,
                "mu_rhat": mu_rhat,
                "n_divergences": n_divergences,

                # MCMC info
                "n_samples": len(lambda_samples),
                "n_chains": chains,
                "n_draws": draws,
                "n_tune": tune,
            }

            if return_inference_data:
                result["inference_data"] = trace

            return result

        except Exception as e:
            return {
                "success": False,
                "message": f"Error during MCMC sampling: {str(e)}",
            }


# Note: EvoVGM would need sequence alignments, not just trees. Since we're working
# with tree-only data, I'm using BayesianBirthDeath which works with the Stadler
# likelihood on tree topology and branch lengths.

