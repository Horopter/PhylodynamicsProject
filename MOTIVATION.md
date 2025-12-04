# Why Density Regression for PhyloDeep's Problem?

## The Problem PhyloDeep Solves

PhyloDeep estimates parameters of phylodynamic models from phylogenetic trees:
- **BD**: Basic Birth-Death model (2 parameters: R_naught, Infectious_period)
- **BDEI**: Birth-Death with Exposed class (3 parameters: R_naught, Infectious_period, Incubation_period)
- **BDSS**: Birth-Death with Superspreading (4 parameters: R_naught, Infectious_period, X_transmission, SS_fraction)

## Current PhyloDeep Approach

**Output:**
- Point estimates for each parameter
- Bootstrap-based 95% confidence intervals (2.5% and 97.5% quantiles)

**How Bootstrap CIs Work:**
1. Pre-compute large database of simulated trees with known parameters
2. For new tree: Find KNN in database (based on tree size, sampling probability)
3. Use prediction errors from neighbors to estimate CI
4. CIs computed independently for each parameter

## Limitations of Current Approach

### 1. **No Parameter Correlations**

**Problem:** Parameters are estimated independently
- R_naught and Infectious_period are likely correlated
- BDSS has 4 parameters that may be correlated
- Cannot capture joint uncertainty

**Example:** 
- High R_naught might correlate with shorter Infectious_period
- Current approach: Two independent CIs
- Reality: Joint distribution shows correlation

**Impact:** 
- Underestimates uncertainty in joint scenarios
- Cannot propagate uncertainty properly in downstream analyses

### 2. **Pre-computed Database Dependency**

**Problem:** Requires large pre-computed database
- Limited to parameter ranges covered in training
- Cannot easily extend to novel parameter regimes
- Database must be maintained and updated

**Example:**
- New pathogen with unusual transmission patterns
- Parameters outside training database ranges
- Bootstrap CI method fails or gives unreliable results

**Impact:**
- Limited applicability to novel scenarios
- Cannot adapt to new phylodynamic models easily

### 3. **Assumes Unimodal Distributions**

**Problem:** Bootstrap CIs assume single mode
- Some trees might have multi-modal parameter distributions
- Cannot represent ambiguity between multiple plausible scenarios

**Example:**
- Tree could be explained by either:
  - High R_naught + low Infectious_period, OR
  - Low R_naught + high Infectious_period
- Bootstrap CI: Single interval (misses multi-modality)

**Impact:**
- Loss of information about parameter ambiguity
- May mislead decision-making

### 4. **Computational Cost**

**Problem:** KNN search over large database
- Must search through pre-computed database for each tree
- Scales poorly with database size
- Not suitable for real-time or large-scale analysis

**Impact:**
- Slower inference
- Higher memory requirements

## How Density Regression Addresses These Limitations

### 1. **Captures Parameter Correlations**

**Solution:** Joint probability distributions
- Models all parameters simultaneously
- Captures correlations through covariance structure
- Enables proper uncertainty propagation

**Example:**
- GP or MDN outputs joint distribution: p(R_naught, Infectious_period, X_transmission, SS_fraction | tree)
- Can sample correlated parameter sets
- Properly quantifies joint uncertainty

### 2. **No Pre-computed Database**

**Solution:** Learns distributions from training data
- Once trained, no database needed
- Can extrapolate (with uncertainty) to novel regimes
- Easier to extend to new models

**Example:**
- Train on simulated trees with known parameters
- Model learns mapping: tree features → parameter distribution
- Inference: Single forward pass, no database search

### 3. **Multi-modal Distributions**

**Solution:** Mixture models (e.g., Mixture Density Networks)
- Can represent multi-modal distributions
- Captures parameter ambiguity
- Provides richer uncertainty information

**Example:**
- MDN with 5 components can represent 5 different plausible scenarios
- Each component has its own mean and variance
- Weighted mixture captures multi-modality

### 4. **Fast Inference**

**Solution:** Single forward pass
- Once trained, inference is very fast
- No database search required
- Suitable for real-time and large-scale analysis

**Example:**
- PhyloDeep Bootstrap: ~seconds (database search)
- Density Regression: ~milliseconds (single forward pass)
- 100-1000x faster for large-scale studies

## When Density Regression is Most Valuable

### ✅ High Value Scenarios:

1. **BDSS Model** (4 correlated parameters)
   - Most benefit from capturing correlations
   - Joint uncertainty quantification critical

2. **Novel Pathogens**
   - Parameters outside training database
   - Need extrapolation with uncertainty

3. **Decision-Making**
   - Need full distributions for risk assessment
   - Not just point estimates + CIs

4. **Large-Scale Studies**
   - Many trees to analyze
   - Computational efficiency matters

5. **Model Comparison**
   - Comparing BD vs BDEI vs BDSS
   - Need proper uncertainty quantification

### ⚠️ When PhyloDeep's Approach is Sufficient:

1. **Single Tree Analysis**
   - One tree, known parameter ranges
   - Point estimate + CI sufficient

2. **Well-Covered Parameter Space**
   - Parameters within training database
   - Bootstrap CIs reliable

3. **Simple Models (BD)**
   - Only 2 parameters, less correlation
   - Independent CIs less problematic

## Specific Example: BDSS Model

**BDSS Parameters:**
- R_naught: Basic reproduction number
- Infectious_period: Duration of infectiousness
- X_transmission: Transmission rate multiplier
- SS_fraction: Fraction of superspreaders

**Why Correlations Matter:**
- High SS_fraction might correlate with high X_transmission
- R_naught depends on both transmission rates and infectious period
- Cannot understand these relationships with independent CIs

**Density Regression Solution:**
- Joint distribution: p(R_naught, Infectious_period, X_transmission, SS_fraction | tree)
- Captures all correlations
- Enables proper uncertainty quantification
- Supports downstream Bayesian analysis

## Conclusion

Density regression is **highly applicable** to PhyloDeep's problem, especially for:
- **BDSS model** with correlated parameters
- **Novel scenarios** outside training database
- **Decision-making** requiring full distributions
- **Large-scale studies** needing computational efficiency

The key advantage is moving from **independent point estimates + CIs** to **joint probability distributions** that properly capture uncertainty and parameter relationships.

