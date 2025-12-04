# Combined Talking Points: PhyloDeep and MLE (7-9 Minutes)

## Overview Section (1 minute)
- **Introduction**: "We compare two methods for birth-death tree parameter estimation: PhyloDeep (deep learning) and MLE (classical maximum likelihood estimation)"
- **Key Question**: "How do these methods compare in terms of accuracy, speed, and applicability across different tree sizes?"
- **Context**: "Each method has unique strengths - we'll see when to use which method"

---

## PhyloDeep Section (2-3 Minutes)

### Slide 1: PhyloDeep: How We Used the Package (1 minute)
- **Pre-trained Models Used**:
  - "We ONLY estimated BD (Birth-Death) trees - used PhyloDeep's BD models exclusively"
  - "Model Type: BD pre-trained models from PhyloDeep package"
- **Model Selection by Tree Size**:
  - "Medium trees (50 ≤ n < 200): PhyloDeep automatically uses BD models trained on medium-sized trees"
  - "Large trees (n ≥ 200): PhyloDeep automatically uses BD models trained on large trees"
  - "Automatic selection: PhyloDeep's paramdeep function selects the appropriate model based on tree size"
  - "We used both medium and large tree models - selection was automatic"
- **Implementation**:
  - "Called paramdeep(tree_file, sampling_prob=0.5, model='BD', vector_representation='FULL')"
  - "Used FULL (CNN-based CBLV) representation, not SUMSTATS"
  - "Set ci_computation=False for point estimates only"

### Slide 2: PhyloDeep: Key Features (45 seconds)
- **Pre-trained BD Models**:
  - "Used existing BD models trained on 3.9 million trees (Voznica et al.)"
  - "Fast inference: milliseconds per tree"
  - "Automatic model selection by tree size"
- **FULL Tree Representation**:
  - "CNN-based CBLV representation"
  - "Converts tree to fixed-size vector for neural network processing"
- **Output**:
  - "Point estimates: λ, μ, R₀, infectious period"

### Slide 3: PhyloDeep: Advantages and Limitations (30 seconds)
- **Advantages**:
  - "✓ Lower RMSE than MLE on large trees (n ≥ 50)"
  - "✓ Extremely fast inference (milliseconds)"
  - "✓ Trained on massive dataset (3.9M trees)"
- **Limitations**:
  - "✗ Requires minimum 50 tips (cannot handle small trees)"
  - "✗ Black-box model (less interpretable)"
  - "✗ Point estimates only in our implementation"
- **Best Use Case**: "Large trees (n ≥ 50) where speed and accuracy are priorities"

### Slide 4: Method Comparison: PhyloDeep vs MLE (30 seconds)
- **Key Differences from Table**:
  - **Approach**: PhyloDeep (deep learning) vs MLE (classical stats)
  - **Training**: PhyloDeep (pre-trained 3.9M) vs MLE (none)
  - **Optimization**: PhyloDeep (neural network) vs MLE (single L-BFGS-B)
  - **Min Size**: PhyloDeep (50 tips) vs MLE (none)
  - **Speed**: PhyloDeep (milliseconds) vs MLE (seconds)
  - **Accuracy**: PhyloDeep (best on large) vs MLE (baseline)
  - **Interpretable**: PhyloDeep (no) vs MLE (yes)

### When to Use:
- **Small (n < 50)**: MLE only (PhyloDeep unavailable)
- **Medium (50 ≤ n < 200)**: PhyloDeep (speed) or MLE (interpretability)
- **Large (n ≥ 200)**: PhyloDeep (accuracy)

---

## MLE Section (2-3 Minutes)

### Slide 1: Maximum Likelihood Estimation: Overview (1 minute)
- **Introduction**: "MLE uses numerical optimization of the Stadler (2010) likelihood to estimate birth and death rates"
- **Key Feature**: "Works with tree data only - no nucleotide sequences required"
- **Implementation**: "Uses scipy.optimize.minimize with L-BFGS-B algorithm"
- **Advantages**:
  - "Statistically principled (maximum likelihood)"
  - "Works on small trees (no minimum tip size requirement)"
  - "Provides point estimates with optimization diagnostics"
  - "Fast computation (seconds per tree)"
  - "Based on well-established Stadler (2010) theory"

### Slide 2: MLE Implementation: Key Features (1 minute)
- **Survival Probability Term**: "Includes -r·T term to help identify μ separately from λ"
- **Regularization for Large Trees**: "Penalty prevents μ from getting stuck at lower bound"
- **Robust Error Handling**: "Checks for invalid trees, handles optimization failures gracefully"
- **Method-of-Moments Initialization**: "Smart starting values ensure faster convergence"

### Slide 3: MLE: Advantages and Limitations (45 seconds)
- **Advantages**:
  - "✓ Statistically principled (maximum likelihood)"
  - "✓ No minimum tree size requirement"
  - "✓ Works on small trees (n < 50)"
  - "✓ Provides optimization diagnostics"
  - "✓ Fast computation (seconds per tree)"
  - "✓ Based on well-established Stadler (2010) theory"
- **Limitations**:
  - "✗ Higher RMSE than PhyloDeep on large trees"
  - "✗ Point estimates only (no uncertainty quantification)"
  - "✗ Requires careful initialization for convergence"
  - "✗ Can struggle with very large trees (n > 500)"
- **Best Use Case**: "Small to medium trees where statistical rigor is important"

---

## Combined Results Discussion (2-3 Minutes)

### Two-Method Comparison by Tree Size
- **Small Trees (n < 50)**:
  - "Only MLE works reliably"
  - "PhyloDeep cannot handle these trees"
  - "MLE provides statistically principled estimates"
- **Medium Trees (50 ≤ n < 200)**:
  - "PhyloDeep has lower RMSE on average"
  - "PhyloDeep RMSE: λ=0.056, μ=0.047 vs MLE: λ=0.234, μ=0.281"
  - "MLE provides interpretability and statistical rigor"
- **Large Trees (n ≥ 200)**:
  - "PhyloDeep has lower RMSE on average"
  - "PhyloDeep RMSE: λ=0.034, μ=0.028 vs MLE: λ=0.237, μ=0.262"
  - "PhyloDeep is fastest"
  - "MLE provides baseline with statistical rigor"

### Statistical Efficiency Summary
- **PhyloDeep's Advantage**: "Training on 3.9M trees enables better accuracy on average at all tree sizes where both methods apply"
- **MLE's Strength**: "Statistically principled, works on all tree sizes including small trees where PhyloDeep cannot be used"
- **Takeaway**: "PhyloDeep performs better on average at all tree sizes (n ≥ 50), but MLE is the only option for small trees (n < 50)"

### Method Selection Guide
- **Use PhyloDeep when**:
  - "Tree has n ≥ 50 tips"
  - "Speed is critical"
  - "Maximum accuracy needed on large trees"
- **Use MLE when**:
  - "Tree has n < 50 tips (PhyloDeep unavailable)"
  - "Need statistical rigor and interpretability"
  - "Want optimization diagnostics"

---

## Bayesian MCMC Section (1 minute)

### Slide: Bayesian MCMC: Attempted Analysis (45 seconds)
- **What We Attempted**:
  - "Implemented Bayesian MCMC using PyMC for birth-death parameter estimation"
  - "Used same Stadler (2010) likelihood as MLE"
  - "Sequential processing to avoid PyTensor cache conflicts"
  - "Goal: Full posterior distributions with uncertainty quantification"
- **Initial Results**:
  - "Successfully analyzed 3,550 trees (out of 8,655 total)"
  - "Tree size range: 10-181 tips"
  - "Success rate: 99.4% (3,530/3,550 trees)"
- **Why We Did Not Pursue Further**:
  - "RMSE was not better than MLE or PhyloDeep"
  - "Computationally intensive (minutes per tree)"
  - "Sequential processing too slow for full dataset"
  - "No clear advantage to justify computational cost"
- **Conclusion**: "Bayesian MCMC provides uncertainty quantification but does not improve RMSE, so we focused on MLE and PhyloDeep for the main analysis."

## Total Time: ~7-9 minutes

### Tips for Delivery:

#### PhyloDeep Section:
1. **Focus on HOW we used the package** - We used pre-trained BD models, both medium and large tree models
2. **Emphasize automatic model selection** - PhyloDeep automatically selects medium models for 50-199 tips, large models for ≥200 tips
3. **Be clear about what we did** - ONLY estimated BD trees, used BD models exclusively
4. **Implementation details** - Called paramdeep with model='BD', vector_representation='FULL', ci_computation=False

#### MLE Section:
1. **Emphasize statistical rigor** - Based on well-established Stadler (2010) theory
2. **Highlight the small tree advantage** - fills gap where PhyloDeep fails
3. **Explain the implementation** - L-BFGS-B optimization with regularization
4. **Connect to classical statistics** - maximum likelihood estimation

#### Combined Discussion:
1. **Show method selection logic** - when to use which method
2. **Acknowledge trade-offs** - no method is perfect
3. **Highlight complementarity** - methods work together, not in competition
4. **Emphasize each method's niche** - PhyloDeep for large trees, MLE for small trees

### Potential Questions:

#### PhyloDeep Questions:
- **Q**: "Why does PhyloDeep require 50 tips minimum?"
  - **A**: "The pre-trained BD models were trained on trees with at least 50 tips. Smaller trees have different statistical properties that the models haven't learned. We excluded 1,722 trees with n < 50 from our PhyloDeep analysis."

- **Q**: "Which PhyloDeep models did you use?"
  - **A**: "We used the pre-trained BD (Birth-Death) models with FULL representation. PhyloDeep automatically selected the appropriate model based on tree size - medium tree models for 50-199 tips, large tree models for ≥200 tips."

- **Q**: "Did you compute confidence intervals?"
  - **A**: "No, we set ci_computation=False to get point estimates only. This was faster and sufficient for our comparison with MLE."

- **Q**: "How does PhyloDeep select models?"
  - **A**: "PhyloDeep's paramdeep function automatically selects the appropriate pre-trained model based on tree size. We only specified BD model type and FULL representation - the model selection was automatic."

#### MLE Questions:
- **Q**: "Why use MLE when PhyloDeep is more accurate?"
  - **A**: "MLE works on small trees (n < 50) where PhyloDeep cannot be used. Also, MLE provides statistical rigor and interpretability that PhyloDeep's black-box model cannot offer."

- **Q**: "How does MLE handle large trees?"
  - **A**: "MLE uses regularization to prevent parameters from collapsing to boundaries in large trees. While it may have higher RMSE than PhyloDeep on large trees, it still provides statistically principled estimates."

- **Q**: "What are the advantages of MLE?"
  - **A**: "MLE is statistically principled, works on all tree sizes, provides optimization diagnostics, and is based on well-established theory. It's also interpretable unlike PhyloDeep's neural network."

#### Comparison Questions:
- **Q**: "Which method should I use?"
  - **A**: "Depends on your needs: Small trees (n < 50) → MLE only. Large trees (n ≥ 50) → PhyloDeep for speed/accuracy. Medium trees → Both methods are viable, choose based on speed vs statistical rigor."

- **Q**: "Why not always use PhyloDeep since it's most accurate?"
  - **A**: "PhyloDeep only works on large trees (n ≥ 50). For small trees, MLE is the only option. Also, MLE provides statistical rigor and interpretability that PhyloDeep cannot."

- **Q**: "How do the methods compare?"
  - **A**: "PhyloDeep has better RMSE on large trees due to training on 3.9M trees, but MLE works on all tree sizes and provides statistical rigor. Each method has its niche."

#### Bayesian Questions:
- **Q**: "Why didn't you complete the Bayesian analysis?"
  - **A**: "We attempted Bayesian MCMC and successfully analyzed 3,550 trees, but the RMSE was not better than MLE or PhyloDeep. Given the computational cost (minutes per tree) and no clear advantage, we focused on MLE and PhyloDeep for the main analysis."

- **Q**: "What did Bayesian MCMC provide?"
  - **A**: "Bayesian MCMC provides full posterior distributions with uncertainty quantification, which is valuable. However, for our goal of comparing RMSE, it did not outperform the other methods, so we did not pursue it further."
