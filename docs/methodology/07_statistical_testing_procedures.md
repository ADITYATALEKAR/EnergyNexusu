# Statistical Testing and Validation Procedures

**Author:** Aditya Talekar (ec24018@qmul.ac.uk)  
**Project:** EnergyNexus - Advanced LSTM Architectures for Energy Forecasting  
**Date:** 2024-2025  
**Supervisor:** Saqib Iqbal  

## Table of Contents
1. [Hypothesis Testing Framework](#hypothesis-testing-framework)
2. [Significance Testing Procedures](#significance-testing-procedures)
3. [Multiple Comparison Corrections](#multiple-comparison-corrections)
4. [Implementation Guidelines](#implementation-guidelines)
5. [Code Examples](#code-examples)

---

## Hypothesis Testing Framework

### 1. Research Hypotheses

#### 1.1 Primary Hypotheses

**H₁: Advanced LSTM Architectures Performance**
- **H₁₀ (Null):** Advanced LSTM architectures (attention, encoder-decoder, multivariate, ensemble) perform no better than baseline LSTM in energy forecasting accuracy
- **H₁₁ (Alternative):** At least one advanced LSTM architecture significantly outperforms baseline LSTM in energy forecasting accuracy
- **Metric:** Mean Absolute Error (MAE) across forecast horizons
- **Significance Level:** α = 0.05

**H₂: Multi-variate vs Single-variate Forecasting**
- **H₂₀ (Null):** Multi-variate LSTM forecasting provides no improvement over single-variate forecasting for energy demand prediction
- **H₂₁ (Alternative):** Multi-variate LSTM forecasting significantly improves energy demand prediction accuracy
- **Metric:** RMSE for energy demand forecasting
- **Significance Level:** α = 0.05

**H₃: Ensemble Uncertainty Quantification**
- **H₃₀ (Null):** Ensemble LSTM provides no reliable uncertainty quantification (coverage rate = random)
- **H₃₁ (Alternative):** Ensemble LSTM provides statistically reliable uncertainty quantification (coverage rate significantly different from random)
- **Metric:** 95% prediction interval coverage rate
- **Significance Level:** α = 0.05

#### 1.2 Secondary Hypotheses

**H₄: Attention Mechanism Interpretability**
- **H₄₀:** Attention weights show no significant correlation with known energy system patterns
- **H₄₁:** Attention weights significantly correlate with established energy system temporal patterns
- **Test:** Correlation analysis with domain knowledge validation

**H₅: Forecast Horizon Performance Degradation**
- **H₅₀:** Model performance degradation with forecast horizon is not significantly different across architectures
- **H₅₁:** Advanced architectures show significantly less performance degradation at longer forecast horizons
- **Test:** Repeated measures ANOVA across forecast horizons

### 2. Statistical Assumptions

#### 2.1 Data Assumptions
- **Independence:** Time series residuals should be approximately independent after model fitting
- **Stationarity:** Forecast errors should be stationary for valid statistical testing
- **Normality:** For parametric tests, residuals should be approximately normally distributed
- **Homoscedasticity:** Variance of errors should be constant across time periods

#### 2.2 Model Assumptions
- **Identical Training Conditions:** All models trained on identical datasets with same preprocessing
- **Fair Comparison:** Hyperparameters optimized fairly for each architecture
- **Consistent Evaluation:** Same evaluation metrics and procedures applied to all models

---

## Significance Testing Procedures

### 1. Diebold-Mariano Test for Forecast Accuracy

The Diebold-Mariano (DM) test is the gold standard for comparing forecast accuracy between models.

#### 1.1 Test Formulation
**Null Hypothesis:** H₀: E[L(e₁ₜ) - L(e₂ₜ)] = 0  
**Alternative Hypothesis:** H₁: E[L(e₁ₜ) - L(e₂ₜ)] ≠ 0

Where:
- L(·) is the loss function (e.g., squared error, absolute error)
- e₁ₜ and e₂ₜ are forecast errors from models 1 and 2 at time t

#### 1.2 Test Statistic
```
DM = d̄ / √(V̂(d̄))
```

Where:
- d̄ = (1/T) Σ dₜ (mean loss differential)
- dₜ = L(e₁ₜ) - L(e₂ₜ) (loss differential at time t)
- V̂(d̄) = estimated variance of d̄ accounting for autocorrelation

#### 1.3 Critical Values
- **Two-tailed test:** Reject H₀ if |DM| > t₀.₀₂₅,T-1
- **One-tailed test:** Reject H₀ if DM > t₀.₀₅,T-1 (for testing if model 1 is better)

#### 1.4 Application to EnergyNexus
- Compare each advanced architecture against baseline LSTM
- Compare advanced architectures against each other
- Separate tests for each forecast horizon (1h, 6h, 24h)
- Separate tests for each target variable (demand, solar, wind)

### 2. Harvey-Leybourne-Newbold Modification

For small samples (T < 100), use the Harvey-Leybourne-Newbold (HLN) modification:

#### 2.1 Modified Test Statistic
```
HLN = DM × √[(T+1-2h+T⁻¹h(h-1))/T]
```

Where:
- h = forecast horizon
- T = sample size

#### 2.2 Application
Use HLN modification when:
- Test sample size < 100 observations
- Multi-step ahead forecasts (h > 1)
- Conservative testing approach desired

### 3. Paired t-Test for Model Performance

#### 3.1 Test Setup
**Use Case:** When forecast errors are approximately normally distributed

**Test Statistic:**
```
t = d̄ / (sₐ/√T)
```

Where:
- d̄ = mean difference in performance metrics
- sₐ = standard deviation of differences
- T = number of test observations

#### 3.2 Assumptions Check
Before applying t-test:
1. **Normality Test:** Shapiro-Wilk test on difference series
2. **Independence Test:** Ljung-Box test on difference series
3. **Outlier Detection:** Identify and handle extreme observations

### 4. Wilcoxon Signed-Rank Test (Non-parametric)

#### 4.1 When to Use
- Forecast error differences are not normally distributed
- Robust alternative to paired t-test
- Presence of outliers in performance differences

#### 4.2 Test Procedure
1. Calculate performance differences: dᵢ = metric₁ᵢ - metric₂ᵢ
2. Rank absolute differences: R(|dᵢ|)
3. Calculate test statistic: W = Σ sgn(dᵢ) × R(|dᵢ|)
4. Compare against critical values for given α and sample size

### 5. Time Series Cross-Validation Tests

#### 5.1 Rolling Window Cross-Validation
**Procedure:**
1. Use expanding/rolling window approach
2. Train models on window [1, t]
3. Test on window [t+1, t+h]
4. Roll window forward and repeat
5. Aggregate results across all test windows

#### 5.2 Statistical Testing on CV Results
- Apply DM test to aggregated CV forecasts
- Account for overlapping training sets in variance estimation
- Use clustered standard errors for valid inference

### 6. Bootstrap Testing Procedures

#### 6.1 Block Bootstrap for Time Series
**Purpose:** Generate distribution of test statistics under null hypothesis

**Procedure:**
1. Define block length l (typically l = ⌊T^(1/3)⌋)
2. Resample blocks with replacement
3. Reconstruct bootstrap time series
4. Calculate test statistic on bootstrap sample
5. Repeat B times (typically B = 1000)
6. Compare original test statistic to bootstrap distribution

#### 6.2 Moving Block Bootstrap Algorithm
```
1. Choose block length l
2. Create overlapping blocks: B₁ = {X₁,...,Xₗ}, B₂ = {X₂,...,Xₗ₊₁}, etc.
3. Sample ⌈T/l⌉ blocks with replacement
4. Concatenate to form bootstrap series
5. Calculate test statistic
6. Repeat for bootstrap distribution
```

---

## Multiple Comparison Corrections

### 1. Problem Statement

When comparing multiple models or testing multiple hypotheses simultaneously, the probability of making at least one Type I error (false positive) increases dramatically.

#### 1.1 Family-Wise Error Rate (FWER)
For m independent tests each with significance level α:
- **Probability of at least one Type I error:** 1 - (1-α)^m
- **Example:** For 10 tests with α = 0.05: FWER ≈ 0.40 (40% chance of false positive)

### 2. Bonferroni Correction

#### 2.1 Method
**Adjusted significance level:** α* = α/m

Where:
- α = desired family-wise error rate (typically 0.05)
- m = number of comparisons
- α* = significance level for each individual test

#### 2.2 Application to EnergyNexus
For comparing 4 advanced architectures against baseline:
- **Number of comparisons:** m = 4
- **Individual test significance:** α* = 0.05/4 = 0.0125
- **Conservative but guarantees FWER ≤ 0.05**

#### 2.3 Advantages and Limitations
**Advantages:**
- Simple to implement
- Guarantees strong FWER control
- No assumptions about test dependencies

**Limitations:**
- Very conservative (low statistical power)
- May miss true differences
- Treats all comparisons equally

### 3. Holm-Bonferroni Sequential Method

#### 3.1 Step-Down Procedure
1. Order p-values: p₍₁₎ ≤ p₍₂₎ ≤ ... ≤ p₍ₘ₎
2. For i = 1, 2, ..., m:
   - If p₍ᵢ₎ ≤ α/(m+1-i), reject H₍ᵢ₎ and continue
   - If p₍ᵢ₎ > α/(m+1-i), accept H₍ᵢ₎ and all remaining hypotheses

#### 3.2 Advantages over Bonferroni
- More powerful (higher probability of detecting true effects)
- Still controls FWER at level α
- Sequential testing stops at first non-significant result

### 4. False Discovery Rate (FDR) Control

#### 4.1 Benjamini-Hochberg Procedure
**Objective:** Control expected proportion of false discoveries among rejected hypotheses

**Procedure:**
1. Order p-values: p₍₁₎ ≤ p₍₂₎ ≤ ... ≤ p₍ₘ₎
2. Find largest k such that: p₍ₖ₎ ≤ (k/m) × α
3. Reject hypotheses H₍₁₎, H₍₂₎, ..., H₍ₖ₎

#### 4.2 When to Use FDR
- Large number of comparisons
- Exploratory data analysis
- When some false positives are acceptable
- Higher statistical power desired

### 5. Tukey's Honestly Significant Difference (HSD)

#### 5.1 Application
**Use Case:** Comparing all pairs of model means simultaneously

**Test Statistic:**
```
q = (x̄ᵢ - x̄ⱼ) / √(MSE/n)
```

Where:
- x̄ᵢ, x̄ⱼ = sample means for models i and j
- MSE = mean square error from ANOVA
- n = sample size per group

#### 5.2 Critical Value
Compare q against critical value from Studentized Range distribution: q₀.₀₅,k,df

### 6. Dunnett's Test

#### 6.1 Multiple Comparisons Against Control
**Purpose:** Compare multiple treatment groups against a single control (baseline LSTM)

**Advantages:**
- More powerful than Bonferroni when comparing against control
- Accounts for correlation structure in comparisons
- Designed specifically for control vs. treatment comparisons

#### 6.2 Test Statistic
```
dᵢ = (x̄ᵢ - x̄control) / √(MSE × (1/nᵢ + 1/ncontrol))
```

Compare against critical values from Dunnett's table.

---

## Implementation Guidelines

### 1. Testing Hierarchy

#### 1.1 Primary Analysis Pipeline
1. **Omnibus Test:** ANOVA to test if any model differs from others
2. **Pairwise Comparisons:** If omnibus test significant, proceed with pairwise tests
3. **Multiple Comparison Correction:** Apply appropriate correction method
4. **Effect Size Estimation:** Calculate practical significance measures

#### 1.2 Secondary Analysis
1. **Robustness Checks:** Non-parametric alternatives
2. **Sensitivity Analysis:** Different significance levels and correction methods
3. **Bootstrap Validation:** Confirm results using bootstrap procedures

### 2. Statistical Power Analysis

#### 2.1 Sample Size Determination
**For Diebold-Mariano Test:**
```
n = 2 × (z_{α/2} + z_β)² × σ²_d / δ²
```

Where:
- z_{α/2} = critical value for Type I error
- z_β = critical value for Type II error (power = 1-β)
- σ²_d = variance of loss differential
- δ = minimum detectable difference

#### 2.2 Post-hoc Power Analysis
Calculate achieved power for non-significant results:
```
Power = Φ(|d̄|/SE_d - z_{α/2})
```

### 3. Reporting Guidelines

#### 3.1 Required Information
- **Test statistic and p-value**
- **Degrees of freedom (where applicable)**
- **Confidence intervals**
- **Effect size measures**
- **Multiple comparison correction applied**
- **Sample size and power analysis**

#### 3.2 Result Interpretation
- **Statistical significance:** p-value interpretation
- **Practical significance:** Effect size and confidence intervals
- **Economic significance:** Cost-benefit analysis of forecast improvements

---

## Code Examples

### 1. Diebold-Mariano Test Implementation

```python
import numpy as np
from scipy import stats
import pandas as pd

def diebold_mariano_test(errors1, errors2, h=1):
    """
    Diebold-Mariano test for forecast accuracy comparison.
    
    Parameters:
    errors1, errors2: forecast errors from two models
    h: forecast horizon
    
    Returns:
    dm_stat: Diebold-Mariano test statistic
    p_value: two-tailed p-value
    """
    
    # Calculate loss differential
    d = errors1**2 - errors2**2  # Using squared loss
    d_bar = np.mean(d)
    
    # Calculate variance with Newey-West correction for autocorrelation
    T = len(d)
    gamma_0 = np.var(d, ddof=1)
    
    # Autocorrelation corrections for h > 1
    gamma = 0
    for j in range(1, h):
        gamma_j = np.cov(d[:-j], d[j:])[0, 1] if j < T else 0
        gamma += gamma_j
    
    var_d = (gamma_0 + 2 * gamma) / T
    
    # Harvey-Leybourne-Newbold modification for small samples
    if T < 100:
        factor = np.sqrt((T + 1 - 2*h + T**(-1) * h * (h-1)) / T)
        dm_stat = d_bar / np.sqrt(var_d) * factor
    else:
        dm_stat = d_bar / np.sqrt(var_d)
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    return dm_stat, p_value

# Example usage
def compare_models(model1_errors, model2_errors, model_names, alpha=0.05):
    """Compare multiple models using Diebold-Mariano test with corrections."""
    
    n_models = len(model1_errors)
    results = []
    
    # Perform pairwise comparisons
    for i in range(n_models):
        for j in range(i+1, n_models):
            dm_stat, p_value = diebold_mariano_test(
                model1_errors[i], model1_errors[j]
            )
            
            results.append({
                'Model1': model_names[i],
                'Model2': model_names[j],
                'DM_Statistic': dm_stat,
                'p_value': p_value,
                'Significant': p_value < alpha
            })
    
    return pd.DataFrame(results)
```

### 2. Multiple Comparison Corrections

```python
def apply_multiple_comparison_corrections(p_values, alpha=0.05):
    """
    Apply various multiple comparison corrections.
    
    Parameters:
    p_values: array of p-values from individual tests
    alpha: family-wise error rate
    
    Returns:
    DataFrame with corrected results
    """
    
    results = pd.DataFrame({
        'Original_p_value': p_values
    })
    
    m = len(p_values)
    
    # Bonferroni correction
    results['Bonferroni_alpha'] = alpha / m
    results['Bonferroni_Significant'] = p_values < (alpha / m)
    
    # Holm-Bonferroni correction
    sorted_indices = np.argsort(p_values)
    holm_results = np.zeros(m, dtype=bool)
    
    for i, idx in enumerate(sorted_indices):
        adjusted_alpha = alpha / (m - i)
        if p_values[idx] <= adjusted_alpha:
            holm_results[idx] = True
        else:
            break  # Stop at first non-significant result
    
    results['Holm_Significant'] = holm_results
    
    # Benjamini-Hochberg (FDR) correction
    sorted_p = np.sort(p_values)
    fdr_threshold = 0
    
    for i in range(m-1, -1, -1):
        if sorted_p[i] <= (i+1) * alpha / m:
            fdr_threshold = sorted_p[i]
            break
    
    results['BH_Significant'] = p_values <= fdr_threshold
    
    return results

# Example usage
def statistical_testing_pipeline(model_results, baseline_model='baseline_lstm'):
    """
    Complete statistical testing pipeline for model comparison.
    """
    
    # Extract model names and errors
    model_names = list(model_results.keys())
    model_errors = {name: results['errors'] for name, results in model_results.items()}
    
    # Compare each model against baseline
    p_values = []
    comparisons = []
    
    for model_name in model_names:
        if model_name != baseline_model:
            dm_stat, p_value = diebold_mariano_test(
                model_errors[baseline_model],
                model_errors[model_name]
            )
            p_values.append(p_value)
            comparisons.append(f"{model_name} vs {baseline_model}")
    
    # Apply multiple comparison corrections
    corrected_results = apply_multiple_comparison_corrections(p_values)
    corrected_results['Comparison'] = comparisons
    
    return corrected_results
```

### 3. Bootstrap Testing

```python
def block_bootstrap_test(data1, data2, block_length=None, n_bootstrap=1000):
    """
    Block bootstrap test for time series data.
    
    Parameters:
    data1, data2: time series data
    block_length: length of blocks for bootstrap
    n_bootstrap: number of bootstrap samples
    
    Returns:
    bootstrap_p_value: p-value from bootstrap test
    """
    
    if block_length is None:
        block_length = int(len(data1)**(1/3))
    
    # Original test statistic
    original_diff = np.mean(data1) - np.mean(data2)
    
    # Combine data for bootstrap under null hypothesis
    combined_data = np.concatenate([data1 - np.mean(data1), data2 - np.mean(data2)])
    
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        # Generate bootstrap sample
        bootstrap_sample = block_bootstrap_sample(combined_data, len(data1) + len(data2), block_length)
        
        # Split back into two samples
        boot_sample1 = bootstrap_sample[:len(data1)]
        boot_sample2 = bootstrap_sample[len(data1):]
        
        # Calculate bootstrap test statistic
        boot_stat = np.mean(boot_sample1) - np.mean(boot_sample2)
        bootstrap_stats.append(boot_stat)
    
    # Calculate p-value
    bootstrap_stats = np.array(bootstrap_stats)
    p_value = 2 * min(
        np.mean(bootstrap_stats >= abs(original_diff)),
        np.mean(bootstrap_stats <= -abs(original_diff))
    )
    
    return p_value

def block_bootstrap_sample(data, sample_length, block_length):
    """Generate block bootstrap sample."""
    
    n = len(data)
    n_blocks = int(np.ceil(sample_length / block_length))
    
    # Generate random starting positions
    start_positions = np.random.randint(0, n - block_length + 1, n_blocks)
    
    # Concatenate blocks
    bootstrap_sample = []
    for start_pos in start_positions:
        block = data[start_pos:start_pos + block_length]
        bootstrap_sample.extend(block)
    
    return np.array(bootstrap_sample[:sample_length])
```

### 4. Comprehensive Testing Framework

```python
class EnergyForecastingStatTests:
    """
    Comprehensive statistical testing framework for energy forecasting models.
    """
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.results = {}
    
    def run_comprehensive_tests(self, model_results, baseline='baseline_lstm'):
        """
        Run comprehensive statistical testing pipeline.
        
        Parameters:
        model_results: dict with model names as keys and results as values
        baseline: name of baseline model for comparisons
        """
        
        print("Running Comprehensive Statistical Testing Pipeline")
        print("=" * 60)
        
        # 1. Descriptive statistics
        self.descriptive_analysis(model_results)
        
        # 2. Normality tests
        self.normality_tests(model_results)
        
        # 3. Pairwise comparisons
        self.pairwise_comparisons(model_results, baseline)
        
        # 4. Multiple comparison corrections
        self.multiple_comparison_analysis()
        
        # 5. Bootstrap validation
        self.bootstrap_validation(model_results, baseline)
        
        # 6. Generate report
        self.generate_statistical_report()
    
    def descriptive_analysis(self, model_results):
        """Calculate descriptive statistics for all models."""
        
        desc_stats = {}
        for model_name, results in model_results.items():
            errors = results['errors']
            desc_stats[model_name] = {
                'mean_error': np.mean(errors),
                'std_error': np.std(errors),
                'mae': np.mean(np.abs(errors)),
                'rmse': np.sqrt(np.mean(errors**2)),
                'min_error': np.min(errors),
                'max_error': np.max(errors),
                'skewness': stats.skew(errors),
                'kurtosis': stats.kurtosis(errors)
            }
        
        self.results['descriptive_stats'] = desc_stats
        print("✓ Descriptive analysis completed")
    
    def normality_tests(self, model_results):
        """Test normality of forecast errors."""
        
        normality_results = {}
        for model_name, results in model_results.items():
            errors = results['errors']
            
            # Shapiro-Wilk test
            shapiro_stat, shapiro_p = stats.shapiro(errors[:5000])  # Limit sample size
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.kstest(errors, 'norm', 
                                       args=(np.mean(errors), np.std(errors)))
            
            normality_results[model_name] = {
                'shapiro_statistic': shapiro_stat,
                'shapiro_p_value': shapiro_p,
                'shapiro_normal': shapiro_p > self.alpha,
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p,
                'ks_normal': ks_p > self.alpha
            }
        
        self.results['normality_tests'] = normality_results
        print("✓ Normality tests completed")
    
    def pairwise_comparisons(self, model_results, baseline):
        """Perform pairwise statistical comparisons."""
        
        model_names = list(model_results.keys())
        comparison_results = []
        
        for model_name in model_names:
            if model_name != baseline:
                errors_baseline = model_results[baseline]['errors']
                errors_model = model_results[model_name]['errors']
                
                # Diebold-Mariano test
                dm_stat, dm_p = diebold_mariano_test(errors_baseline, errors_model)
                
                # Paired t-test
                error_diff = errors_baseline**2 - errors_model**2
                t_stat, t_p = stats.ttest_1samp(error_diff, 0)
                
                # Wilcoxon signed-rank test
                wilcoxon_stat, wilcoxon_p = stats.wilcoxon(error_diff)
                
                comparison_results.append({
                    'model': model_name,
                    'baseline': baseline,
                    'dm_statistic': dm_stat,
                    'dm_p_value': dm_p,
                    't_statistic': t_stat,
                    't_p_value': t_p,
                    'wilcoxon_statistic': wilcoxon_stat,
                    'wilcoxon_p_value': wilcoxon_p
                })
        
        self.results['pairwise_comparisons'] = comparison_results
        print("✓ Pairwise comparisons completed")
    
    def multiple_comparison_analysis(self):
        """Apply multiple comparison corrections."""
        
        if 'pairwise_comparisons' not in self.results:
            print("Error: Run pairwise comparisons first")
            return
        
        # Extract p-values for different tests
        dm_p_values = [comp['dm_p_value'] for comp in self.results['pairwise_comparisons']]
        t_p_values = [comp['t_p_value'] for comp in self.results['pairwise_comparisons']]
        
        # Apply corrections
        dm_corrected = apply_multiple_comparison_corrections(dm_p_values, self.alpha)
        t_corrected = apply_multiple_comparison_corrections(t_p_values, self.alpha)
        
        self.results['multiple_comparisons'] = {
            'diebold_mariano': dm_corrected,
            'paired_t_test': t_corrected
        }
        print("✓ Multiple comparison corrections applied")
    
    def bootstrap_validation(self, model_results, baseline, n_bootstrap=1000):
        """Validate results using bootstrap methods."""
        
        bootstrap_results = []
        
        for model_name, results in model_results.items():
            if model_name != baseline:
                errors_baseline = model_results[baseline]['errors']
                errors_model = results['errors']
                
                # Block bootstrap test
                bootstrap_p = block_bootstrap_test(
                    errors_baseline**2, errors_model**2, 
                    n_bootstrap=n_bootstrap
                )
                
                bootstrap_results.append({
                    'model': model_name,
                    'baseline': baseline,
                    'bootstrap_p_value': bootstrap_p,
                    'bootstrap_significant': bootstrap_p < self.alpha
                })
        
        self.results['bootstrap_validation'] = bootstrap_results
        print("✓ Bootstrap validation completed")
    
    def generate_statistical_report(self):
        """Generate comprehensive statistical testing report."""
        
        print("\n" + "="*80)
        print("STATISTICAL TESTING REPORT")
        print("="*80)
        
        # Summary of findings
        if 'multiple_comparisons' in self.results:
            dm_results = self.results['multiple_comparisons']['diebold_mariano']
            significant_models = dm_results[dm_results['Bonferroni_Significant']].index
            
            print(f"\nSIGNIFICANT MODELS (Bonferroni corrected α = {self.alpha}):")
            if len(significant_models) > 0:
                for idx in significant_models:
                    comp = self.results['pairwise_comparisons'][idx]
                    print(f"  ✓ {comp['model']} vs {comp['baseline']}: p = {comp['dm_p_value']:.4f}")
            else:
                print("  No models show significant improvement over baseline")
        
        # Power analysis recommendations
        print(f"\nRECOMMENDations:")
        print("  1. Consider effect size in addition to statistical significance")
        print("  2. Validate results on independent test set")
        print("  3. Assess practical significance for energy applications")
        print("  4. Consider ensemble approaches for improved robustness")
        
        return self.results

# Example usage
if __name__ == "__main__":
    # Mock model results for demonstration
    np.random.seed(42)
    model_results = {
        'baseline_lstm': {'errors': np.random.normal(0, 1, 1000)},
        'attention_lstm': {'errors': np.random.normal(-0.2, 0.9, 1000)},
        'encoder_decoder_lstm': {'errors': np.random.normal(-0.1, 0.95, 1000)},
        'multivariate_lstm': {'errors': np.random.normal(-0.3, 0.85, 1000)},
        'ensemble_lstm': {'errors': np.random.normal(-0.25, 0.8, 1000)}
    }
    
    # Run comprehensive testing
    test_framework = EnergyForecastingStatTests(alpha=0.05)
    results = test_framework.run_comprehensive_tests(model_results)
```

---

## Summary

This document provides a comprehensive framework for statistical testing and validation of advanced LSTM architectures in energy forecasting. The methodology ensures:

1. **Rigorous hypothesis testing** with appropriate statistical methods
2. **Proper handling of multiple comparisons** to control false positive rates
3. **Robust validation** through bootstrap and cross-validation approaches
