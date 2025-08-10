# Uncertainty Quantification Methodology

**Author:** Aditya Talekar (ec24018@qmul.ac.uk)  
**Supervisor:** Saqib Iqbal  
**Institution:** Queen Mary University of London  
**Project:** EnergyNexus - Advanced LSTM Architectures for Energy Forecasting  

---

## Table of Contents
1. [Introduction to Uncertainty Quantification](#introduction-to-uncertainty-quantification)
2. [Types of Uncertainty in Energy Forecasting](#types-of-uncertainty-in-energy-forecasting)
3. [Uncertainty Quantification Methods](#uncertainty-quantification-methods)
4. [Ensemble-Based Uncertainty Estimation](#ensemble-based-uncertainty-estimation)
5. [Bayesian Neural Networks](#bayesian-neural-networks)
6. [Prediction Interval Construction](#prediction-interval-construction)
7. [Calibration Assessment](#calibration-assessment)
8. [Implementation Framework](#implementation-framework)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Practical Applications](#practical-applications)

---

## Introduction to Uncertainty Quantification

### 1.1 Motivation for Uncertainty Quantification in Energy Forecasting

Energy system operations involve critical decisions with significant economic and reliability implications. Traditional point forecasts provide single-valued predictions without indicating confidence levels, which limits their utility for:

- **Risk Management:** Understanding potential forecast errors for contingency planning
- **Decision Making:** Incorporating uncertainty into optimization algorithms
- **Grid Reliability:** Assessing probability of demand exceeding supply capacity
- **Economic Dispatch:** Considering forecast uncertainty in generation scheduling
- **Regulatory Compliance:** Meeting uncertainty reporting requirements

### 1.2 Uncertainty Quantification Objectives

The primary objectives of implementing uncertainty quantification in our advanced LSTM architectures are:

1. **Probabilistic Forecasting:** Provide probability distributions rather than point estimates
2. **Confidence Intervals:** Generate prediction intervals with specified coverage probabilities
3. **Risk Assessment:** Enable risk-aware decision making in energy systems
4. **Model Reliability:** Assess and communicate model confidence levels
5. **Uncertainty Propagation:** Understand how input uncertainties affect output predictions

### 1.3 Theoretical Foundation

#### Mathematical Framework
Let **y** be the true energy demand/generation at time *t*, and **ŷ** be the model prediction. The uncertainty quantification framework aims to estimate:

```
P(y | x, θ, D)
```

Where:
- **x** = input features (weather, historical data, etc.)
- **θ** = model parameters
- **D** = training dataset
- **P(y | x, θ, D)** = predictive distribution

#### Sources of Uncertainty
1. **Aleatoric Uncertainty (Data Uncertainty):**
   - Inherent randomness in the data
   - Measurement noise
   - Natural variability in energy systems

2. **Epistemic Uncertainty (Model Uncertainty):**
   - Uncertainty in model parameters
   - Model structure uncertainty
   - Limited training data

---

## Types of Uncertainty in Energy Forecasting

### 2.1 Aleatoric Uncertainty

#### 2.1.1 Weather-Related Uncertainty
**Sources:**
- Meteorological forecast errors
- Micro-climate variations
- Extreme weather events

**Characteristics:**
- Irreducible with more data
- Varies with weather conditions
- Higher during transition seasons

**Mathematical Representation:**
```
σ_aleatoric²(x) = E[(y - E[y|x])² | x]
```

#### 2.1.2 Demand Variability Uncertainty
**Sources:**
- Consumer behavior variations
- Economic activity fluctuations
- Seasonal demand patterns

**Implementation Example:**
```python
def estimate_aleatoric_uncertainty(historical_errors, features):
    """
    Estimate aleatoric uncertainty using residual analysis
    """
    # Group errors by similar conditions
    grouped_errors = group_by_conditions(historical_errors, features)
    
    # Calculate conditional variance
    aleatoric_variance = {}
    for condition, errors in grouped_errors.items():
        aleatoric_variance[condition] = np.var(errors)
    
    return aleatoric_variance
```

### 2.2 Epistemic Uncertainty

#### 2.2.1 Parameter Uncertainty
**Sources:**
- Finite training dataset
- Model parameter estimation errors
- Optimization convergence issues

**Bayesian Treatment:**
```
p(θ | D) ∝ p(D | θ) × p(θ)
```

#### 2.2.2 Model Structure Uncertainty
**Sources:**
- Architecture selection
- Hyperparameter choices
- Feature selection decisions

**Model Averaging Approach:**
```
p(y | x, D) = Σ_k p(y | x, M_k, θ_k) × p(M_k | D)
```

### 2.3 Uncertainty Propagation

#### Input-to-Output Uncertainty Mapping
```python
def uncertainty_propagation(input_uncertainty, model_sensitivity):
    """
    Propagate input uncertainties through the model
    """
    # Linear approximation (first-order Taylor expansion)
    output_variance = np.sum((model_sensitivity ** 2) * input_uncertainty)
    
    # Higher-order effects (Monte Carlo)
    monte_carlo_samples = []
    for _ in range(1000):
        noisy_input = add_uncertainty(input_features, input_uncertainty)
        output_sample = model.predict(noisy_input)
        monte_carlo_samples.append(output_sample)
    
    mc_variance = np.var(monte_carlo_samples)
    
    return output_variance, mc_variance
```

---

## Uncertainty Quantification Methods

### 3.1 Method Classification

| **Method Category** | **Advantages** | **Disadvantages** | **Computational Cost** |
|-------------------|---------------|------------------|----------------------|
| **Ensemble Methods** | Easy to implement, Model-agnostic | Multiple model training | High |
| **Bayesian Neural Networks** | Principled uncertainty, Single model | Complex implementation | Medium |
| **Dropout-based Methods** | Fast inference, Minimal changes | Approximate uncertainty | Low |
| **Quantile Regression** | Direct interval prediction | Limited distributional info | Low |
| **Gaussian Processes** | Exact Bayesian inference | Scalability issues | Very High |

### 3.2 Selection Criteria for Energy Forecasting

#### Computational Requirements
- **Real-time Applications:** Fast methods (dropout, quantile regression)
- **Batch Processing:** Ensemble methods acceptable
- **Research Applications:** Bayesian methods for rigor

#### Uncertainty Quality Requirements
- **Critical Applications:** High-quality uncertainty (Bayesian, ensemble)
- **Operational Use:** Good enough uncertainty (dropout, quantile)
- **Regulatory Reporting:** Well-calibrated uncertainty (ensemble, Bayesian)

---

## Ensemble-Based Uncertainty Estimation

### 4.1 Ensemble Design Principles

#### 4.1.1 Diversity Mechanisms
**Bootstrap Aggregating (Bagging):**
```python
def create_bootstrap_ensemble(base_model, X_train, y_train, n_models=10):
    """
    Create ensemble using bootstrap sampling for diversity
    """
    ensemble_models = []
    
    for i in range(n_models):
        # Bootstrap sample
        bootstrap_indices = np.random.choice(len(X_train), 
                                           size=len(X_train), 
                                           replace=True)
        X_bootstrap = X_train[bootstrap_indices]
        y_bootstrap = y_train[bootstrap_indices]
        
        # Train model on bootstrap sample
        model = clone_model(base_model)
        model.fit(X_bootstrap, y_bootstrap)
        ensemble_models.append(model)
    
    return ensemble_models
```

**Architecture Diversity:**
```python
def create_diverse_ensemble():
    """
    Create ensemble with diverse LSTM architectures
    """
    ensemble_configs = [
        {'lstm_units': [64, 32], 'dropout': 0.2, 'learning_rate': 0.001},
        {'lstm_units': [80, 48], 'dropout': 0.3, 'learning_rate': 0.0008},
        {'lstm_units': [48, 24], 'dropout': 0.15, 'learning_rate': 0.0012}
    ]
    
    ensemble_models = []
    for config in ensemble_configs:
        model = build_lstm_model(**config)
        ensemble_models.append(model)
    
    return ensemble_models
```

#### 4.1.2 Training Diversity
**Different Initialization:**
- Random weight initialization with different seeds
- Different optimization algorithms (Adam, RMSprop, SGD)
- Varying learning rates and schedules

**Data Diversity:**
- Bootstrap sampling of training data
- Different feature subsets
- Temporal subsampling strategies

### 4.2 Uncertainty Estimation from Ensembles

#### 4.2.1 Predictive Mean and Variance
```python
def ensemble_uncertainty_estimation(ensemble_models, X_test):
    """
    Estimate uncertainty from ensemble predictions
    """
    predictions = []
    
    # Collect predictions from all ensemble members
    for model in ensemble_models:
        pred = model.predict(X_test)
        predictions.append(pred)
    
    predictions = np.array(predictions)  # Shape: (n_models, n_samples, n_outputs)
    
    # Ensemble statistics
    ensemble_mean = np.mean(predictions, axis=0)
    ensemble_variance = np.var(predictions, axis=0)
    ensemble_std = np.sqrt(ensemble_variance)
    
    # Additional statistics
    ensemble_min = np.min(predictions, axis=0)
    ensemble_max = np.max(predictions, axis=0)
    ensemble_median = np.median(predictions, axis=0)
    
    uncertainty_metrics = {
        'mean': ensemble_mean,
        'variance': ensemble_variance,
        'std': ensemble_std,
        'min': ensemble_min,
        'max': ensemble_max,
        'median': ensemble_median,
        'raw_predictions': predictions
    }
    
    return uncertainty_metrics
```

#### 4.2.2 Prediction Interval Construction
```python
def construct_prediction_intervals(ensemble_predictions, confidence_levels=[0.68, 0.95]):
    """
    Construct prediction intervals from ensemble predictions
    """
    intervals = {}
    
    for confidence_level in confidence_levels:
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(ensemble_predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(ensemble_predictions, upper_percentile, axis=0)
        
        intervals[f'{confidence_level:.0%}'] = {
            'lower': lower_bound,
            'upper': upper_bound,
            'width': upper_bound - lower_bound
        }
    
    return intervals
```

### 4.3 Advanced Ensemble Techniques

#### 4.3.1 Weighted Ensembles
```python
def optimize_ensemble_weights(ensemble_predictions, true_values, method='mse'):
    """
    Optimize ensemble weights based on validation performance
    """
    from scipy.optimize import minimize
    
    def objective(weights):
        weights = weights / np.sum(weights)  # Normalize weights
        weighted_pred = np.sum(ensemble_predictions * weights[:, None, None], axis=0)
        
        if method == 'mse':
            return np.mean((weighted_pred - true_values) ** 2)
        elif method == 'mae':
            return np.mean(np.abs(weighted_pred - true_values))
    
    # Initialize equal weights
    n_models = ensemble_predictions.shape[0]
    initial_weights = np.ones(n_models) / n_models
    
    # Optimize weights
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n_models)]
    
    result = minimize(objective, initial_weights, 
                     method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x
```

#### 4.3.2 Hierarchical Ensembles
```python
def create_hierarchical_ensemble(base_models, meta_model):
    """
    Create two-level ensemble with meta-learning
    """
    # Level 1: Base model predictions
    base_predictions = []
    for model in base_models:
        pred = model.predict(X_validation)
        base_predictions.append(pred)
    
    # Level 2: Meta-model learns to combine base predictions
    meta_features = np.column_stack(base_predictions)
    meta_model.fit(meta_features, y_validation)
    
    return base_models, meta_model
```

---

## Bayesian Neural Networks

### 5.1 Theoretical Foundation

#### 5.1.1 Bayesian Framework for Neural Networks
In Bayesian neural networks, model parameters are treated as random variables with probability distributions:

```
p(θ | D) = p(D | θ) × p(θ) / p(D)
```

**Prior Distribution:** p(θ)
- Encodes beliefs about parameters before seeing data
- Common choices: Gaussian, Laplace, or hierarchical priors

**Likelihood:** p(D | θ)
- Probability of observing data given parameters
- For regression: typically Gaussian likelihood

**Posterior Distribution:** p(θ | D)
- Updated beliefs about parameters after seeing data
- Usually intractable, requiring approximation methods

#### 5.1.2 Predictive Distribution
```
p(y* | x*, D) = ∫ p(y* | x*, θ) × p(θ | D) dθ
```

This integral captures both aleatoric and epistemic uncertainty.

### 5.2 Variational Inference for BNNs

#### 5.2.1 Variational Approximation
Since exact Bayesian inference is intractable, we approximate the posterior p(θ | D) with a simpler distribution q(θ | φ):

```python
def variational_inference_loss(y_true, y_pred, kl_divergence_loss):
    """
    Variational inference loss combining data likelihood and KL divergence
    """
    # Reconstruction loss (negative log-likelihood)
    reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # KL divergence between approximate and prior distributions
    kl_loss = kl_divergence_loss
    
    # Variational free energy
    total_loss = reconstruction_loss + kl_loss
    
    return total_loss
```

#### 5.2.2 Reparameterization Trick
```python
class VariationalDense(tf.keras.layers.Layer):
    """
    Variational dense layer with learnable mean and variance
    """
    def __init__(self, units, prior_std=1.0):
        super().__init__()
        self.units = units
        self.prior_std = prior_std
    
    def build(self, input_shape):
        # Weight parameters (mean and log variance)
        self.w_mean = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            name='w_mean'
        )
        self.w_log_var = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            name='w_log_var'
        )
        
        # Bias parameters
        self.b_mean = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            name='b_mean'
        )
        self.b_log_var = self.add_weight(
            shape=(self.units,),
            initializer='random_normal',
            name='b_log_var'
        )
    
    def call(self, inputs, training=True):
        if training:
            # Sample weights using reparameterization trick
            w_std = tf.exp(0.5 * self.w_log_var)
            w_noise = tf.random.normal(tf.shape(self.w_mean))
            w_sample = self.w_mean + w_std * w_noise
            
            b_std = tf.exp(0.5 * self.b_log_var)
            b_noise = tf.random.normal(tf.shape(self.b_mean))
            b_sample = self.b_mean + b_std * b_noise
        else:
            # Use mean weights for prediction
            w_sample = self.w_mean
            b_sample = self.b_mean
        
        output = tf.matmul(inputs, w_sample) + b_sample
        
        # Calculate KL divergence
        kl_div = self.compute_kl_divergence()
        self.add_loss(kl_div)
        
        return output
    
    def compute_kl_divergence(self):
        """Compute KL divergence between approximate and prior distributions"""
        # KL divergence for weights
        w_kl = -0.5 * tf.reduce_sum(
            1 + self.w_log_var - tf.square(self.w_mean) - tf.exp(self.w_log_var)
        )
        
        # KL divergence for biases
        b_kl = -0.5 * tf.reduce_sum(
            1 + self.b_log_var - tf.square(self.b_mean) - tf.exp(self.b_log_var)
        )
        
        return w_kl + b_kl
```

### 5.3 Monte Carlo Dropout

#### 5.3.1 Theoretical Justification
Monte Carlo Dropout approximates Bayesian inference by:
1. Training with dropout as a regularization technique
2. Keeping dropout active during inference
3. Running multiple forward passes to sample from approximate posterior

```python
def mc_dropout_uncertainty(model, X_test, n_samples=100, dropout_rate=0.1):
    """
    Estimate uncertainty using Monte Carlo Dropout
    """
    # Enable dropout during inference
    predictions = []
    
    for _ in range(n_samples):
        # Forward pass with dropout active
        pred = model(X_test, training=True)
        predictions.append(pred.numpy())
    
    predictions = np.array(predictions)
    
    # Calculate statistics
    mc_mean = np.mean(predictions, axis=0)
    mc_variance = np.var(predictions, axis=0)
    mc_std = np.sqrt(mc_variance)
    
    return mc_mean, mc_variance, mc_std
```

#### 5.3.2 Concrete Dropout
```python
class ConcreteDropout(tf.keras.layers.Layer):
    """
    Concrete Dropout with learnable dropout probability
    """
    def __init__(self, weight_regularizer=1e-6, dropout_regularizer=1e-5):
        super().__init__()
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
    
    def build(self, input_shape):
        # Learnable dropout probability (in logit space)
        self.p_logit = self.add_weight(
            name='p_logit',
            shape=(),
            initializer='zeros',
            trainable=True
        )
    
    def call(self, inputs, training=True):
        # Convert logit to probability
        p = tf.nn.sigmoid(self.p_logit)
        
        if training:
            # Concrete dropout (differentiable)
            eps = 1e-7
            temp = 0.1
            
            unif_noise = tf.random.uniform(tf.shape(inputs))
            drop_prob = tf.log(p + eps) - tf.log(1 - p + eps) + tf.log(unif_noise + eps) - tf.log(1 - unif_noise + eps)
            drop_prob = tf.nn.sigmoid(drop_prob / temp)
            
            random_tensor = 1 - drop_prob
            retain_prob = 1 - p
            inputs *= random_tensor / retain_prob
        
        # Add regularization loss
        regularization = p * tf.log(p + 1e-8) + (1 - p) * tf.log(1 - p + 1e-8)
        regularization *= self.dropout_regularizer
        self.add_loss(regularization)
        
        return inputs
```

---

## Prediction Interval Construction

### 6.1 Parametric Approaches

#### 6.1.1 Gaussian Assumption
When prediction errors are assumed to follow a Gaussian distribution:

```python
def gaussian_prediction_intervals(mean_prediction, prediction_std, confidence_levels):
    """
    Construct prediction intervals assuming Gaussian distribution
    """
    from scipy import stats
    
    intervals = {}
    
    for confidence_level in confidence_levels:
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        lower_bound = mean_prediction - z_score * prediction_std
        upper_bound = mean_prediction + z_score * prediction_std
        
        intervals[f'{confidence_level:.0%}'] = {
            'lower': lower_bound,
            'upper': upper_bound,
            'width': 2 * z_score * prediction_std
        }
    
    return intervals
```

#### 6.1.2 Student-t Distribution
For small samples or when accounting for parameter uncertainty:

```python
def student_t_prediction_intervals(mean_prediction, prediction_std, 
                                  degrees_freedom, confidence_levels):
    """
    Construct prediction intervals using Student-t distribution
    """
    from scipy import stats
    
    intervals = {}
    
    for confidence_level in confidence_levels:
        alpha = 1 - confidence_level
        t_score = stats.t.ppf(1 - alpha/2, degrees_freedom)
        
        lower_bound = mean_prediction - t_score * prediction_std
        upper_bound = mean_prediction + t_score * prediction_std
        
        intervals[f'{confidence_level:.0%}'] = {
            'lower': lower_bound,
            'upper': upper_bound,
            'width': 2 * t_score * prediction_std
        }
    
    return intervals
```

### 6.2 Non-Parametric Approaches

#### 6.2.1 Quantile Regression
```python
def quantile_regression_intervals(X_train, y_train, X_test, quantiles=[0.025, 0.975]):
    """
    Construct prediction intervals using quantile regression
    """
    from sklearn.ensemble import GradientBoostingRegressor
    
    intervals = {}
    
    for quantile in quantiles:
        # Train quantile regression model
        model = GradientBoostingRegressor(
            loss='quantile',
            alpha=quantile,
            n_estimators=100
        )
        model.fit(X_train, y_train)
        
        # Generate predictions
        predictions = model.predict(X_test)
        intervals[f'quantile_{quantile}'] = predictions
    
    # Construct confidence intervals
    confidence_intervals = {}
    if 0.025 in quantiles and 0.975 in quantiles:
        confidence_intervals['95%'] = {
            'lower': intervals['quantile_0.025'],
            'upper': intervals['quantile_0.975'],
            'width': intervals['quantile_0.975'] - intervals['quantile_0.025']
        }
    
    return confidence_intervals
```

#### 6.2.2 Conformal Prediction
```python
def conformal_prediction_intervals(calibration_residuals, new_predictions, 
                                  confidence_level=0.95):
    """
    Construct prediction intervals using conformal prediction
    """
    # Calculate quantile of absolute residuals
    alpha = 1 - confidence_level
    residual_quantile = np.quantile(np.abs(calibration_residuals), 1 - alpha)
    
    # Construct prediction intervals
    lower_bound = new_predictions - residual_quantile
    upper_bound = new_predictions + residual_quantile
    
    intervals = {
        f'{confidence_level:.0%}': {
            'lower': lower_bound,
            'upper': upper_bound,
            'width': 2 * residual_quantile
        }
    }
    
    return intervals
```

### 6.3 Adaptive Prediction Intervals

#### 6.3.1 Heteroscedastic Models
```python
class HeteroscedasticLSTM(tf.keras.Model):
    """
    LSTM model that predicts both mean and variance
    """
    def __init__(self, lstm_units, dense_units):
        super().__init__()
        self.lstm_layer = tf.keras.layers.LSTM(lstm_units, return_sequences=False)
        self.mean_layer = tf.keras.layers.Dense(dense_units, activation='linear', name='mean')
        self.var_layer = tf.keras.layers.Dense(dense_units, activation='softplus', name='variance')
    
    def call(self, inputs):
        lstm_output = self.lstm_layer(inputs)
        mean_pred = self.mean_layer(lstm_output)
        var_pred = self.var_layer(lstm_output)
        
        return mean_pred, var_pred
    
    def loss_function(self, y_true, predictions):
        """Custom loss function for heteroscedastic model"""
        mean_pred, var_pred = predictions
        
        # Negative log-likelihood assuming Gaussian distribution
        nll = 0.5 * tf.log(2 * np.pi * var_pred) + 0.5 * tf.square(y_true - mean_pred) / var_pred
        
        return tf.reduce_mean(nll)
```

#### 6.3.2 Time-Varying Uncertainty
```python
def time_varying_uncertainty_estimation(historical_errors, time_features):
    """
    Model time-varying uncertainty using historical error patterns
    """
    from sklearn.ensemble import RandomForestRegressor
    
    # Use absolute errors as target for uncertainty model
    abs_errors = np.abs(historical_errors)
    
    # Train uncertainty model
    uncertainty_model = RandomForestRegressor(n_estimators=100)
    uncertainty_model.fit(time_features, abs_errors)
    
    return uncertainty_model

def adaptive_prediction_intervals(base_predictions, time_features, 
                                uncertainty_model, confidence_level=0.95):
    """
    Construct adaptive prediction intervals with time-varying width
    """
    # Predict uncertainty for each time point
    predicted_uncertainty = uncertainty_model.predict(time_features)
    
    # Construct intervals
    from scipy import stats
    z_score = stats.norm.ppf(1 - (1 - confidence_level)/2)
    
    lower_bound = base_predictions - z_score * predicted_uncertainty
    upper_bound = base_predictions + z_score * predicted_uncertainty
    
    return lower_bound, upper_bound, predicted_uncertainty
```

---

## Calibration Assessment

### 7.1 Calibration Metrics

#### 7.1.1 Coverage Probability
```python
def calculate_coverage_probability(y_true, prediction_intervals, nominal_level=0.95):
    """
    Calculate empirical coverage probability of prediction intervals
    """
    lower_bounds = prediction_intervals['lower']
    upper_bounds = prediction_intervals['upper']
    
    # Check if true values fall within intervals
    coverage_indicators = (y_true >= lower_bounds) & (y_true <= upper_bounds)
    empirical_coverage = np.mean(coverage_indicators)
    
    # Statistical test for correct coverage
    n = len(y_true)
    expected_coverage = nominal_level
    
    # Standard error under null hypothesis
    se = np.sqrt(expected_coverage * (1 - expected_coverage) / n)
    
    # Z-score for coverage test
    z_score = (empirical_coverage - expected_coverage) / se
    
    # P-value (two-tailed test)
    from scipy import stats
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_score)))
    
    coverage_results = {
        'empirical_coverage': empirical_coverage,
        'nominal_coverage': nominal_level,
        'coverage_error': empirical_coverage - nominal_level,
        'z_score': z_score,
        'p_value': p_value,
        'is_well_calibrated': p_value > 0.05
    }
    
    return coverage_results
```

#### 7.1.2 Sharpness (Average Width)
```python
def calculate_sharpness(prediction_intervals):
    """
    Calculate sharpness (average width) of prediction intervals
    """
    interval_widths = prediction_intervals['upper'] - prediction_intervals['lower']
    
    sharpness_metrics = {
        'mean_width': np.mean(interval_widths),
        'median_width': np.median(interval_widths),
        'std_width': np.std(interval_widths),
        'min_width': np.min(interval_widths),
        'max_width': np.max(interval_widths)
    }
    
    return sharpness_metrics
```

#### 7.1.3 Calibration Score
```python
def calculate_calibration_score(y_true, prediction_intervals, confidence_levels):
    """
    Calculate comprehensive calibration score across multiple confidence levels
    """
    calibration_results = {}
    total_calibration_error = 0
    
    for confidence_level in confidence_levels:
        level_key = f'{confidence_level:.0%}'
        intervals = prediction_intervals[level_key]
        
        coverage_result = calculate_coverage_probability(y_true, intervals, confidence_level)
        sharpness_result = calculate_sharpness(intervals)
        
        calibration_error = np.abs(coverage_result['coverage_error'])
        total_calibration_error += calibration_error
        
        calibration_results[level_key] = {
            'coverage': coverage_result,
            'sharpness': sharpness_result,
            'calibration_error': calibration_error
        }
    
    # Overall calibration score (lower is better)
    overall_calibration_score = total_calibration_error / len(confidence_levels)
    
    calibration_results['overall'] = {
        'calibration_score': overall_calibration_score,
        'is_well_calibrated': overall_calibration_score < 0.05  # 5% threshold
    }
    
    return calibration_results
```

### 7.2 Reliability Diagrams

#### 7.2.1 Probability Integral Transform (PIT)
```python
def probability_integral_transform(y_true, predicted_cdf):
    """
    Calculate PIT values for distributional calibration assessment
    """
    pit_values = []
    
    for i, y_val in enumerate(y_true):
        # Calculate CDF value at true observation
        pit_val = predicted_cdf[i](y_val)
        pit_values.append(pit_val)
    
    pit_values = np.array(pit_values)
    
    # PIT histogram should be uniform for well-calibrated predictions
    from scipy import stats
    
    # Kolmogorov-Smirnov test for uniformity
    ks_statistic, ks_p_value = stats.kstest(pit_values, 'uniform')
    
    pit_results = {
        'pit_values': pit_values,
        'ks_statistic': ks_statistic,
        'ks_p_value': ks_p_value,
        'is_uniform': ks_p_value > 0.05
    }
    
    return pit_results

def plot_pit_histogram(pit_values, n_bins=10):
    """
    Create PIT histogram for visual calibration assessment
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # PIT histogram
    ax1.hist(pit_values, bins=n_bins, density=True, alpha=0.7, edgecolor='black')
    ax1.axhline(y=1, color='red', linestyle='--', label='Perfect Calibration')
    ax1.set_xlabel('PIT Values')
    ax1.set_ylabel('Density')
    ax1.set_title('PIT Histogram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot against uniform distribution
    from scipy import stats
    stats.probplot(pit_values, dist=stats.uniform, plot=ax2)
    ax2.set_title('Q-Q Plot vs Uniform Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

#### 7.2.2 Reliability Diagram
```python
def create_reliability_diagram(y_true, predicted_probabilities, n_bins=10):
    """
    Create reliability diagram for probabilistic predictions
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (predicted_probabilities > bin_lower) & (predicted_probabilities <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Calculate empirical accuracy in this bin
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = predicted_probabilities[in_bin].mean()
            
            bin_centers.append((bin_lower + bin_upper) / 2)
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(in_bin.sum())
    
    # Calculate calibration metrics
    reliability_error = np.sum([
        count * np.abs(acc - conf) for acc, conf, count in 
        zip(bin_accuracies, bin_confidences, bin_counts)
    ]) / len(y_true)
    
    # Expected Calibration Error (ECE)
    ece = reliability_error
    
    # Maximum Calibration Error (MCE)
    mce = np.max([np.abs(acc - conf) for acc, conf in zip(bin_accuracies, bin_confidences)])
    
    reliability_results = {
        'bin_centers': bin_centers,
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts,
        'expected_calibration_error': ece,
        'maximum_calibration_error': mce
    }
    
    return reliability_results

def plot_reliability_diagram(reliability_results):
    """
    Plot reliability diagram
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    
    # Reliability curve
    ax.plot(reliability_results['bin_confidences'], 
           reliability_results['bin_accuracies'], 
           'bo-', label='Model Calibration')
    
    # Add confidence intervals based on bin sizes
    for i, (conf, acc, count) in enumerate(zip(
        reliability_results['bin_confidences'],
        reliability_results['bin_accuracies'],
        reliability_results['bin_counts']
    )):
        # Standard error for binomial proportion
        se = np.sqrt(acc * (1 - acc) / count) if count > 0 else 0
        ax.errorbar(conf, acc, yerr=1.96*se, fmt='none', color='blue', alpha=0.5)
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'Reliability Diagram (ECE: {reliability_results["expected_calibration_error"]:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    return fig
```

### 7.3 Calibration Improvement Methods

#### 7.3.1 Platt Scaling
```python
def platt_scaling_calibration(uncalibrated_scores, y_true):
    """
    Apply Platt scaling for probability calibration
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    
    # Fit logistic regression to map scores to probabilities
    platt_model = LogisticRegression()
    platt_model.fit(uncalibrated_scores.reshape(-1, 1), y_true)
    
    # Generate calibrated probabilities
    calibrated_probs = platt_model.predict_proba(uncalibrated_scores.reshape(-1, 1))[:, 1]
    
    return calibrated_probs, platt_model
```

#### 7.3.2 Isotonic Regression
```python
def isotonic_regression_calibration(uncalibrated_scores, y_true):
    """
    Apply isotonic regression for probability calibration
    """
    from sklearn.isotonic import IsotonicRegression
    
    # Fit isotonic regression
    isotonic_model = IsotonicRegression(out_of_bounds='clip')
    calibrated_probs = isotonic_model.fit_transform(uncalibrated_scores, y_true)
    
    return calibrated_probs, isotonic_model
```

#### 7.3.3 Temperature Scaling
```python
def temperature_scaling_calibration(logits, y_true, validation_logits, validation_y_true):
    """
    Apply temperature scaling for neural network calibration
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    class TemperatureScaling(nn.Module):
        def __init__(self):
            super().__init__()
            self.temperature = nn.Parameter(torch.ones(1))
        
        def forward(self, logits):
            return logits / self.temperature
    
    # Initialize temperature scaling model
    temp_model = TemperatureScaling()
    
    # Convert to tensors
    val_logits_tensor = torch.FloatTensor(validation_logits)
    val_labels_tensor = torch.LongTensor(validation_y_true)
    
    # Optimize temperature on validation set
    optimizer = optim.LBFGS([temp_model.temperature], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()
    
    def eval():
        optimizer.zero_grad()
        loss = criterion(temp_model(val_logits_tensor), val_labels_tensor)
        loss.backward()
        return loss
    
    optimizer.step(eval)
    
    # Apply temperature scaling to test logits
    test_logits_tensor = torch.FloatTensor(logits)
    calibrated_logits = temp_model(test_logits_tensor).detach().numpy()
    
    return calibrated_logits, temp_model.temperature.item()
```

---

## Implementation Framework

### 8.1 Comprehensive Uncertainty Quantification System

```python
class UncertaintyQuantificationFramework:
    """
    Comprehensive framework for uncertainty quantification in energy forecasting
    """
    
    def __init__(self, base_model, uncertainty_method='ensemble'):
        self.base_model = base_model
        self.uncertainty_method = uncertainty_method
        self.calibration_data = None
        self.uncertainty_models = {}
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train uncertainty quantification framework
        """
        if self.uncertainty_method == 'ensemble':
            self._fit_ensemble(X_train, y_train, X_val, y_val)
        elif self.uncertainty_method == 'bayesian':
            self._fit_bayesian(X_train, y_train, X_val, y_val)
        elif self.uncertainty_method == 'mc_dropout':
            self._fit_mc_dropout(X_train, y_train, X_val, y_val)
        elif self.uncertainty_method == 'quantile':
            self._fit_quantile_regression(X_train, y_train, X_val, y_val)
        
        # Store calibration data
        if X_val is not None and y_val is not None:
            self.calibration_data = (X_val, y_val)
    
    def _fit_ensemble(self, X_train, y_train, X_val, y_val):
        """Fit ensemble of models for uncertainty quantification"""
        self.ensemble_models = create_bootstrap_ensemble(
            self.base_model, X_train, y_train, n_models=10
        )
    
    def _fit_bayesian(self, X_train, y_train, X_val, y_val):
        """Fit Bayesian neural network"""
        # Implementation of variational inference
        self.bayesian_model = self._build_bayesian_model()
        self.bayesian_model.fit(X_train, y_train, 
                               validation_data=(X_val, y_val))
    
    def _fit_mc_dropout(self, X_train, y_train, X_val, y_val):
        """Fit model with Monte Carlo dropout"""
        self.mc_dropout_model = self._build_mc_dropout_model()
        self.mc_dropout_model.fit(X_train, y_train,
                                 validation_data=(X_val, y_val))
    
    def _fit_quantile_regression(self, X_train, y_train, X_val, y_val):
        """Fit quantile regression models"""
        quantiles = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]
        self.quantile_models = {}
        
        for q in quantiles:
            model = clone_model(self.base_model)
            model.compile(loss=quantile_loss(q))
            model.fit(X_train, y_train, validation_data=(X_val, y_val))
            self.quantile_models[q] = model
    
    def predict_with_uncertainty(self, X_test, confidence_levels=[0.68, 0.95]):
        """
        Generate predictions with uncertainty quantification
        """
        if self.uncertainty_method == 'ensemble':
            return self._ensemble_predict(X_test, confidence_levels)
        elif self.uncertainty_method == 'bayesian':
            return self._bayesian_predict(X_test, confidence_levels)
        elif self.uncertainty_method == 'mc_dropout':
            return self._mc_dropout_predict(X_test, confidence_levels)
        elif self.uncertainty_method == 'quantile':
            return self._quantile_predict(X_test, confidence_levels)
    
    def _ensemble_predict(self, X_test, confidence_levels):
        """Generate ensemble predictions with uncertainty"""
        predictions = []
        for model in self.ensemble_models:
            pred = model.predict(X_test)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        uncertainty_results = ensemble_uncertainty_estimation(
            self.ensemble_models, X_test
        )
        
        # Construct prediction intervals
        intervals = construct_prediction_intervals(
            predictions, confidence_levels
        )
        
        return {
            'mean': uncertainty_results['mean'],
            'std': uncertainty_results['std'],
            'intervals': intervals,
            'raw_predictions': predictions
        }
    
    def _bayesian_predict(self, X_test, confidence_levels):
        """Generate Bayesian predictions with uncertainty"""
        # Sample from posterior distribution
        n_samples = 100
        predictions = []
        
        for _ in range(n_samples):
            pred = self.bayesian_model(X_test, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Construct intervals
        intervals = construct_prediction_intervals(
            predictions, confidence_levels
        )
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'intervals': intervals,
            'epistemic_uncertainty': std_pred
        }
    
    def _mc_dropout_predict(self, X_test, confidence_levels):
        """Generate MC Dropout predictions with uncertainty"""
        mean_pred, var_pred, std_pred = mc_dropout_uncertainty(
            self.mc_dropout_model, X_test, n_samples=100
        )
        
        # Construct Gaussian intervals
        intervals = gaussian_prediction_intervals(
            mean_pred, std_pred, confidence_levels
        )
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'variance': var_pred,
            'intervals': intervals
        }
    
    def _quantile_predict(self, X_test, confidence_levels):
        """Generate quantile regression predictions"""
        quantile_predictions = {}
        
        for q, model in self.quantile_models.items():
            pred = model.predict(X_test)
            quantile_predictions[q] = pred
        
        # Construct intervals from quantiles
        intervals = {}
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            lower_q = alpha / 2
            upper_q = 1 - alpha / 2
            
            if lower_q in quantile_predictions and upper_q in quantile_predictions:
                intervals[f'{conf_level:.0%}'] = {
                    'lower': quantile_predictions[lower_q],
                    'upper': quantile_predictions[upper_q],
                    'width': quantile_predictions[upper_q] - quantile_predictions[lower_q]
                }
        
        return {
            'median': quantile_predictions.get(0.5),
            'quantiles': quantile_predictions,
            'intervals': intervals
        }
    
    def calibrate_uncertainty(self, method='isotonic'):
        """
        Calibrate uncertainty estimates using validation data
        """
        if self.calibration_data is None:
            raise ValueError("No calibration data available")
        
        X_cal, y_cal = self.calibration_data
        
        # Get uncalibrated predictions
        uncalibrated_results = self.predict_with_uncertainty(X_cal)
        
        if method == 'isotonic':
            calibrated_model = self._isotonic_calibration(
                uncalibrated_results, y_cal
            )
        elif method == 'platt':
            calibrated_model = self._platt_calibration(
                uncalibrated_results, y_cal
            )
        
        self.calibration_model = calibrated_model
        return calibrated_model
    
    def evaluate_uncertainty_quality(self, X_test, y_test, confidence_levels=[0.68, 0.95]):
        """
        Evaluate quality of uncertainty quantification
        """
        # Generate predictions
        uncertainty_results = self.predict_with_uncertainty(X_test, confidence_levels)
        
        # Calculate calibration metrics
        calibration_results = {}
        
        for conf_level in confidence_levels:
            level_key = f'{conf_level:.0%}'
            if level_key in uncertainty_results['intervals']:
                intervals = uncertainty_results['intervals'][level_key]
                
                coverage_result = calculate_coverage_probability(
                    y_test, intervals, conf_level
                )
                sharpness_result = calculate_sharpness(intervals)
                
                calibration_results[level_key] = {
                    'coverage': coverage_result,
                    'sharpness': sharpness_result
                }
        
        # Additional uncertainty quality metrics
        if 'std' in uncertainty_results:
            # Uncertainty correlation with absolute errors
            abs_errors = np.abs(y_test - uncertainty_results['mean'])
            uncertainty_correlation = np.corrcoef(
                abs_errors.flatten(), 
                uncertainty_results['std'].flatten()
            )[0, 1]
            
            calibration_results['uncertainty_correlation'] = uncertainty_correlation
        
        return calibration_results

def quantile_loss(quantile):
    """Custom quantile loss function for Keras"""
    def loss(y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(quantile * error, (quantile - 1) * error))
    return loss
```

### 8.2 Energy-Specific Uncertainty Applications

```python
class EnergyUncertaintyApplications:
    """
    Energy-specific applications of uncertainty quantification
    """
    
    def __init__(self, uq_framework):
        self.uq_framework = uq_framework
    
    def reserve_requirement_calculation(self, demand_forecast_uncertainty, 
                                      confidence_level=0.95):
        """
        Calculate spinning reserve requirements based on demand uncertainty
        """
        # Extract uncertainty bounds
        demand_intervals = demand_forecast_uncertainty['intervals'][f'{confidence_level:.0%}']
        
        # Reserve requirement = difference between upper bound and mean forecast
        mean_demand = demand_forecast_uncertainty['mean']
        upper_bound = demand_intervals['upper']
        
        reserve_requirement = upper_bound - mean_demand
        
        # Additional safety margin
        safety_margin = 0.05 * mean_demand  # 5% safety margin
        total_reserve = reserve_requirement + safety_margin
        
        return {
            'base_reserve': reserve_requirement,
            'safety_margin': safety_margin,
            'total_reserve': total_reserve,
            'reserve_percentage': (total_reserve / mean_demand) * 100
        }
    
    def renewable_curtailment_risk(self, renewable_forecast_uncertainty,
                                  grid_capacity, confidence_level=0.95):
        """
        Assess risk of renewable energy curtailment
        """
        # Get renewable generation intervals
        renewable_intervals = renewable_forecast_uncertainty['intervals'][f'{confidence_level:.0%}']
        
        # Calculate curtailment probability
        upper_bound = renewable_intervals['upper']
        curtailment_risk = np.mean(upper_bound > grid_capacity)
        
        # Expected curtailed energy
        excess_generation = np.maximum(0, upper_bound - grid_capacity)
        expected_curtailment = np.mean(excess_generation)
        
        return {
            'curtailment_probability': curtailment_risk,
            'expected_curtailment_mwh': expected_curtailment,
            'curtailment_percentage': (expected_curtailment / np.mean(upper_bound)) * 100
        }
    
    def economic_dispatch_uncertainty(self, demand_uncertainty, 
                                    renewable_uncertainty, 
                                    generation_costs):
        """
        Incorporate uncertainty into economic dispatch optimization
        """
        # Sample from uncertainty distributions
        n_scenarios = 1000
        
        demand_samples = self._sample_from_uncertainty(demand_uncertainty, n_scenarios)
        renewable_samples = self._sample_from_uncertainty(renewable_uncertainty, n_scenarios)
        
        # Calculate net demand for each scenario
        net_demand_samples = demand_samples - renewable_samples
        
        # Optimize dispatch for each scenario
        dispatch_costs = []
        for net_demand in net_demand_samples:
            optimal_dispatch = self._optimize_dispatch(net_demand, generation_costs)
            dispatch_costs.append(optimal_dispatch['total_cost'])
        
        # Calculate expected cost and risk metrics
        expected_cost = np.mean(dispatch_costs)
        cost_std = np.std(dispatch_costs)
        cost_var = np.percentile(dispatch_costs, 95)  # 95% VaR
        
        return {
            'expected_cost': expected_cost,
            'cost_standard_deviation': cost_std,
            'cost_value_at_risk_95': cost_var,
            'cost_scenarios': dispatch_costs
        }
    
    def _sample_from_uncertainty(self, uncertainty_results, n_samples):
        """Sample from uncertainty distribution"""
        if 'raw_predictions' in uncertainty_results:
            # Use ensemble samples
            ensemble_preds = uncertainty_results['raw_predictions']
            n_models = ensemble_preds.shape[0]
            
            samples = []
            for _ in range(n_samples):
                model_idx = np.random.randint(0, n_models)
                sample_idx = np.random.randint(0, ensemble_preds.shape[1])
                samples.append(ensemble_preds[model_idx, sample_idx])
            
            return np.array(samples)
        else:
            # Use Gaussian approximation
            mean = uncertainty_results['mean']
            std = uncertainty_results['std']
            
            return np.random.normal(mean, std, n_samples)
    
    def _optimize_dispatch(self, net_demand, generation_costs):
        """Simplified economic dispatch optimization"""
        # Sort generators by marginal cost
        sorted_generators = sorted(generation_costs.items(), key=lambda x: x[1]['marginal_cost'])
        
        total_cost = 0
        remaining_demand = net_demand
        dispatch = {}
        
        for gen_name, gen_data in sorted_generators:
            if remaining_demand <= 0:
                dispatch[gen_name] = 0
                continue
            
            # Dispatch up to capacity or remaining demand
            dispatched = min(remaining_demand, gen_data['capacity'])
            dispatch[gen_name] = dispatched
            
            # Calculate cost
            cost = dispatched * gen_data['marginal_cost']
            total_cost += cost
            
            remaining_demand -= dispatched
        
        return {
            'dispatch': dispatch,
            'total_cost': total_cost,
            'unserved_energy': max(0, remaining_demand)
        }
```

---

## Evaluation Metrics

### 9.1 Comprehensive Uncertainty Evaluation

```python
def comprehensive_uncertainty_evaluation(y_true, uncertainty_predictions, 
                                       confidence_levels=[0.68, 0.95]):
    """
    Perform comprehensive evaluation of uncertainty quantification quality
    """
    evaluation_results = {
        'calibration_metrics': {},
        'sharpness_metrics': {},
        'scoring_rules': {},
        'distributional_tests': {}
    }
    
    # 1. Calibration Assessment
    for conf_level in confidence_levels:
        level_key = f'{conf_level:.0%}'
        
        if level_key in uncertainty_predictions['intervals']:
            intervals = uncertainty_predictions['intervals'][level_key]
            
            # Coverage probability
            coverage_result = calculate_coverage_probability(y_true, intervals, conf_level)
            
            # Sharpness
            sharpness_result = calculate_sharpness(intervals)
            
            evaluation_results['calibration_metrics'][level_key] = coverage_result
            evaluation_results['sharpness_metrics'][level_key] = sharpness_result
    
    # 2. Proper Scoring Rules
    if 'mean' in uncertainty_predictions and 'std' in uncertainty_predictions:
        mean_pred = uncertainty_predictions['mean']
        std_pred = uncertainty_predictions['std']
        
        # Continuous Ranked Probability Score (CRPS)
        crps_score = calculate_crps_gaussian(y_true, mean_pred, std_pred)
        
        # Logarithmic Score
        log_score = calculate_logarithmic_score(y_true, mean_pred, std_pred)
        
        # Energy Score
        energy_score = calculate_energy_score(y_true, mean_pred, std_pred)
        
        evaluation_results['scoring_rules'] = {
            'crps': crps_score,
            'logarithmic_score': log_score,
            'energy_score': energy_score
        }
    
    # 3. Distributional Tests
    if 'raw_predictions' in uncertainty_predictions:
        raw_preds = uncertainty_predictions['raw_predictions']
        
        # PIT analysis
        predicted_cdf = create_empirical_cdf_from_samples(raw_preds)
        pit_result = probability_integral_transform(y_true, predicted_cdf)
        
        evaluation_results['distributional_tests']['pit'] = pit_result
    
    # 4. Overall Uncertainty Quality Score
    overall_score = calculate_overall_uncertainty_score(evaluation_results)
    evaluation_results['overall_score'] = overall_score
    
    return evaluation_results

def calculate_crps_gaussian(y_true, mean_pred, std_pred):
    """Calculate Continuous Ranked Probability Score for Gaussian predictions"""
    from scipy import stats
    
    # Standardize
    z = (y_true - mean_pred) / std_pred
    
    # CRPS formula for Gaussian distribution
    crps = std_pred * (z * (2 * stats.norm.cdf(z) - 1) + 
                      2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    
    return np.mean(crps)

def calculate_logarithmic_score(y_true, mean_pred, std_pred):
    """Calculate logarithmic score (negative log-likelihood)"""
    # Negative log-likelihood for Gaussian distribution
    log_score = 0.5 * np.log(2 * np.pi * std_pred**2) + \
                0.5 * ((y_true - mean_pred) / std_pred)**2
    
    return np.mean(log_score)

def calculate_energy_score(y_true, mean_pred, std_pred):
    """Calculate energy score for uncertainty evaluation"""
    # Simplified energy score implementation
    # Generate samples from predicted distribution
    n_samples = 100
    samples = np.random.normal(mean_pred[:, None], std_pred[:, None], 
                              (len(mean_pred), n_samples))
    
    # Energy score calculation
    energy_scores = []
    for i in range(len(y_true)):
        true_val = y_true[i]
        pred_samples = samples[i]
        
        # Expected distance to true value
        term1 = np.mean(np.abs(pred_samples - true_val))
        
        # Expected distance between samples
        term2 = 0.5 * np.mean([np.abs(pred_samples[j] - pred_samples[k]) 
                              for j in range(n_samples) 
                              for k in range(n_samples)])
        
        energy_score = term1 - term2
        energy_scores.append(energy_score)
    
    return np.mean(energy_scores)

def calculate_overall_uncertainty_score(evaluation_results):
    """Calculate overall uncertainty quality score"""
    score_components = []
    
    # Calibration component (lower is better for calibration error)
    calibration_errors = []
    for level_metrics in evaluation_results['calibration_metrics'].values():
        calibration_errors.append(abs(level_metrics['coverage_error']))
    
    if calibration_errors:
        avg_calibration_error = np.mean(calibration_errors)
        calibration_score = max(0, 1 - 2 * avg_calibration_error)  # Penalize errors > 50%
        score_components.append(('calibration', calibration_score, 0.4))
    
    # Sharpness component (lower width is better, but normalize by scale)
    sharpness_scores = []
    for level_metrics in evaluation_results['sharpness_metrics'].values():
        # Normalize by range of true values for scale-invariance
        normalized_width = level_metrics['mean_width'] / 4  # Assume 4-sigma range
        sharpness_score = max(0, 1 - normalized_width)
        sharpness_scores.append(sharpness_score)
    
    if sharpness_scores:
        avg_sharpness_score = np.mean(sharpness_scores)
        score_components.append(('sharpness', avg_sharpness_score, 0.3))
    
    # Scoring rules component (lower is better, normalize)
    if 'scoring_rules' in evaluation_results:
        # Use CRPS as representative scoring rule
        crps = evaluation_results['scoring_rules']['crps']
        # Normalize CRPS (assume reasonable range)
        crps_score = max(0, 1 - crps / 100)  # Adjust normalization as needed
        score_components.append(('scoring_rules', crps_score, 0.3))
    
    # Calculate weighted average
    if score_components:
        weighted_score = sum(score * weight for _, score, weight in score_components)
        total_weight = sum(weight for _, _, weight in score_components)
        overall_score = weighted_score / total_weight
    else:
        overall_score = 0.5  # Neutral score if no components available
    
    return {
        'overall_score': overall_score,
        'components': score_components,
        'interpretation': interpret_uncertainty_score(overall_score)
    }

def interpret_uncertainty_score(score):
    """Interpret overall uncertainty quality score"""
    if score >= 0.9:
        return "Excellent uncertainty quantification"
    elif score >= 0.8:
        return "Very good uncertainty quantification"
    elif score >= 0.7:
        return "Good uncertainty quantification"
    elif score >= 0.6:
        return "Acceptable uncertainty quantification"
    elif score >= 0.5:
        return "Poor uncertainty quantification"
    else:
        return "Very poor uncertainty quantification"
```

---

## Practical Applications

### 10.1 Real-Time Energy System Applications

```python
class RealTimeUncertaintySystem:
    """
    Real-time uncertainty quantification system for energy operations
    """
    
    def __init__(self, uq_framework, update_frequency=15):  # 15-minute updates
        self.uq_framework = uq_framework
        self.update_frequency = update_frequency
        self.alert_thresholds = self._define_alert_thresholds()
        self.decision_support = EnergyDecisionSupport()