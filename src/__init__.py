"""
EnergyNexus: Energy System Forecasting and Optimization
========================================================

A comprehensive energy forecasting and optimization system developed for
electricity demand prediction and renewable energy integration analysis.

Author: Aditya Talekar (ec24018@qmul.ac.uk)
Institution: Queen Mary University of London
Program: MSc Data Science and AI - 2024/25
Supervisor: Saqib Iqbal

Project Overview:
-----------------
EnergyNexus is a sophisticated energy analytics platform that combines:
- Advanced LSTM neural networks for electricity demand forecasting
- Multi-variate time series analysis for renewable energy integration
- Comprehensive data quality assessment and preprocessing pipelines
- Energy system optimization algorithms
- Real-time monitoring and evaluation frameworks

Core Modules:
-------------
- data_pipeline: Data collection, processing, and quality assessment
- forecasting: LSTM models and prediction algorithms
- optimization: Energy system optimization and scheduling
- evaluation: Model validation and performance assessment
- energy_system: Domain-specific energy system components
- simulation: System simulation and scenario analysis

Key Features:
-------------
- Multi-horizon electricity demand forecasting (1h, 6h, 24h)
- Renewable energy generation prediction and integration
- Advanced data quality assessment with automated issue detection
- LSTM-based neural network architectures for time series forecasting
- Comprehensive model evaluation and validation frameworks
- Energy system optimization for cost and emissions reduction

Technical Architecture:
-----------------------
The system follows a modular design with clear separation of concerns:
- Data layer: Collection, storage, and preprocessing
- Model layer: LSTM architectures and forecasting algorithms
- Analytics layer: Evaluation, optimization, and insights
- Presentation layer: Visualization and reporting

Dependencies:
-------------
Core scientific computing stack:
- pandas >= 1.5.0 (data manipulation and analysis)
- numpy >= 1.23.0 (numerical computing)
- scikit-learn >= 1.1.0 (machine learning utilities)
- tensorflow >= 2.8.0 (deep learning and LSTM models)

Specialized energy and time series:
- scipy >= 1.9.0 (statistical analysis and signal processing)
- statsmodels >= 0.13.0 (time series analysis)

Data collection and APIs:
- requests >= 2.28.0 (HTTP requests for EIA API)
- pyyaml >= 6.0 (configuration file parsing)

Visualization and reporting:
- matplotlib >= 3.5.0 (plotting and visualization)
- seaborn >= 0.11.0 (statistical data visualization)
- plotly >= 5.10.0 (interactive visualizations)

Usage Examples:
---------------
>>> from data_pipeline.collectors.eia_collector import EIADataCollector
>>> from forecasting.models.lstm_model import LSTMForecaster
>>> from evaluation.validators.data_quality import DataQualityAssessment
>>> 
>>> # Initialize data collection
>>> collector = EIADataCollector()
>>> data = collector.get_electricity_demand(region='US48')
>>> 
>>> # Assess data quality
>>> quality_assessment = DataQualityAssessment(data)
>>> quality_score = quality_assessment.calculate_overall_quality()
>>> 
>>> # Train LSTM forecasting model
>>> forecaster = LSTMForecaster(sequence_length=48, forecast_horizons=[1, 6, 24])
>>> model = forecaster.train(data)
>>> predictions = forecaster.predict(data)

Research Context:
-----------------
This system supports MSc research in:
- Electricity demand forecasting using advanced LSTM architectures
- Renewable energy integration and grid optimization
- Data quality assessment methodologies for energy systems
- Multi-variate time series analysis for energy applications
- Machine learning applications in energy system operations

License and Academic Use:
-------------------------
This software is developed for academic research purposes as part of an
MSc Data Science and AI program at Queen Mary University of London.

For academic collaboration or research inquiries, please contact:
Aditya Talekar - ec24018@qmul.ac.uk

Version Information:
--------------------
"""

# Version information
__version__ = "1.0.0"
__author__ = "Aditya Talekar"
__email__ = "ec24018@qmul.ac.uk"
__institution__ = "Queen Mary University of London"
__program__ = "MSc Data Science and AI"
__year__ = "2024/25"

# Project metadata
__title__ = "EnergyNexus"
__description__ = "Energy System Forecasting and Optimization Platform"
__url__ = "https://github.com/aditya-talekar/EnergyNexus"
__license__ = "Academic Research License"

# Core module imports for convenience
try:
    # Data pipeline components
    from .data_pipeline import collectors, processors, storage
    
    # Forecasting models
    from .forecasting import models, evaluation
    
    # Energy system components  
    from .energy_system import grid, renewable, demand
    
    # Optimization algorithms
    from .optimization import scheduling, cost_optimization
    
    # Evaluation and validation
    from .evaluation import validators, metrics
    
    # Simulation frameworks
    from .simulation import scenarios, monte_carlo
    
    # Set up module availability flags
    DATA_PIPELINE_AVAILABLE = True
    FORECASTING_AVAILABLE = True
    OPTIMIZATION_AVAILABLE = True
    EVALUATION_AVAILABLE = True
    SIMULATION_AVAILABLE = True
    
except ImportError as e:
    # Graceful handling of missing modules during development
    import warnings
    warnings.warn(f"Some EnergyNexus modules not available: {e}", ImportWarning)
    
    # Set availability flags
    DATA_PIPELINE_AVAILABLE = False
    FORECASTING_AVAILABLE = False
    OPTIMIZATION_AVAILABLE = False
    EVALUATION_AVAILABLE = False
    SIMULATION_AVAILABLE = False

# Utility functions for system information
def get_version_info():
    """
    Get comprehensive version and system information.
    
    Returns:
        dict: Version and system information
    """
    return {
        'version': __version__,
        'author': __author__,
        'institution': __institution__,
        'program': __program__,
        'title': __title__,
        'description': __description__,
        'modules_available': {
            'data_pipeline': DATA_PIPELINE_AVAILABLE,
            'forecasting': FORECASTING_AVAILABLE,
            'optimization': OPTIMIZATION_AVAILABLE,
            'evaluation': EVALUATION_AVAILABLE,
            'simulation': SIMULATION_AVAILABLE
        }
    }

def print_system_info():
    """Print comprehensive system information."""
    info = get_version_info()
    
    print("=" * 60)
    print(f"{info['title']} v{info['version']}")
    print(f"{info['description']}")
    print("=" * 60)
    print(f"Author: {info['author']}")
    print(f"Institution: {info['institution']}")
    print(f"Program: {info['program']}")
    print("")
    print("Module Availability:")
    for module, available in info['modules_available'].items():
        status = "✓ Available" if available else "✗ Not Available"
        print(f"  {module}: {status}")
    print("=" * 60)

def check_dependencies():
    """
    Check if required dependencies are installed.
    
    Returns:
        dict: Dictionary of dependency availability
    """
    dependencies = {
        'pandas': False,
        'numpy': False,
        'scikit-learn': False,
        'tensorflow': False,
        'scipy': False,
        'matplotlib': False,
        'seaborn': False,
        'requests': False,
        'yaml': False
    }
    
    # Check each dependency
    for dep in dependencies.keys():
        try:
            if dep == 'scikit-learn':
                import sklearn
            elif dep == 'yaml':
                import yaml
            else:
                __import__(dep)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
    
    return dependencies

def print_dependency_status():
    """Print status of all required dependencies."""
    deps = check_dependencies()
    
    print("Dependency Status:")
    print("-" * 30)
    
    for dep, available in deps.items():
        status = "✓ Installed" if available else "✗ Missing"
        print(f"  {dep}: {status}")
    
    missing = [dep for dep, available in deps.items() if not available]
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
    else:
        print("\n✓ All dependencies are installed!")

# Configuration and setup utilities
def setup_environment():
    """
    Set up the EnergyNexus environment with optimal configuration.
    
    This function configures matplotlib, sets random seeds for reproducibility,
    and applies other system-wide settings.
    """
    import warnings
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Set random seeds for reproducibility
    try:
        import numpy as np
        np.random.seed(42)
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(42)
    except ImportError:
        pass
    
    # Configure matplotlib for publication-quality plots
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("Set1")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        
    except ImportError:
        pass
    
    print("EnergyNexus environment configured successfully!")

# Module constants
DEFAULT_CONFIG = {
    'data': {
        'sequence_length': 48,
        'forecast_horizons': [1, 6, 24],
        'target_variable': 'energy_demand',
        'quality_threshold': 0.8
    },
    'model': {
        'lstm_units': [64, 32],
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100
    },
    'paths': {
        'data_raw': 'data/raw',
        'data_processed': 'data/processed',
        'models': 'models',
        'results': 'results',
        'logs': 'logs'
    }
}

# Export key components for easy access
__all__ = [
    # Version and metadata
    '__version__', '__author__', '__title__', '__description__',
    
    # Utility functions
    'get_version_info', 'print_system_info', 
    'check_dependencies', 'print_dependency_status',
    'setup_environment',
    
    # Configuration
    'DEFAULT_CONFIG',
    
    # Module availability flags
    'DATA_PIPELINE_AVAILABLE', 'FORECASTING_AVAILABLE',
    'OPTIMIZATION_AVAILABLE', 'EVALUATION_AVAILABLE', 
    'SIMULATION_AVAILABLE'
]

# Initialize the environment when module is imported
if __name__ != "__main__":
    # Only run setup if being imported, not if run directly
    try:
        setup_environment()
    except Exception:
        # Silently continue if setup fails
        pass