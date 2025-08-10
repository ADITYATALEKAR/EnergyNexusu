"""
Forecasting Models Package
EnergyNexus MSc Project

This package contains machine learning models for energy forecasting in hybrid power systems.
The models are specifically designed to handle the temporal dependencies and non-linear
relationships present in renewable energy generation data.

Current Models:
- MultiHorizonEnergyLSTM: Deep learning model for multi-horizon renewable energy forecasting
- EnergyLSTMForecaster: Complete forecasting system with training and evaluation capabilities

Future Extensions:
- Transformer-based models for long-sequence forecasting
- Ensemble methods combining multiple forecasting approaches
- Physics-informed neural networks incorporating energy system constraints

Author: Aditya Talekar (ec24018@qmul.ac.uk)
Supervisor: Saqib Iqbal
QMUL MSc Data Science and AI - 2024/25
"""

# I import the main forecasting classes to make them available at package level
# This allows users to import directly from the package: from forecasting.models import EnergyLSTMForecaster
from .lstm_model import MultiHorizonEnergyLSTM, EnergyLSTMForecaster

# Package version information for thesis documentation and reproducibility
__version__ = "1.0.0"

# I make the main classes available when someone imports the package
# This follows Python best practices for package organization
__all__ = [
    "MultiHorizonEnergyLSTM",      # The core LSTM neural network architecture
    "EnergyLSTMForecaster"         # The complete forecasting system wrapper
]

# Package metadata for academic documentation and citations
__author__ = "Aditya Talekar"
__email__ = "ec24018@qmul.ac.uk" 
__institution__ = "Queen Mary University of London"
__project__ = "EnergyNexus: Advanced Scheduling Algorithms for Integrated Power Systems"
__research_area__ = "Renewable Energy Forecasting using Deep Learning"

# I include model information that will be useful for my thesis documentation
__models_info__ = {
    "MultiHorizonEnergyLSTM": {
        "type": "Deep Learning - LSTM Neural Network",
        "purpose": "Multi-horizon renewable energy forecasting",
        "input": "Time series sequences of energy and weather data",
        "output": "Forecasts for 1h, 6h, and 24h horizons with uncertainty estimates",
        "key_features": [
            "Bidirectional LSTM processing for temporal context",
            "Multi-head attention mechanism for pattern recognition", 
            "Simultaneous multi-horizon prediction capability",
            "Integrated uncertainty quantification",
            "Energy-specific architecture optimizations"
        ]
    },
    "EnergyLSTMForecaster": {
        "type": "Complete Forecasting System",
        "purpose": "End-to-end energy forecasting pipeline",
        "capabilities": [
            "Specialized data preprocessing for energy time series",
            "Advanced training strategies with early stopping",
            "Comprehensive model evaluation with energy-specific metrics",
            "Professional visualization and reporting tools",
            "Model persistence and deployment support"
        ]
    }
}

# I provide usage examples that will be helpful for documentation
__usage_example__ = """
Example usage for energy forecasting:

    from forecasting.models import EnergyLSTMForecaster
    import pandas as pd
    
    # Initialize the forecasting system
    forecaster = EnergyLSTMForecaster(
        sequence_length=48,        # Use 48 hours of historical data
        forecast_horizons=[1, 6, 24],  # Predict 1h, 6h, and 24h ahead
        hidden_size=128,           # LSTM hidden state size
        num_layers=3               # Number of LSTM layers
    )
    
    # Prepare your energy data
    target_columns = ['solar_generation', 'wind_generation']
    feature_columns = ['solar_generation', 'wind_generation', 'temperature', 'hour_sin', 'hour_cos']
    
    X, y, features = forecaster.prepare_data(your_data, target_columns, feature_columns)
    
    # Build and train the model
    forecaster.build_model(input_size=len(features))
    forecaster.train_model(X_train, y_train, X_val, y_val, epochs=100)
    
    # Generate forecasts
    predictions = forecaster.predict(X_test, return_uncertainty=True)
    
    # Evaluate performance
    results = forecaster.evaluate_model(X_test, y_test)
    
    # Create visualizations
    forecaster.plot_training_results()
    forecaster.plot_prediction_results(X_test, y_test)
"""

"""
Talekar, A. (2025). "EnergyNexus: Advanced Scheduling Algorithms for Integrated Power Systems"
MSc Thesis, Queen Mary University of London, School of Electronic Engineering and Computer Science.
Supervisor: Saqib Iqbal.
"""