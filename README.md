# EnergyNexus: LSTM-Based Energy Demand, Energy Price, and Wind Energy Generation Prediction


About Project : # LSTM-Based Energy Demand, Energy Price, and Wind Energy Generation Prediction

A comprehensive deep learning framework for multi-domain energy forecasting using Long Short-Term Memory (LSTM) networks with advanced signal processing techniques.

## 🎯 Overview

This project addresses the unprecedented complexity of modern energy systems by developing three specialized LSTM-based architectures for:

- **Energy Demand Prediction** (95.9% R²)
- **Energy Price Forecasting** (70.3% R²) 
- **Wind Energy Generation Prediction** (93.08% R²)

The research demonstrates successful architectural transfer learning between domains while addressing data acquisition challenges through comprehensive synthetic dataset generation modeled on London metropolitan characteristics.

## 🏗️ Architecture

### 1. Energy Demand Focused LSTM Model
- **Multi-Scale Feature Extraction**: Three parallel CNN layers (kernel sizes: 3, 7, 15)
- **Bidirectional LSTM Processing**: 3-layer architecture (112→84→112 units)
- **Attention Mechanism**: Multi-head attention with 6 heads
- **Temporal Aggregation**: Global average/max pooling + last step extraction
- **Performance**: 95.9% R² across 1h, 6h, 24h forecasting horizons

### 2. Energy Price Focused Model
- **Transfer Learning**: Adapts proven demand architecture
- **Enhanced Regularization**: Increased dropout rates (0.15→0.20, 0.10→0.15)
- **Hybrid Loss Function**: 0.7×Huber + 0.3×MAE
- **Domain-Specific Scaling**: RobustScaler for features, MinMaxScaler for prices
- **Performance**: 70.3% R² with consistent accuracy across horizons

### 3. Wavelet-LSTM Wind Prediction Model
- **Wavelet Preprocessing**: Daubechies 3 (db3) decomposition
- **Multi-Scale Analysis**: Captures frequency components at different temporal scales
- **LSTM Architecture**: Sequential processing of decomposed signals
- **Performance**: 93.08% R² with superior wind pattern recognition

## 📊 Datasets

### Primary Dataset: Synthetic London Energy
- **Coverage**: 10 years (2014-2023), hourly resolution
- **Scale**: 1,051,500 records across 12 London boroughs
- **Features**: 125 engineered variables including:
  - 27 lag features (1-168 hours)
  - 36 rolling window statistics
  - 18 weather variables
  - 15 economic indicators
  - 12 temporal encodings

### Validation Dataset: German Renewable Energy
- **Coverage**: Full year 2016 (8,760 hours)
- **Features**: 23 variables including multi-height wind measurements
- **Quality**: 99.1% data completeness

## 🚀 Key Features

### Advanced Preprocessing
- **Continuous Temporal Sampling**: Preserves autocorrelation structures
- **Hierarchical Feature Engineering**: Domain-specific feature prioritization
- **Wavelet Decomposition**: Multi-resolution signal analysis for wind data

### Model Innovations
- **Architecture Transfer Learning**: Successful adaptation between energy domains
- **Multi-Head Attention**: Automatic temporal importance weighting
- **Ensemble Intelligence**: Dynamic model combination strategies

### Synthetic Data Generation
- **London Metropolitan Modeling**: Realistic infrastructure parameters
- **Multi-Component Weather System**: Temperature, wind, solar resource modeling
- **Market Dynamics**: Comprehensive price formation modeling

## 📈 Results
<img width="500" height="700" alt="9d4920fd-c5e2-4ea9-a125-460173512319" src="https://github.com/user-attachments/assets/78e0899f-3b43-4ecf-9e2b-bd90a2dfce19" />
<img width="500" height="700" alt="d7dce7fd-690a-4498-9078-4a4e36b1a855" src="https://github.com/user-attachments/assets/166c0ada-b091-45df-92a1-2b059ce2e2bc" />

| Model | Domain | R² Score | MAE | MAPE |
|-------|--------|----------|-----|------|
| LSTM-Demand | Energy Demand | 95.9% | 2,725 MW | 4.2% |
| LSTM-Price | Energy Pricing | 70.3% | £8.06/MWh | - |
| Wavelet-LSTM | Wind Generation | 93.08% | 1,754 MW | 21.28% |

### Forecast Horizons Performance
- **1-hour**: 95.7% R² (demand), 69.7% R² (price)
- **6-hour**: 96.8% R² (demand), 69.8% R² (price) 
- **24-hour**: 95.1% R² (demand), 71.3% R² (price)
<img width="500" height="700" alt="7e9d74b3-c802-40c6-a245-40c7245ada15" src="https://github.com/user-attachments/assets/3ad9172e-4d06-4025-9c5d-dafbf0d538ea" />

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/lstm-energy-forecasting.git
cd lstm-energy-forecasting

# Install dependencies
pip install -r requirements.txt

# Setup environment
python setup.py install
```

### Dependencies
```
tensorflow>=2.8.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
pywavelets>=1.1.1
matplotlib>=3.5.0
seaborn>=0.11.0
```

## 💻 Usage

### Quick Start
```python
from models import LSTMEnergyPredictor, WaveletLSTMPredictor
from data import SyntheticLondonGenerator

# Load synthetic London data
generator = SyntheticLondonGenerator()
data = generator.generate_dataset(years=10, boroughs=12)

# Initialize demand prediction model
demand_model = LSTMEnergyPredictor(
    model_type='demand',
    sequence_length=24,
    forecast_horizons=[1, 6, 24]
)

# Train model
demand_model.fit(data, epochs=20, batch_size=64)

# Generate predictions
predictions = demand_model.predict(test_data)
```

### Wind Energy Forecasting
```python
# Initialize wavelet-enhanced wind model
wind_model = WaveletLSTMPredictor(
    wavelet='db3',
    decomposition_levels=2
)

# Preprocess with wavelet decomposition
processed_data = wind_model.wavelet_preprocess(wind_data)

# Train and predict
wind_model.fit(processed_data)
wind_predictions = wind_model.predict(test_wind_data)
```

## 📁 Project Structure

```
lstm-energy-forecasting/
├── data/
│   ├── synthetic_london_generator.py
│   ├── german_renewable_loader.py
│   └── preprocessing.py
├── models/
│   ├── lstm_demand.py
│   ├── lstm_price.py
│   ├── wavelet_lstm_wind.py
│   └── base_model.py
├── utils/
│   ├── feature_engineering.py
│   ├── evaluation_metrics.py
│   └── visualization.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_results_analysis.ipynb
├── configs/
│   └── model_configs.yaml
└── tests/
    └── test_models.py
```

## 🔬 Methodology

### Data Processing Pipeline
1. **Continuous Temporal Sampling**: Maintains 50,000 observation windows
2. **Feature Engineering**: Domain-specific hierarchical prioritization
3. **Quality Optimization**: Forward/backward fill with domain constraints
4. **Sequence Generation**: 24-hour sequences for diurnal pattern capture

### Training Strategy
- **Chronological Splitting**: 70% train, 20% validation, 10% test
- **Advanced Callbacks**: Early stopping, learning rate reduction, model checkpointing
- **Regularization**: Dropout, batch normalization, gradient clipping

## 📊 Evaluation Metrics

- **R² Score**: Variance explanation (primary metric)
- **MAE**: Mean Absolute Error in domain units
- **MAPE**: Mean Absolute Percentage Error
- **RMSE**: Root Mean Square Error
- **Index of Agreement**: Model-observation correspondence

## 🎯 Applications

### Smart Grid Management
- **Load Balancing**: Real-time demand-generation matching
- **Economic Dispatch**: Optimal resource allocation
- **Grid Stability**: Proactive stability management

### Energy Trading
- **Price Forecasting**: Market timing optimization
- **Risk Management**: Volatility prediction
- **Portfolio Optimization**: Multi-asset energy trading

### Renewable Integration
- **Wind Farm Planning**: Optimal turbine placement
- **Grid Integration**: Renewable penetration optimization
- **Storage Management**: Battery scheduling optimization

## 🔮 Future Work

### Immediate Enhancements
- **Real-World Validation**: Partnership with energy utilities
- **Uncertainty Quantification**: Bayesian deep learning integration
- **Ensemble Architectures**: Multi-model combination strategies

### Advanced Development
- **Optimization Integration**: Unit commitment and economic dispatch
- **Transformer Integration**: Hybrid LSTM-Transformer ensembles
- **Multi-Region Scaling**: Continental energy system modeling
```
