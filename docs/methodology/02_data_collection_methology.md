# Data Collection and Preparation Methodology

## Executive Summary

This document outlines the comprehensive data collection and preparation methodology employed in the EnergyNexus project for developing advanced LSTM architectures for energy forecasting. The methodology encompasses both real-world data acquisition from authoritative sources and synthetic data generation for controlled experimentation, ensuring robust model development and validation.

## Data Sources

### Primary Sources

#### 1. Energy Information Administration (EIA) API
**Source:** U.S. Energy Information Administration  
**Endpoint:** `https://api.eia.gov/v2/electricity/rto/region-data/data/`  
**Access Method:** RESTful API with authentication key  
**Update Frequency:** Real-time to hourly updates  
**Coverage:** Regional transmission organization (RTO) data across North America  

**Data Categories:**
- Electricity demand by region (MW)
- Net generation by fuel type (MW)
- Interchange between regions (MW)
- Day-ahead and real-time pricing ($/MWh)

**Data Quality:**
- Official government source with high reliability
- Standardized measurement protocols
- Regular quality control and validation
- Historical data consistency maintained

**Limitations:**
- Geographic coverage primarily North American
- Some regions have limited historical depth
- Real-time data may have brief delays (15-30 minutes)
- API rate limits require careful request management

#### 2. Open-Meteo Weather Data API
**Source:** Open-Meteo Historical Weather API  
**Endpoint:** `https://archive-api.open-meteo.com/v1/archive`  
**Access Method:** Public API with no authentication required  
**Update Frequency:** Hourly historical data, daily updates  
**Coverage:** Global coverage with high spatial resolution  

**Meteorological Variables:**
- Temperature at 2 meters (°C)
- Relative humidity (%)
- Wind speed at 10 meters (m/s)
- Wind direction (degrees)
- Solar radiation (W/m²)
- Precipitation (mm)
- Cloud cover (%)
- Atmospheric pressure (hPa)

**Data Quality:**
- ERA5 reanalysis data from ECMWF
- High spatial resolution (0.25° × 0.25°)
- Consistent temporal coverage since 1940
- Quality-controlled and bias-corrected

**Processing Requirements:**
- Coordinate-based data extraction
- Temporal alignment with energy data
- Unit conversion and standardization
- Spatial interpolation for specific locations

#### 3. Synthetic Data Generation Framework
**Purpose:** Controlled experimental environment for model development  
**Methodology:** Physics-based simulation with stochastic components  
**Validation:** Cross-referenced with real-world patterns and statistics  

### Secondary Sources

#### Grid System Indicators
- **Frequency Data:** Real-time grid frequency measurements (Hz)
- **Voltage Data:** Transmission system voltage levels (kV)
- **Capacity Factors:** Renewable energy capacity utilization rates (%)
- **Market Data:** Energy trading prices and volumes

#### Economic Indicators
- **Fuel Prices:** Natural gas, coal, and oil pricing data
- **Carbon Pricing:** Emissions trading system prices
- **Demand Response:** Load curtailment and shifting programs
- **Economic Activity:** Industrial production and commercial activity indices

## Data Types and Specifications

### Energy Demand Data
**Variable Name:** `energy_demand`  
**Units:** Megawatts (MW)  
**Temporal Resolution:** Hourly  
**Data Type:** Continuous numerical  
**Range:** 200-2000 MW (typical regional range)  
**Missing Data Tolerance:** <2% acceptable  

**Characteristics:**
- Strong daily periodicity (24-hour cycles)
- Weekly patterns (weekday/weekend differences)
- Seasonal variations (heating/cooling demands)
- Weather dependency (temperature correlation)
- Economic activity correlation

### Renewable Generation Data

#### Solar Generation
**Variable Name:** `solar_generation`  
**Units:** Megawatts (MW)  
**Temporal Resolution:** Hourly  
**Data Type:** Continuous numerical (≥0)  
**Dependencies:** Solar irradiance, cloud cover, temperature  

**Generation Model:**
```
Solar_Output = Nameplate_Capacity × Solar_Irradiance × Temperature_Efficiency × Cloud_Attenuation
```

**Key Factors:**
- Solar elevation angle (time of day, season)
- Cloud cover impact (0-100% attenuation)
- Temperature effects on panel efficiency
- Seasonal irradiance variations

#### Wind Generation
**Variable Name:** `wind_generation`  
**Units:** Megawatts (MW)  
**Temporal Resolution:** Hourly  
**Data Type:** Continuous numerical (≥0)  
**Dependencies:** Wind speed, air density, turbine characteristics  

**Power Curve Model:**
- Cut-in speed: 3 m/s (generation begins)
- Rated speed: 12 m/s (maximum output)
- Cut-out speed: 25 m/s (safety shutdown)
- Cubic relationship in operational range

### Weather Variables

#### Temperature
**Variable Name:** `temperature`  
**Units:** Degrees Celsius (°C)  
**Temporal Resolution:** Hourly  
**Measurement Height:** 2 meters above ground  
**Range:** -30°C to +50°C (regional variations)  

**Impact on Energy Systems:**
- Heating demand (inverse relationship below 18°C)
- Cooling demand (positive relationship above 22°C)
- Solar panel efficiency (negative relationship above 25°C)
- Transmission line capacity (thermal effects)

#### Wind Speed and Direction
**Variable Names:** `wind_speed`, `wind_direction`  
**Units:** m/s, degrees  
**Temporal Resolution:** Hourly  
**Measurement Height:** 10 meters above ground  
**Range:** 0-40 m/s, 0-360 degrees  

#### Solar Irradiance and Cloud Cover
**Variable Names:** `solar_radiation`, `cloud_cover`  
**Units:** W/m², percentage  
**Temporal Resolution:** Hourly  
**Dependencies:** Solar angle, atmospheric conditions  

### Grid System Indicators

#### Grid Frequency
**Variable Name:** `grid_frequency`  
**Units:** Hertz (Hz)  
**Nominal Value:** 50.0 Hz (Europe/Asia) or 60.0 Hz (North America)  
**Operational Range:** ±0.5 Hz from nominal  
**Significance:** Real-time supply-demand balance indicator  

#### Energy Price
**Variable Name:** `energy_price`  
**Units:** Currency per MWh ($/MWh, €/MWh)  
**Temporal Resolution:** Hourly (day-ahead) or real-time  
**Range:** 0-500 $/MWh (typical market range)  
**Factors:** Supply-demand balance, fuel costs, grid constraints  

## Data Quality Assurance

### Validation Procedures

#### 1. Missing Data Detection and Treatment

**Detection Methods:**
- Automated null value identification
- Timestamp gap analysis
- Expected range validation
- Temporal consistency checks

**Treatment Strategies:**
```python
# Missing Data Handling Hierarchy
1. Linear interpolation (gaps < 3 hours)
2. Seasonal decomposition interpolation (3-24 hour gaps)
3. Similar day substitution (24+ hour gaps)
4. Statistical imputation using related variables
5. Data exclusion (if >48 hour gap or >5% missing in period)
```

**Quality Metrics:**
- Missing data percentage per variable
- Interpolation method usage statistics
- Validation against holdout known values
- Temporal distribution of missing data

#### 2. Outlier Identification and Treatment

**Detection Methods:**
- Statistical outliers (3-sigma rule)
- Physical constraint violations
- Temporal anomaly detection
- Cross-variable consistency checks

**Physical Constraints:**
```python
# Energy System Constraints
- energy_demand: > 0 MW
- solar_generation: 0 ≤ value ≤ nameplate_capacity
- wind_generation: 0 ≤ value ≤ nameplate_capacity  
- temperature: -50°C ≤ value ≤ 60°C
- wind_speed: 0 ≤ value ≤ 50 m/s
- grid_frequency: 49.5 ≤ value ≤ 50.5 Hz (European standard)
```

**Treatment Procedures:**
1. **Soft Outliers (2-3 sigma):** Flag for review, apply smoothing
2. **Hard Outliers (>3 sigma):** Replace with interpolated values
3. **Physical Violations:** Clamp to valid ranges or interpolate
4. **Systematic Errors:** Investigate source and apply corrections

#### 3. Temporal Consistency Validation

**Consistency Checks:**
- Monotonic timestamp progression
- Regular interval validation (no skipped hours)
- Seasonal pattern consistency
- Day-of-week pattern validation

**Temporal Quality Metrics:**
- Timestamp regularity score
- Seasonal decomposition residuals
- Autocorrelation function analysis
- Periodogram for frequency domain validation

#### 4. Cross-Variable Correlation Validation

**Expected Correlations:**
- Temperature vs. energy demand (U-shaped relationship)
- Solar irradiance vs. solar generation (positive, with efficiency curve)
- Wind speed vs. wind generation (cubic relationship in operating range)
- Energy demand vs. price (positive correlation with lag effects)

**Validation Procedures:**
```python
# Correlation Validation Framework
1. Calculate rolling correlation coefficients
2. Compare against expected ranges
3. Identify periods of unusual correlation
4. Investigate and document anomalies
5. Apply corrections or flag for exclusion
```

### Data Preprocessing Pipeline

#### 1. Temporal Alignment and Resampling

**Standardization Process:**
- Convert all timestamps to UTC
- Resample to consistent hourly intervals
- Handle daylight saving time transitions
- Align data from multiple sources

**Resampling Methods:**
- **Energy Data:** Sum for cumulative values, mean for instantaneous
- **Weather Data:** Linear interpolation for continuous variables
- **Price Data:** Forward fill for step functions
- **System Indicators:** Appropriate aggregation by variable type

#### 2. Feature Engineering and Cyclical Encoding

**Temporal Features:**
```python
# Cyclical Encoding for Temporal Patterns
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
day_of_week_sin = sin(2π × day_of_week / 7)  
day_of_week_cos = cos(2π × day_of_week / 7)
month_sin = sin(2π × month / 12)
month_cos = cos(2π × month / 12)
day_of_year_sin = sin(2π × day_of_year / 365.25)
day_of_year_cos = cos(2π × day_of_year / 365.25)
```

**Derived Features:**
- **Renewable Penetration:** `(solar + wind) / total_demand × 100`
- **Supply-Demand Balance:** `total_generation - energy_demand`
- **Temperature Heating/Cooling Degree Days**
- **Business Hours Indicator:** Boolean for operational hours
- **Peak/Off-Peak Indicators:** Time-based demand classifications

#### 3. Multi-variate Sequence Creation

**Sequence Parameters:**
- **Sequence Length:** 48 hours (captures daily patterns and dependencies)
- **Forecast Horizons:** 1, 6, 24 hours (operational to planning timescales)
- **Stride:** 1 hour (overlapping sequences for maximum data utilization)
- **Padding:** Zero-padding for sequences at boundaries

**Sequence Structure:**
```python
# Input Sequence (X): [batch_size, sequence_length, n_features]
# Shape: [N, 48, 20] for 20 input features over 48 hours

# Output Targets (y): [batch_size, n_targets × n_horizons]  
# Shape: [N, 9] for 3 targets × 3 horizons
# Targets: [demand_1h, demand_6h, demand_24h, solar_1h, solar_6h, solar_24h, wind_1h, wind_6h, wind_24h]
```

#### 4. Train/Validation/Test Splitting

**Temporal Splitting Strategy:**
- **Chronological Order Maintained:** No random shuffling to preserve temporal dependencies
- **Split Ratios:** 70% training, 15% validation, 15% test
- **Buffer Periods:** No gaps between sets to maintain continuity
- **Seasonal Coverage:** Ensure all sets contain representative seasonal patterns

**Split Implementation:**
```python
# Temporal Split Boundaries
total_sequences = len(X_sequences)
train_end = int(0.70 * total_sequences)
val_end = int(0.85 * total_sequences)

X_train = X_sequences[:train_end]
X_val = X_sequences[train_end:val_end]  
X_test = X_sequences[val_end:]

# Ensure no data leakage through sequence overlap
```

## Synthetic Data Generation

### Rationale

#### Controlled Experimental Conditions
- **Known Ground Truth:** Precisely defined relationships for validation
- **Parameter Control:** Ability to modify specific system characteristics
- **Scenario Testing:** Extreme conditions and stress testing capabilities
- **Reproducibility:** Identical datasets for comparative studies

#### Model Development Benefits
- **Debugging and Validation:** Clear cause-effect relationships
- **Algorithm Testing:** Performance under known conditions
- **Edge Case Generation:** Rare events and extreme scenarios
- **Benchmarking:** Standardized comparison datasets

### Generation Methodology

#### Physics-Based Energy System Modeling

**Energy Demand Modeling:**
```python
# Multi-component Demand Model
base_demand = 500  # MW baseline
daily_pattern = 180 × max(0, sin((hour - 6) × π / 12))  # Daily cycle
weekly_pattern = 60 × sin((hour % 168) × 2π / 168)     # Weekly cycle
seasonal_pattern = 80 × sin(day_of_year × 2π / 365.25) # Annual cycle

# Weather Dependencies
heating_demand = max(0, (18 - temperature) × 15)        # Heating below 18°C
cooling_demand = max(0, (temperature - 22) × 20)        # Cooling above 22°C

# Economic Activity
business_multiplier = 1.0 + 0.3 × is_business_hours    # Commercial activity
weekend_factor = 0.85 if is_weekend else 1.0           # Weekend reduction

# Stochastic Component
noise = AutoRegressive(φ=0.3) + White_Noise(σ=25)      # Realistic noise structure

total_demand = (base_demand + daily_pattern + weekly_pattern + seasonal_pattern + 
                heating_demand + cooling_demand) × business_multiplier × weekend_factor + noise
```

**Renewable Generation Modeling:**

**Solar Generation:**
```python
# Solar Power Model
solar_elevation = max(0, sin((hour - 12) × π / 12))     # Solar angle
seasonal_factor = 1 + 0.3 × sin(day_of_year × 2π / 365) # Seasonal variation
cloud_attenuation = 1 - cloud_cover × 0.8              # Cloud impact
temperature_efficiency = 1 - max(0, (temperature - 25) × 0.004)  # Temperature derating

solar_generation = nameplate_capacity × solar_elevation × seasonal_factor × 
                   cloud_attenuation × temperature_efficiency + noise
```

**Wind Generation:**
```python
# Wind Power Curve Model
def wind_power_curve(wind_speed):
    if wind_speed < 3:        # Below cut-in
        return 0
    elif wind_speed < 12:     # Cubic region  
        return nameplate_capacity × ((wind_speed - 3) / 9)³
    elif wind_speed < 25:     # Rated region
        return nameplate_capacity
    else:                     # Above cut-out
        return 0

# Add persistence and variability
wind_generation = wind_power_curve(wind_speed) + AutoRegressive_Noise(φ=0.7, σ=15)
```

#### Realistic Temporal Patterns and Correlations

**Cross-Variable Correlations:**
- **Temperature-Demand:** U-shaped relationship with heating/cooling thresholds
- **Wind Speed-Generation:** Cubic power curve with cut-in/cut-out
- **Solar-Cloud:** Inverse relationship with realistic attenuation
- **Price-Demand:** Economic dispatch and market dynamics

**Temporal Dependencies:**
- **Autoregressive Components:** AR(1) processes for realistic persistence
- **Seasonal Cycles:** Multiple nested periodicities (daily, weekly, annual)
- **Weather Patterns:** Realistic meteorological progression
- **Economic Cycles:** Business day and seasonal economic activity

#### Weather-Dependent Renewable Generation

**Meteorological Modeling:**
```python
# Realistic Weather Generation
temperature = baseline_temp + seasonal_cycle + daily_cycle + weather_noise
wind_speed = wind_climatology + frontal_systems + turbulence
cloud_cover = cloud_climatology + weather_systems + local_effects
solar_radiation = clear_sky_model × (1 - cloud_attenuation)
```

**Physical Constraints:**
- Energy conservation principles
- Thermodynamic relationships
- Atmospheric physics constraints
- Grid stability requirements

#### Economic Factors in Demand Modeling

**Price Formation Model:**
```python
# Market-Based Pricing
marginal_cost = fuel_prices + operational_costs + emissions_costs
supply_curve = economic_dispatch(generation_mix, marginal_cost)
demand_curve = price_elasticity(base_demand, economic_factors)
market_price = market_clearing(supply_curve, demand_curve)

# Add market dynamics
price_volatility = GARCH_model(historical_volatility)
final_price = market_price + volatility + transmission_constraints
```

**Economic Drivers:**
- Industrial production cycles
- Commercial activity patterns
- Residential consumption habits
- Seasonal economic variations

### Data Validation and Quality Control

#### Statistical Validation
- **Distribution Matching:** Compare synthetic vs. real data distributions
- **Correlation Preservation:** Validate cross-variable relationships
- **Spectral Analysis:** Frequency domain comparison
- **Extreme Value Analysis:** Tail behavior validation

#### Physics Validation
- **Energy Balance:** Conservation law verification
- **Physical Constraints:** Feasibility checks
- **System Stability:** Grid operation requirements
- **Causality:** Proper cause-effect relationships

#### Temporal Validation
- **Autocorrelation Functions:** Time dependency validation
- **Seasonal Decomposition:** Pattern component verification
- **Trend Analysis:** Long-term behavior consistency
- **Change Point Detection:** Structural break identification

## Data Storage and Management

### Database Architecture
- **Time Series Database:** InfluxDB for high-frequency temporal data
- **Relational Database:** PostgreSQL for metadata and relationships
- **File Storage:** HDF5 for large array datasets
- **Version Control:** Git LFS for data versioning

### Data Governance
- **Access Controls:** Role-based permissions
- **Audit Trails:** Complete data lineage tracking
- **Backup Procedures:** Redundant storage systems
- **Privacy Compliance:** Data anonymization protocols

### Performance Optimization
- **Indexing Strategy:** Time-based partitioning
- **Compression:** Lossless compression for storage efficiency
- **Caching:** In-memory cache for frequently accessed data
- **Parallel Processing:** Distributed data processing capabilities

## Quality Metrics and Reporting

### Data Quality Scorecard
- **Completeness:** Percentage of non-missing values
- **Accuracy:** Validation against reference sources
- **Consistency:** Cross-variable and temporal consistency
- **Timeliness:** Data freshness and update frequency
- **Validity:** Adherence to business rules and constraints

### Automated Quality Monitoring
- **Real-time Validation:** Continuous quality checks
- **Alert Systems:** Automated notification of quality issues
- **Dashboard Reporting:** Visual quality metrics tracking
- **Trend Analysis:** Quality degradation detection

### Documentation Standards
- **Metadata Cataloging:** Comprehensive data dictionaries
- **Lineage Tracking:** Complete data provenance
- **Processing Logs:** Detailed transformation records
- **Quality Reports:** Regular assessment summaries

This comprehensive methodology ensures robust, high-quality data for advanced LSTM model development and provides the foundation for reliable energy forecasting research.
```

This detailed methodology document covers all aspects of data collection and preparation for your EnergyNexus project. You can copy-paste this directly into your `02_data_collection_methodology.md` file.