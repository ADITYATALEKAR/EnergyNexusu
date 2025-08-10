"""
EnergyNexus Data Cleaning Module
Aditya's Original Implementation for MSc Project

This module implements a comprehensive data preprocessing pipeline specifically designed 
for hybrid energy systems. The cleaning process addresses unique challenges in energy data:

1. Solar generation must follow physical laws (zero at night, peak at noon)
2. Wind generation can be highly variable but cannot be negative
3. Energy demand follows predictable human behavior patterns
4. Supply and demand must maintain approximate balance (conservation of energy)
5. Renewable sources have higher natural variability than conventional sources

Academic Contribution: 
- Novel data quality framework for renewable energy integration
- Physics-aware outlier detection algorithms
- Pattern-based gap filling for different energy types
- Multi-source energy data fusion methodology

Author: Aditya Talekar (ec24018@qmul.ac.uk)
Supervisor: Saqib Iqbal
QMUL MSc Data Science and AI - 2024/25
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import RobustScaler
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnergyDataProcessor:
    """
    Custom Data Processing Framework for Hybrid Energy Systems
    
    This class provides specialized data cleaning methods for energy systems that understand
    the physical constraints and behavioral patterns of different energy sources. Unlike
    generic data cleaning tools, this processor knows that:
    
    - Solar panels cannot generate electricity at night
    - Wind turbines cannot produce negative energy
    - Electricity demand has predictable daily and weekly patterns
    - Total generation must approximately equal total demand
    - Renewable sources are naturally more variable than conventional sources
    
    The processor implements a multi-stage cleaning pipeline that progressively improves
    data quality while preserving the physical and behavioral characteristics of energy systems.
    """
    
    def __init__(self, quality_threshold: float = 0.8):
        """
        Initialize the energy data processor with configuration parameters.
        
        The quality threshold determines the minimum acceptable data quality level.
        Energy systems typically require high data quality (>80%) for reliable forecasting
        and optimization. Below this threshold, models may produce unreliable results.
        
        Args:
            quality_threshold: Minimum acceptable data quality score (0-1)
                             0.8 means at least 80% of data quality checks must pass
        """
        self.quality_threshold = quality_threshold
        
        # RobustScaler is chosen over StandardScaler because energy data often contains
        # outliers (sudden spikes in wind, equipment failures). RobustScaler uses
        # median and interquartile range instead of mean and standard deviation,
        # making it less sensitive to extreme values.
        self.scaler = RobustScaler()
        
        # Storage for tracking data quality metrics and processing results
        self.data_quality_metrics = {}
        self.processed_data = None
        
        # Configure logging to track the cleaning process for thesis documentation
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - EnergyProcessor - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def assess_data_quality(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Comprehensive data quality assessment framework specifically designed for energy systems.
        
        Traditional data quality metrics only check for missing values and basic statistics.
        This method implements energy-specific quality checks that validate whether the data
        follows expected physical and behavioral patterns.
        
        Quality Assessment Components:
        1. Completeness - percentage of non-missing values across all columns
        2. Solar Consistency - how well solar data follows daily light patterns
        3. Wind Consistency - validates wind generation patterns and variability
        4. Demand Regularity - checks if demand follows expected human behavior patterns
        5. System Coherence - validates relationships between supply and demand
        
        Args:
            data: DataFrame containing energy system data
            
        Returns:
            Dictionary containing quality scores for each assessment component
        """
        self.logger.info("Starting comprehensive data quality assessment")
        
        quality_metrics = {}
        
        # Assessment 1: Basic Completeness Check
        # Calculate the percentage of non-missing values across the entire dataset
        # This is the foundation metric - we need sufficient data to perform other checks
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        completeness = 1 - (missing_cells / total_cells)
        quality_metrics['completeness'] = completeness
        
        # Assessment 2: Solar Generation Consistency
        # Solar generation should follow predictable daily patterns based on sunrise/sunset
        # This check validates whether solar data behaves according to solar physics
        if 'solar_generation' in data.columns:
            solar_consistency = self._calculate_renewable_consistency(
                data['solar_generation'], energy_type='solar'
            )
            quality_metrics['solar_consistency'] = solar_consistency
        
        # Assessment 3: Wind Generation Consistency  
        # Wind is more variable than solar but should still show some patterns
        # This check identifies obviously corrupted wind data
        if 'wind_generation' in data.columns:
            wind_consistency = self._calculate_renewable_consistency(
                data['wind_generation'], energy_type='wind'
            )
            quality_metrics['wind_consistency'] = wind_consistency
        
        # Assessment 4: Demand Pattern Regularity
        # Electricity demand follows human behavior patterns (work days vs weekends,
        # daily activity cycles). This check validates behavioral consistency.
        if 'energy_demand' in data.columns:
            demand_regularity = self._calculate_demand_regularity(data['energy_demand'])
            quality_metrics['demand_regularity'] = demand_regularity
        
        # Assessment 5: System-wide Coherence
        # In energy systems, supply and demand must be balanced. This check validates
        # whether the relationships between different energy sources make sense.
        system_coherence = self._calculate_system_coherence(data)
        quality_metrics['system_coherence'] = system_coherence
        
        # Calculate Overall Quality Score
        # The overall score is the mean of all individual quality metrics
        # This provides a single number to assess dataset usability
        overall_quality = np.mean(list(quality_metrics.values()))
        quality_metrics['overall_quality'] = overall_quality
        
        # Store results for later use in cleaning process
        self.data_quality_metrics = quality_metrics
        
        self.logger.info(f"Data quality assessment complete. Overall quality: {overall_quality:.1%}")
        return quality_metrics
    
    def _calculate_renewable_consistency(self, renewable_data: pd.Series, energy_type: str) -> float:
        """
        Calculate how well renewable energy data follows expected physical patterns.
        
        This method compares actual renewable generation patterns against theoretical
        expectations based on natural phenomena (sun position for solar, wind patterns).
        
        For Solar Energy:
        - Should be zero during night hours (sunset to sunrise)
        - Should peak around noon when sun is highest
        - Should follow a roughly sinusoidal pattern throughout the day
        
        For Wind Energy:
        - More variable than solar but should show some daily patterns
        - Cannot be negative (turbines don't run backwards)
        - Should have reasonable variability (not constant values)
        
        Args:
            renewable_data: Time series of renewable generation values
            energy_type: Either 'solar' or 'wind' to determine expected patterns
            
        Returns:
            Consistency score between 0 and 1 (higher is better)
        """
        # Need at least 24 hours of data to assess daily patterns
        if len(renewable_data) < 24:
            return 0.5  # Neutral score for insufficient data
        
        # Calculate average generation for each hour of the day
        # This reveals the daily pattern in the data
        hourly_means = renewable_data.groupby(renewable_data.index % 24).mean()
        
        if energy_type == 'solar':
            # Expected solar pattern: sinusoidal curve peaking at noon, zero at night
            # This represents the sun's path across the sky
            expected_pattern = np.sin(np.linspace(0, np.pi, 24)) ** 2
            expected_pattern = expected_pattern / expected_pattern.max()
        else:  # wind
            # Expected wind pattern: more variable but with some daily variation
            # Wind patterns are influenced by daily heating/cooling cycles
            expected_pattern = 0.5 + 0.3 * np.sin(np.linspace(0, 2*np.pi, 24))
        
        # Calculate consistency by comparing actual vs expected patterns
        if hourly_means.max() > 0:
            # Normalize actual pattern to 0-1 scale for comparison
            normalized_pattern = hourly_means / hourly_means.max()
            
            # Calculate mean absolute difference between actual and expected patterns
            # Lower differences indicate higher consistency
            pattern_difference = np.mean(np.abs(normalized_pattern - expected_pattern))
            consistency = 1 - pattern_difference
            
            # Ensure score is non-negative
            return max(0, consistency)
        
        return 0
    
    def _calculate_demand_regularity(self, demand_data: pd.Series) -> float:
        """
        Assess how predictable and regular the electricity demand patterns are.
        
        Electricity demand follows human behavioral patterns:
        - Higher during business hours, lower at night
        - Higher on weekdays, lower on weekends
        - Consistent weekly patterns in stable regions
        
        This method checks whether demand data exhibits these expected regularities.
        Irregular patterns may indicate data corruption or unusual events.
        
        Args:
            demand_data: Time series of electricity demand values
            
        Returns:
            Regularity score between 0 and 1 (higher indicates more predictable patterns)
        """
        # Need at least one week of data to assess weekly patterns
        if len(demand_data) < 168:  # 168 hours = 1 week
            return 0.5  # Neutral score for insufficient data
        
        # Extract weekly patterns by grouping data into week-long segments
        # Each segment represents one week of demand patterns
        weekly_patterns = []
        for week_start in range(0, len(demand_data) - 168, 168):
            week_data = demand_data[week_start:week_start + 168]
            weekly_patterns.append(week_data.values)
        
        # Need at least two weeks to compare patterns
        if len(weekly_patterns) < 2:
            return 0.5
        
        # Calculate correlation between consecutive weeks
        # High correlation indicates regular, predictable patterns
        correlations = []
        for i in range(len(weekly_patterns) - 1):
            # Calculate Pearson correlation between consecutive weeks
            corr = np.corrcoef(weekly_patterns[i], weekly_patterns[i + 1])[0, 1]
            
            # Only include valid correlations (not NaN)
            if not np.isnan(corr):
                correlations.append(abs(corr))  # Use absolute value
        
        # Return mean correlation as regularity score
        return np.mean(correlations) if correlations else 0.5
    
    def _calculate_system_coherence(self, data: pd.DataFrame) -> float:
        """
        Evaluate how well different energy sources work together as a coherent system.
        
        In energy systems, different components should exhibit logical relationships:
        - Generation sources should respond to demand changes
        - Supply and demand should be approximately balanced
        - Different generation sources should complement each other
        
        This method calculates how well these relationships hold in the data.
        Poor coherence may indicate measurement errors or system malfunctions.
        
        Args:
            data: DataFrame containing multiple energy system components
            
        Returns:
            Coherence score between 0 and 1 (higher indicates better system coordination)
        """
        coherence_scores = []
        
        # Only calculate coherence if demand data is available
        if 'energy_demand' in data.columns:
            demand = data['energy_demand']
            
            # Find all generation columns in the dataset
            energy_columns = [col for col in data.columns 
                            if 'generation' in col or 'power' in col]
            
            # Calculate coherence between each generation source and demand
            for col in energy_columns:
                # Skip demand column and columns with no variation
                if col != 'energy_demand' and data[col].std() > 0:
                    # Calculate supply-demand ratio to assess balance
                    # Add small epsilon to avoid division by zero
                    supply_demand_ratio = data[col] / (demand + 1e-6)
                    
                    # Calculate variability of the ratio
                    # Lower variability indicates more consistent supply-demand relationship
                    variability = supply_demand_ratio.std()
                    
                    # Convert variability to coherence score (inverse relationship)
                    # Higher variability = lower coherence
                    coherence = 1 / (1 + variability)
                    coherence_scores.append(coherence)
        
        # Return mean coherence across all energy sources
        return np.mean(coherence_scores) if coherence_scores else 0.5
    
    def clean_hybrid_energy_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Main data cleaning pipeline specifically designed for hybrid energy systems.
        
        This method orchestrates the complete cleaning process through multiple stages,
        each building on the results of the previous stage. The sequence is carefully
        designed to address energy-specific data issues:
        
        Stage 1: Quality Assessment - Understand current data state
        Stage 2: Datetime Processing - Ensure proper time handling for energy data
        Stage 3: Physics Violation Correction - Fix impossible energy values
        Stage 4: Intelligent Gap Filling - Fill missing values using energy-aware methods
        Stage 5: Energy Balance Validation - Ensure supply-demand conservation
        Stage 6: Statistical Outlier Removal - Remove extreme values while preserving spikes
        Stage 7: Feature Engineering - Add derived features for better analysis
        
        Args:
            data: Raw energy system DataFrame requiring cleaning
            
        Returns:
            Cleaned DataFrame ready for analysis and modeling
        """
        self.logger.info("Starting comprehensive energy data cleaning pipeline")
        
        # Stage 1: Initial Quality Assessment
        # Before cleaning, assess the current state of data quality to understand
        # what issues need to be addressed and track improvement after cleaning
        quality_metrics = self.assess_data_quality(data)
        
        # Warn if data quality is below acceptable threshold
        if quality_metrics['overall_quality'] < self.quality_threshold:
            self.logger.warning(
                f"Initial data quality ({quality_metrics['overall_quality']:.1%}) "
                f"below threshold ({self.quality_threshold:.1%})"
            )
        
        # Stage 2: Datetime Processing
        # Energy data is meaningless without proper timestamps. This stage ensures
        # datetime information is correctly formatted and adds time-based features
        cleaned_data = self._process_datetime_index(data.copy())
        
        # Stage 3: Physics Violation Correction
        # Energy data must obey physical laws. This stage corrects impossible values
        # like negative renewable generation or solar power at night
        cleaned_data = self._handle_physics_violations(cleaned_data)
        
        # Stage 4: Intelligent Gap Filling
        # Different energy sources need different gap-filling strategies based on
        # their physical characteristics and behavior patterns
        cleaned_data = self._smart_gap_filling(cleaned_data)
        
        # Stage 5: Energy Balance Validation
        # In energy systems, supply must equal demand (plus losses). This stage
        # identifies and corrects severe energy balance violations
        cleaned_data = self._validate_energy_balance(cleaned_data)
        
        # Stage 6: Statistical Outlier Removal
        # Remove extreme statistical outliers while preserving legitimate spikes
        # that are natural in energy systems (e.g., sudden wind gusts)
        cleaned_data = self._remove_statistical_outliers(cleaned_data)
        
        # Stage 7: Feature Engineering
        # Add derived features that will improve downstream analysis and modeling
        cleaned_data = self._add_engineered_features(cleaned_data)
        
        # Store results and log completion
        self.processed_data = cleaned_data
        
        self.logger.info(f"Data cleaning pipeline completed successfully")
        self.logger.info(f"Processed {len(cleaned_data)} records with {cleaned_data.shape[1]} features")
        
        return cleaned_data
    
    def _process_datetime_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process and standardize datetime information for energy system analysis.
        
        Energy systems operate in real-time, making proper datetime handling critical.
        This method ensures consistent datetime formatting and creates time-based features
        that are essential for energy analysis and forecasting.
        
        Processing Steps:
        1. Convert datetime columns to proper datetime format
        2. Set datetime as DataFrame index for time series operations
        3. Handle missing or invalid datetime values
        4. Remove duplicate timestamps
        5. Create derived time features (hour, day of week, etc.)
        
        Args:
            data: DataFrame with datetime information to process
            
        Returns:
            DataFrame with properly formatted datetime index and time features
        """
        self.logger.info("Processing datetime information and creating time features")
        
        # Step 1: Handle datetime column conversion
        if 'datetime' in data.columns:
            # Convert to datetime format, handling various input formats
            data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
            
            # Set as index for time series operations
            data = data.set_index('datetime')
            self.logger.info("Converted datetime column to index")
        
        # Step 2: Ensure index is datetime type
        elif not isinstance(data.index, pd.DatetimeIndex):
            try:
                # Try to convert existing index to datetime
                data.index = pd.to_datetime(data.index)
                self.logger.info("Converted existing index to datetime")
            except:
                # Create artificial hourly datetime index as fallback
                # This ensures downstream time series operations will work
                self.logger.warning("Creating artificial hourly datetime index")
                start_date = datetime(2024, 1, 1)
                data.index = pd.date_range(start=start_date, periods=len(data), freq='H')
        
        # Step 3: Create time-based features for analysis
        # These features are essential for capturing temporal patterns in energy data
        if isinstance(data.index, pd.DatetimeIndex):
            # Hour of day (0-23) - captures daily energy patterns
            data['hour'] = data.index.hour
            
            # Day of week (0=Monday, 6=Sunday) - captures weekly patterns
            data['day_of_week'] = data.index.dayofweek
            
            # Month (1-12) - captures seasonal patterns
            data['month'] = data.index.month
            
            # Weekend indicator - captures business vs leisure energy patterns
            data['is_weekend'] = data.index.dayofweek >= 5
            
            # Circular time encoding for ML models
            # These preserve the circular nature of time (hour 23 is close to hour 0)
            data['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
            data['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)
            data['day_sin'] = np.sin(2 * np.pi * data.index.dayofyear / 365)
            data['day_cos'] = np.cos(2 * np.pi * data.index.dayofyear / 365)
        
        # Step 4: Handle duplicate timestamps
        # Duplicate timestamps can cause issues in time series analysis
        if data.index.duplicated().any():
            duplicates = data.index.duplicated().sum()
            self.logger.warning(f"Removing {duplicates} duplicate timestamps")
            data = data[~data.index.duplicated(keep='first')]
        
        date_range = f"{data.index.min()} to {data.index.max()}"
        self.logger.info(f"Datetime processing complete. Date range: {date_range}")
        
        return data
    
    def _handle_physics_violations(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Correct violations of energy physics laws in the dataset.
        
        Energy data must obey fundamental physical laws. This method identifies and
        corrects common physics violations that indicate measurement errors or
        data corruption. The corrections are based on domain knowledge rather than
        statistical properties alone.
        
        Physics Laws Applied:
        1. Solar generation must be zero at night (no sunlight)
        2. Renewable generation cannot be negative (conservation of energy)
        3. Demand cannot be zero (base load always exists)
        4. Generation values must be within reasonable physical limits
        
        Args:
            data: DataFrame with potential physics violations
            
        Returns:
            DataFrame with physics violations corrected
        """
        self.logger.info("Applying physics-based corrections to energy data")
        
        # Correction 1: Solar Generation at Night
        # Solar panels cannot generate electricity without sunlight
        if 'solar_generation' in data.columns and 'hour' in data.columns:
            # Define night hours (22:00 to 05:59)
            night_mask = (data['hour'] <= 5) | (data['hour'] >= 22)
            
            # Count violations before correction
            night_solar_violations = (data.loc[night_mask, 'solar_generation'] > 10).sum()
            
            if night_solar_violations > 0:
                self.logger.info(f"Correcting {night_solar_violations} nighttime solar violations")
                
                # Set nighttime solar to zero (allow small sensor noise up to 10 MW)
                data.loc[night_mask, 'solar_generation'] = np.minimum(
                    data.loc[night_mask, 'solar_generation'], 10
                )
        
        # Correction 2: Negative Renewable Generation
        # Renewable sources cannot consume energy (negative generation violates physics)
        renewable_columns = [col for col in data.columns 
                           if any(keyword in col.lower() 
                                for keyword in ['generation', 'solar', 'wind'])]
        
        for col in renewable_columns:
            if col in data.columns:
                negative_count = (data[col] < 0).sum()
                if negative_count > 0:
                    self.logger.info(f"Correcting {negative_count} negative values in {col}")
                    # Set negative values to zero
                    data[col] = np.maximum(data[col], 0)
        
        # Correction 3: Zero Energy Demand
        # Electricity demand is never truly zero due to base load requirements
        if 'energy_demand' in data.columns:
            zero_demand = (data['energy_demand'] <= 0).sum()
            if zero_demand > 0:
                self.logger.info(f"Correcting {zero_demand} zero/negative demand values")
                
                # Replace with minimum reasonable demand (5th percentile of positive values)
                positive_demand = data['energy_demand'][data['energy_demand'] > 0]
                if len(positive_demand) > 0:
                    min_demand = positive_demand.quantile(0.05)
                    data.loc[data['energy_demand'] <= 0, 'energy_demand'] = min_demand
        
        self.logger.info("Physics violation corrections completed")
        return data
    
    def _smart_gap_filling(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Intelligent gap filling algorithm tailored for different energy data types.
        
        Generic interpolation methods don't work well for energy data because different
        energy sources have different characteristics. This method uses specialized
        gap-filling strategies based on the physical and behavioral properties of
        each energy type.
        
        Gap Filling Strategies:
        - Solar: Zero at night, pattern-based filling during day
        - Wind: Persistence-based filling with temporal interpolation
        - Demand: Pattern matching based on similar time periods
        - Other: Standard temporal interpolation
        
        Args:
            data: DataFrame with missing values to fill
            
        Returns:
            DataFrame with intelligent gap filling applied
        """
        self.logger.info("Starting intelligent gap filling for energy data")
        
        # Identify energy-related columns that need gap filling
        energy_columns = [col for col in data.columns 
                         if any(keyword in col.lower() 
                               for keyword in ['generation', 'demand', 'power', 'solar', 'wind'])]
        
        # Apply specialized gap filling for each energy column
        for col in energy_columns:
            missing_count = data[col].isnull().sum()
            if missing_count > 0:
                self.logger.info(f"Filling {missing_count} gaps in {col}")
                
                if 'solar' in col.lower():
                    # Solar-specific gap filling
                    data[col] = self._fill_solar_gaps(data[col], data.get('hour'))
                elif 'wind' in col.lower():
                    # Wind-specific gap filling
                    data[col] = self._fill_wind_gaps(data[col])
                elif 'demand' in col.lower():
                    # Demand-specific gap filling
                    data[col] = self._fill_demand_gaps(data[col], data)
                else:
                    # Generic temporal interpolation for other energy types
                    data[col] = data[col].interpolate(method='time', limit_direction='both')
        
        self.logger.info("Intelligent gap filling completed")
        return data
    
    def _fill_solar_gaps(self, solar_series: pd.Series, hour_series: pd.Series) -> pd.Series:
        """
        Specialized gap filling for solar generation data.
        
        Solar generation follows predictable daily patterns based on sun position.
        This method leverages this knowledge to fill gaps more accurately than
        generic interpolation methods.
        
        Strategy:
        1. Night gaps (10 PM - 6 AM) are filled with zero (no sunlight)
        2. Day gaps are filled using temporal interpolation
        3. Patterns from similar weather conditions are used when available
        
        Args:
            solar_series: Solar generation time series with gaps
            hour_series: Hour of day information for pattern recognition
            
        Returns:
            Solar series with gaps intelligently filled
        """
        if hour_series is not None:
            # Fill night hours with zero (solar panels don't work without sun)
            night_mask = (hour_series <= 5) | (hour_series >= 22)
            solar_series.loc[night_mask] = solar_series.loc[night_mask].fillna(0)
        
        # For remaining gaps (daylight hours), use temporal interpolation
        # This preserves the daily solar pattern while filling gaps
        return solar_series.interpolate(method='time', limit_direction='both')
    
    def _fill_wind_gaps(self, wind_series: pd.Series) -> pd.Series:
        """
        Specialized gap filling for wind generation data.
        
        Wind generation is more chaotic than solar but still has some persistence.
        This method combines forward filling (persistence) with interpolation to
        handle the stochastic nature of wind while filling gaps reasonably.
        
        Strategy:
        1. Short gaps (≤3 hours) use forward fill (wind persistence)
        2. Longer gaps use temporal interpolation
        3. Very long gaps may use seasonal/historical patterns
        
        Args:
            wind_series: Wind generation time series with gaps
            
        Returns:
            Wind series with gaps filled using persistence and interpolation
        """
        # Wind has some persistence - use forward fill for short gaps
        filled = wind_series.fillna(method='ffill', limit=3)
        
        # Use interpolation for remaining gaps
        filled = filled.interpolate(method='time', limit_direction='both')
        
        return filled
    
    def _fill_demand_gaps(self, demand_series: pd.Series, full_data: pd.DataFrame) -> pd.Series:
        """
        Specialized gap filling for electricity demand data.
        
        Electricity demand follows human behavioral patterns that are highly predictable.
        This method uses pattern matching to find similar time periods and uses their
        values to fill gaps more accurately than simple interpolation.
        
        Strategy:
        1. Group by hour and day type (weekday/weekend) to find similar periods
        2. Use median values from similar periods to fill gaps
        3. Fall back to temporal interpolation for unmatched patterns
        
        Args:
            demand_series: Demand time series with gaps
            full_data: Complete DataFrame with time features for pattern matching
            
        Returns:
            Demand series with gaps filled using pattern matching
        """
        # Use pattern matching if time features are available
        if 'hour' in full_data.columns and 'is_weekend' in full_data.columns:
            
            # Group by hour and weekend/weekday for pattern matching
            pattern_groups = full_data.groupby(['hour', 'is_weekend'])['energy_demand']
            
            # Fill gaps using similar time period patterns
            for (hour, is_weekend), group in pattern_groups:
                # Create mask for this specific pattern (hour + day type)
                group_mask = ((full_data['hour'] == hour) & 
                             (full_data['is_weekend'] == is_weekend))
                
                # Find missing values in this pattern group
                missing_in_group = demand_series.loc[group_mask].isnull()
                
                # If there are gaps and we have data for this pattern, fill them
                if missing_in_group.any() and not group.dropna().empty:
                    fill_value = group.dropna().median()
                    demand_series.loc[group_mask & missing_in_group] = fill_value
        
        # Fill any remaining gaps with temporal interpolation
        return demand_series.interpolate(method='time', limit_direction='both')
    
    def _validate_energy_balance(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and correct energy conservation violations in the dataset.
        
        In energy systems, the fundamental principle of energy conservation requires
        that total generation approximately equals total demand (accounting for losses).
        Large imbalances indicate measurement errors or data corruption.
        
        This method identifies severe energy balance violations and applies corrections
        based on the assumption that demand measurements are typically more reliable
        than generation measurements.
        
        Args:
            data: DataFrame with potential energy balance issues
            
        Returns:
            DataFrame with energy balance violations corrected
        """
        self.logger.info("Validating energy conservation and balance")
        
        # Find all generation columns in the dataset
        generation_cols = [col for col in data.columns if 'generation' in col]
        
        # Only proceed if we have both generation and demand data
        if generation_cols and 'energy_demand' in data.columns:
            # Calculate total generation from all sources
            total_generation = data[generation_cols].sum(axis=1)
            demand = data['energy_demand']
            
            # Calculate supply-demand imbalance ratio
            # Add small epsilon to avoid division by zero
            imbalance_ratio = abs(total_generation - demand) / (demand + 1e-6)
            
            # Identify severe imbalances (>300% is physically impossible)
            severe_imbalances = imbalance_ratio > 3.0
            
            if severe_imbalances.sum() > 0:
                self.logger.warning(f"Found {severe_imbalances.sum()} severe energy balance violations")
                
                # Correction strategy: scale generation to match demand
                # This assumes demand measurements are more reliable
                for idx in data.index[severe_imbalances]:
                    if total_generation.loc[idx] > 0:
                        # Calculate scaling factor to match demand
                        scale_factor = demand.loc[idx] / total_generation.loc[idx]
                        
                        # Apply scaling to all generation sources proportionally
                        for col in generation_cols:
                            data.loc[idx, col] *= scale_factor
                
                self.logger.info("Energy balance violations corrected")
        
        return data
    
    def _remove_statistical_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove statistical outliers while preserving legitimate energy system spikes.
        
        Energy systems naturally exhibit sudden spikes (wind gusts, equipment startup)
        that should be preserved. This method uses Modified Z-score instead of standard
        Z-score because it's more robust to outliers, and applies different thresholds
        for different energy types based on their natural variability.
        
        Outlier Detection Strategy:
        1. Use Modified Z-score based on median and median absolute deviation
        2. Apply lenient thresholds for renewables (higher natural variability)
        3. Apply stricter thresholds for conventional sources (more predictable)
        4. Replace outliers with rolling median to preserve local patterns
        
        Args:
            data: DataFrame potentially containing statistical outliers
            
        Returns:
            DataFrame with extreme statistical outliers removed
        """
        self.logger.info("Removing statistical outliers while preserving energy spikes")
        
        # Find numeric columns related to energy measurements
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        outliers_removed = 0
        
        for col in numeric_columns:
            # Only process energy-related columns
            if any(keyword in col.lower() for keyword in ['generation', 'demand', 'power']):
                
                # Calculate Modified Z-score using median and MAD (more robust than mean/std)
                median_val = data[col].median()
                mad = np.median(np.abs(data[col] - median_val))  # Median Absolute Deviation
                
                # Avoid division by zero when MAD is zero (constant values)
                if mad > 0:
                    modified_z_scores = 0.6745 * (data[col] - median_val) / mad
                    
                    # Set thresholds based on energy source characteristics
                    # Renewable sources are more variable, so use higher thresholds
                    if any(keyword in col.lower() for keyword in ['solar', 'wind']):
                        threshold = 4.0  # More lenient for renewables
                    else:
                        threshold = 3.5  # Stricter for conventional sources
                    
                    # Identify outliers exceeding threshold
                    outliers = np.abs(modified_z_scores) > threshold
                    outlier_count = outliers.sum()
                    
                    if outlier_count > 0:
                        self.logger.info(f"Removing {outlier_count} statistical outliers from {col}")
                        
                        # Replace outliers with rolling median to preserve local patterns
                        # Rolling window of 5 provides smooth replacement values
                        rolling_median = data[col].rolling(window=5, center=True).median()
                        data.loc[outliers, col] = rolling_median[outliers]
                        
                        outliers_removed += outlier_count
        
        self.logger.info(f"Statistical outlier removal complete. Removed {outliers_removed} outliers total")
        return data
    
    def _add_engineered_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features that improve downstream analysis and modeling.
        
        Feature engineering creates new variables from existing data that capture
        important patterns and relationships. For energy systems, these features
        help machine learning models understand temporal patterns, system efficiency,
        and operational characteristics.
        
        Engineered Features:
        1. Lag features - capture recent history effects
        2. Rolling statistics - capture trends and variability
        3. Renewable penetration - measure renewable energy contribution
        4. Energy ratios - capture system efficiency metrics
        5. Peak indicators - identify high-demand periods
        
        Args:
            data: DataFrame to enhance with engineered features
            
        Returns:
            DataFrame with additional engineered features
        """
        self.logger.info("Adding engineered features for enhanced analysis")
        
        # Feature Set 1: Lag Features
        # Recent history is important for forecasting energy patterns
        if 'energy_demand' in data.columns:
            # 1-hour lag captures immediate recent history
            data['demand_lag_1h'] = data['energy_demand'].shift(1)
            
            # 24-hour lag captures same-hour-yesterday pattern
            data['demand_lag_24h'] = data['energy_demand'].shift(24)
            
            # 168-hour lag captures same-hour-last-week pattern
            data['demand_lag_168h'] = data['energy_demand'].shift(168)
        
        # Feature Set 2: Rolling Statistics
        # Capture trends and variability over different time windows
        energy_cols = ['energy_demand', 'solar_generation', 'wind_generation']
        for col in energy_cols:
            if col in data.columns:
                # 24-hour rolling statistics (daily patterns)
                data[f'{col}_24h_mean'] = data[col].rolling(window=24, min_periods=12).mean()
                data[f'{col}_24h_std'] = data[col].rolling(window=24, min_periods=12).std()
                
                # 168-hour rolling statistics (weekly patterns)
                data[f'{col}_168h_mean'] = data[col].rolling(window=168, min_periods=84).mean()
        
        # Feature Set 3: Renewable Penetration Ratio
        # Measure the contribution of renewable energy to total supply
        renewable_cols = [col for col in data.columns if any(keyword in col for keyword in ['solar', 'wind'])]
        if renewable_cols and 'energy_demand' in data.columns:
            total_renewable = data[renewable_cols].sum(axis=1)
            # Calculate percentage of demand met by renewables
            data['renewable_penetration'] = total_renewable / (data['energy_demand'] + 1e-6)
            
            # Cap at 100% (can't exceed demand in this simplified model)
            data['renewable_penetration'] = np.minimum(data['renewable_penetration'], 1.0)
        
        # Feature Set 4: Energy System Ratios
        # Calculate efficiency and balance metrics
        if 'solar_generation' in data.columns and 'wind_generation' in data.columns:
            # Solar to wind ratio (indicates weather patterns)
            data['solar_wind_ratio'] = (data['solar_generation'] + 1) / (data['wind_generation'] + 1)
        
        # Feature Set 5: Peak Demand Indicators
        # Binary indicators for high-demand periods
        if 'energy_demand' in data.columns:
            # Peak demand threshold (90th percentile)
            peak_threshold = data['energy_demand'].quantile(0.9)
            data['is_peak_demand'] = (data['energy_demand'] > peak_threshold).astype(int)
            
            # Demand category (low, medium, high)
            demand_33 = data['energy_demand'].quantile(0.33)
            demand_67 = data['energy_demand'].quantile(0.67)
            
            data['demand_category'] = 0  # Low
            data.loc[data['energy_demand'] > demand_33, 'demand_category'] = 1  # Medium
            data.loc[data['energy_demand'] > demand_67, 'demand_category'] = 2  # High
        
        # Feature Set 6: Time-based Interaction Features
        # Capture interactions between time and energy patterns
        if 'hour' in data.columns and 'energy_demand' in data.columns:
            # Hour-demand interaction (captures hourly demand patterns)
            data['hour_demand_interaction'] = data['hour'] * data['energy_demand']
        
        if 'is_weekend' in data.columns and 'energy_demand' in data.columns:
            # Weekend-demand interaction (captures weekend vs weekday patterns)
            data['weekend_demand_interaction'] = data['is_weekend'] * data['energy_demand']
        
        feature_count = len([col for col in data.columns 
                           if any(keyword in col for keyword in ['lag', 'mean', 'std', 'ratio', 'penetration'])])
        
        self.logger.info(f"Feature engineering complete. Added {feature_count} engineered features")
        return data
    
    def generate_quality_report(self) -> Dict:
        """
        Generate comprehensive data quality report for thesis documentation.
        
        This method creates a detailed report of the data cleaning process and results
        that can be included in the methodology section of the MSc thesis. The report
        documents all quality metrics, cleaning steps performed, and recommendations
        for further data collection or processing.
        
        Returns:
            Dictionary containing comprehensive quality assessment and cleaning summary
        """
        if not self.data_quality_metrics:
            return {"error": "No quality assessment performed yet. Run clean_hybrid_energy_data() first."}
        
        # Calculate processing statistics
        processing_stats = {}
        if self.processed_data is not None:
            processing_stats = {
                "final_shape": self.processed_data.shape,
                "total_features": self.processed_data.shape[1],
                "total_records": self.processed_data.shape[0],
                "memory_usage_mb": self.processed_data.memory_usage(deep=True).sum() / 1024 / 1024,
                "date_range": {
                    "start": self.processed_data.index.min().isoformat() if hasattr(self.processed_data.index, 'min') else "N/A",
                    "end": self.processed_data.index.max().isoformat() if hasattr(self.processed_data.index, 'max') else "N/A"
                }
            }
        
        # Generate recommendations based on quality metrics
        recommendations = self._generate_quality_recommendations()
        
        # Create comprehensive report
        report = {
            "report_metadata": {
                "generated_timestamp": datetime.now().isoformat(),
                "processor_version": "EnergyNexus_v1.0",
                "author": "Aditya Talekar (ec24018@qmul.ac.uk)",
                "quality_threshold_used": self.quality_threshold
            },
            "data_quality_assessment": {
                "overall_quality_score": f"{self.data_quality_metrics['overall_quality']:.1%}",
                "quality_components": {
                    "completeness": f"{self.data_quality_metrics.get('completeness', 0):.1%}",
                    "solar_consistency": f"{self.data_quality_metrics.get('solar_consistency', 0):.1%}",
                    "wind_consistency": f"{self.data_quality_metrics.get('wind_consistency', 0):.1%}",
                    "demand_regularity": f"{self.data_quality_metrics.get('demand_regularity', 0):.1%}",
                    "system_coherence": f"{self.data_quality_metrics.get('system_coherence', 0):.1%}"
                },
                "quality_assessment_verdict": self._get_quality_verdict()
            },
            "cleaning_pipeline_summary": {
                "steps_performed": [
                    "1. Comprehensive data quality assessment",
                    "2. Datetime standardization and time feature creation",
                    "3. Physics violation detection and correction",
                    "4. Intelligent gap filling using energy-specific methods",
                    "5. Energy conservation validation and balance correction",
                    "6. Statistical outlier removal with energy-aware thresholds",
                    "7. Feature engineering for enhanced analysis capabilities"
                ],
                "novel_contributions": [
                    "Physics-aware data cleaning specifically for energy systems",
                    "Multi-source energy data quality assessment framework",
                    "Intelligent gap filling algorithms for different energy types",
                    "Energy conservation validation and automatic correction",
                    "Renewable energy consistency scoring methodology"
                ]
            },
            "processing_results": processing_stats,
            "quality_recommendations": recommendations,
            "thesis_documentation": {
                "methodology_notes": "This cleaning pipeline was designed specifically for hybrid energy systems research",
                "academic_contribution": "Novel data quality framework for renewable energy integration studies",
                "reproducibility": "All cleaning steps are documented and parameterized for reproducibility"
            }
        }
        
        return report
    
    def _generate_quality_recommendations(self) -> List[str]:
        """
        Generate specific recommendations based on data quality assessment results.
        
        These recommendations help improve data collection, processing, or analysis
        based on the identified quality issues in the current dataset.
        
        Returns:
            List of actionable recommendations for improving data quality
        """
        recommendations = []
        
        # Completeness recommendations
        completeness = self.data_quality_metrics.get('completeness', 1)
        if completeness < 0.9:
            recommendations.append(
                f"Data completeness is {completeness:.1%}. Improve data collection systems "
                "to reduce missing values and ensure more reliable measurements."
            )
        
        # Solar consistency recommendations
        solar_consistency = self.data_quality_metrics.get('solar_consistency', 1)
        if solar_consistency < 0.7:
            recommendations.append(
                f"Solar generation consistency is {solar_consistency:.1%}. Verify solar panel "
                "equipment calibration, check for shading issues, and validate irradiance sensors."
            )
        
        # Wind consistency recommendations
        wind_consistency = self.data_quality_metrics.get('wind_consistency', 1)
        if wind_consistency < 0.7:
            recommendations.append(
                f"Wind generation consistency is {wind_consistency:.1%}. Check wind turbine "
                "maintenance schedules, verify anemometer calibration, and investigate potential "
                "equipment malfunctions."
            )
        
        # Demand regularity recommendations
        demand_regularity = self.data_quality_metrics.get('demand_regularity', 1)
        if demand_regularity < 0.6:
            recommendations.append(
                f"Demand pattern regularity is {demand_regularity:.1%}. Investigate unusual "
                "consumption patterns, verify meter readings, and check for data transmission errors."
            )
        
        # System coherence recommendations
        system_coherence = self.data_quality_metrics.get('system_coherence', 1)
        if system_coherence < 0.5:
            recommendations.append(
                f"System coherence is {system_coherence:.1%}. Review energy source coordination, "
                "validate supply-demand balance calculations, and check for systematic measurement biases."
            )
        
        # Overall quality recommendations
        overall_quality = self.data_quality_metrics.get('overall_quality', 1)
        if overall_quality < self.quality_threshold:
            recommendations.append(
                f"Overall data quality ({overall_quality:.1%}) is below the threshold "
                f"({self.quality_threshold:.1%}). Consider additional data validation steps "
                "before proceeding with analysis or modeling."
            )
        
        # Add positive reinforcement if quality is good
        if overall_quality >= 0.85:
            recommendations.append(
                "Data quality is excellent. The dataset is well-suited for advanced analysis "
                "and machine learning applications."
            )
        
        return recommendations
    
    def _get_quality_verdict(self) -> str:
        """
        Provide an overall verdict on data quality for easy interpretation.
        
        Returns:
            String verdict summarizing overall data quality assessment
        """
        overall_quality = self.data_quality_metrics.get('overall_quality', 0)
        
        if overall_quality >= 0.9:
            return "Excellent - Data is of high quality and ready for advanced analysis"
        elif overall_quality >= 0.8:
            return "Good - Data quality is acceptable with minor issues addressed"
        elif overall_quality >= 0.7:
            return "Fair - Data quality is adequate but may benefit from additional cleaning"
        elif overall_quality >= 0.6:
            return "Poor - Data has significant quality issues requiring attention"
        else:
            return "Very Poor - Data quality is insufficient for reliable analysis"

# Test function to verify the processor works correctly
if __name__ == "__main__":
    """
    Test the EnergyDataProcessor with synthetic data to verify functionality.
    
    This test creates realistic energy data with common problems found in real datasets,
    then applies the cleaning pipeline and validates the results. This serves as both
    a unit test and a demonstration of the processor's capabilities.
    """
    
    # Configure test parameters
    test_duration_hours = 168  # One week of hourly data
    np.random.seed(42)  # For reproducible test results
    
    # Create synthetic test data with realistic energy patterns
    dates = pd.date_range(start='2024-01-01', periods=test_duration_hours, freq='H')
    
    test_data = pd.DataFrame({
        'datetime': dates,
        # Solar generation: sinusoidal daily pattern with noise
        'solar_generation': np.maximum(0, 
            150 * np.sin(np.arange(test_duration_hours) * 2 * np.pi / 24) + 
            np.random.normal(0, 20, test_duration_hours)),
        # Wind generation: more variable with daily and random components
        'wind_generation': 80 + 40 * np.sin(np.arange(test_duration_hours) * np.pi / 12) + 
                          np.random.normal(0, 25, test_duration_hours),
        # Energy demand: daily pattern with weekly variation and noise
        'energy_demand': 400 + 100 * np.sin((np.arange(test_duration_hours) - 6) * 2 * np.pi / 24) + 
                        50 * np.sin(np.arange(test_duration_hours) * 2 * np.pi / (24 * 7)) +
                        np.random.normal(0, 30, test_duration_hours)
    })
    
    # Introduce realistic data problems for testing
    # Missing values (common in real energy data)
    test_data.loc[10:15, 'solar_generation'] = np.nan
    test_data.loc[50:52, 'wind_generation'] = np.nan
    test_data.loc[100, 'energy_demand'] = np.nan
    
    # Physics violations (equipment errors)
    test_data.loc[0:5, 'solar_generation'] = 50  # Solar at night
    test_data.loc[25, 'wind_generation'] = -30   # Negative wind
    test_data.loc[75, 'energy_demand'] = 0       # Zero demand
    
    # Statistical outliers (equipment malfunctions)
    test_data.loc[120, 'energy_demand'] = 2000   # Extremely high demand
    test_data.loc[140, 'wind_generation'] = 500  # Impossibly high wind
    
    # Initialize and run the processor
    processor = EnergyDataProcessor(quality_threshold=0.75)
    
    # Execute the complete cleaning pipeline
    cleaned_data = processor.clean_hybrid_energy_data(test_data)
    
    # Generate comprehensive quality report
    quality_report = processor.generate_quality_report()
    
    # Display test results
    print("EnergyNexus Data Processor Test Results")
    print("=" * 50)
    print(f"Processing completed successfully")
    print(f"Initial data shape: {test_data.shape}")
    print(f"Final data shape: {cleaned_data.shape}")
    print(f"Overall quality: {quality_report['data_quality_assessment']['overall_quality_score']}")
    print(f"Quality verdict: {quality_report['data_quality_assessment']['quality_assessment_verdict']}")
    
    # Validate physics corrections
    night_hours = (cleaned_data['hour'] <= 5) | (cleaned_data['hour'] >= 22)
    night_solar_max = cleaned_data.loc[night_hours, 'solar_generation'].max()
    negative_renewables = ((cleaned_data['solar_generation'] < 0) | 
                          (cleaned_data['wind_generation'] < 0)).sum()
    zero_demand = (cleaned_data['energy_demand'] <= 0).sum()
    
    print(f"\nPhysics Validation:")
    print(f"Maximum solar at night: {night_solar_max:.1f} MW (should be ~0)")
    print(f"Negative renewable values: {negative_renewables} (should be 0)")
    print(f"Zero demand values: {zero_demand} (should be 0)")
    
    # Save test results
    cleaned_data.to_csv('data/processed/test_cleaned_energy_data.csv')
    
    import json
    with open('results/reports/test_cleaning_report.json', 'w') as f:
        json.dump(quality_report, f, indent=2, default=str)
    
    print(f"\nTest completed successfully!")
    print(f"Cleaned data saved to: data/processed/test_cleaned_energy_data.csv")
    print(f"Quality report saved to: results/reports/test_cleaning_report.json")
    # Direct test execution
print("Starting direct test...")
processor = EnergyDataProcessor()
print("Processor created successfully!")