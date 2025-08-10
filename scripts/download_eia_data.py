"""
Energy Information Administration (EIA) Data Collection Script
MSc Data Science and AI Research Project

This comprehensive script implements a systematic approach to collecting essential energy data
from the EIA API for electricity demand forecasting and renewable energy analysis research.
The script focuses on acquiring high-quality, structured datasets suitable for machine learning
applications, particularly LSTM-based forecasting models.

Research Context:
- Primary focus: Electricity demand forecasting using LSTM neural networks
- Secondary analysis: Renewable energy penetration and generation mix optimization
- Data scope: One year of historical demand data with complementary generation data
- Target application: Energy scheduling optimization algorithms

Author: Aditya Talekar (ec24018@qmul.ac.uk)
Institution: Queen Mary University of London
Program: MSc Data Science and AI - 2024/25
Date: 2025
"""
import sys
import os
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

# Configure module path for custom data pipeline components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_pipeline.collectors.eia_collector import EIADataCollector

print('test')
# Configure logging for comprehensive tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/eia_data_collection.log'),
        logging.StreamHandler()
    ]
)

def initialize_data_collector():
    """
    Initialize the EIA data collector with proper error handling and configuration validation.
    
    This function performs the following initialization steps:
    1. Loads API credentials from configuration files
    2. Validates API key format and accessibility
    3. Establishes connection parameters for EIA API endpoints
    4. Sets up data quality validation parameters
    
    Returns:
        EIADataCollector: Configured collector instance
        
    Raises:
        ConnectionError: If API key is invalid or API is unreachable
        ConfigurationError: If configuration files are missing or malformed
    """
    try:
        collector = EIADataCollector()
        logging.info("EIA Data Collector successfully initialized")
        logging.info("API key loaded and validated from configuration")
        return collector
    except Exception as initialization_error:
        logging.error(f"Failed to initialize EIA collector: {initialization_error}")
        logging.error("Please verify API key configuration in config/api_keys.yaml")
        raise initialization_error

def download_electricity_demand_data(collector, start_date, end_date):
    """
    Download comprehensive electricity demand data for the specified time period.
    
    This function implements the core data collection for electricity demand, which serves
    as the primary target variable for LSTM forecasting models. The demand data includes
    hourly consumption patterns across the US48 interconnected grid system.
    
    Data Collection Strategy:
    - Region: US48 (lower 48 states interconnected grid)
    - Frequency: Hourly resolution for high-granularity analysis
    - Time span: One full year to capture seasonal patterns
    - Data quality: Automatic validation of completeness and consistency
    
    Args:
        collector (EIADataCollector): Initialized data collector instance
        start_date (datetime): Beginning of data collection period
        end_date (datetime): End of data collection period
        
    Returns:
        tuple: (success_status, data_summary) where success_status is boolean
               and data_summary contains collection metrics
    """
    logging.info("Beginning electricity demand data collection")
    logging.info(f"Collection period: {start_date.date()} to {end_date.date()}")
    
    try:
        # Execute API call for demand data with specified parameters
        demand_data = collector.get_electricity_demand(
            region='US48',
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            frequency='hourly'
        )
        
        # Validate data quality and completeness
        if demand_data is not None and len(demand_data) > 0:
            # Calculate data quality metrics
            total_records = len(demand_data)
            min_demand = demand_data['electricity_demand_mw'].min()
            max_demand = demand_data['electricity_demand_mw'].max()
            mean_demand = demand_data['electricity_demand_mw'].mean()
            
            # Identify data gaps or anomalies
            missing_values = demand_data['electricity_demand_mw'].isnull().sum()
            data_completeness = ((total_records - missing_values) / total_records) * 100
            
            # Save validated data with comprehensive metadata
            save_success = collector.save_collected_data(demand_data, "us_electricity_demand_2024")
            
            if save_success:
                logging.info(f"Successfully collected {total_records:,} demand records")
                logging.info(f"Demand statistics: Min={min_demand:.0f}MW, Max={max_demand:.0f}MW, Mean={mean_demand:.0f}MW")
                logging.info(f"Data completeness: {data_completeness:.1f}%")
                logging.info("Demand data saved to: data/raw/us_electricity_demand_2024.csv")
                
                return True, {
                    'records': total_records,
                    'min_demand': min_demand,
                    'max_demand': max_demand,
                    'mean_demand': mean_demand,
                    'completeness': data_completeness
                }
            else:
                logging.error("Failed to save demand data to storage")
                return False, None
        else:
            logging.error("No demand data returned from EIA API")
            return False, None
            
    except Exception as demand_error:
        logging.error(f"Error during demand data collection: {demand_error}")
        return False, None

def download_generation_mix_data(collector, start_date, end_date):
    """
    Collect electricity generation mix data by fuel type and technology.
    
    This function downloads generation data segmented by fuel types (coal, natural gas,
    nuclear, renewable sources, etc.) to support renewable energy penetration analysis
    and generation scheduling optimization research.
    
    Analysis Applications:
    - Renewable energy integration patterns
    - Fuel mix optimization for cost and emissions
    - Generation capacity utilization analysis
    - Correlation analysis between demand and generation sources
    
    Args:
        collector (EIADataCollector): Initialized data collector instance
        start_date (datetime): Beginning of collection period
        end_date (datetime): End of collection period
        
    Returns:
        tuple: (success_status, generation_summary) with collection results
    """
    logging.info("Initiating generation mix data collection")
    logging.info("Focus: Fuel type distribution and renewable penetration analysis")
    
    try:
        # Collect generation mix data with hourly granularity
        generation_data = collector.get_generation_mix(
            region='US48',
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            frequency='hourly'
        )
        
        if generation_data is not None and len(generation_data) > 0:
            # Analyze fuel type composition
            fuel_types = generation_data['fuel_type'].unique()
            total_generation = generation_data['generation_mw'].sum()
            
            # Calculate fuel mix percentages for validation
            fuel_mix_analysis = generation_data.groupby('fuel_type')['generation_mw'].sum()
            fuel_percentages = (fuel_mix_analysis / total_generation * 100).round(2)
            
            # Save generation data with metadata
            save_success = collector.save_collected_data(generation_data, "us_generation_mix_3months")
            
            if save_success:
                logging.info(f"Successfully collected {len(generation_data):,} generation records")
                logging.info(f"Fuel types identified: {', '.join(fuel_types)}")
                logging.info(f"Total generation captured: {total_generation:,.0f} MWh")
                logging.info("Generation mix analysis:")
                for fuel, percentage in fuel_percentages.items():
                    logging.info(f"  {fuel}: {percentage:.1f}%")
                logging.info("Generation data saved to: data/raw/us_generation_mix_3months.csv")
                
                return True, {
                    'records': len(generation_data),
                    'fuel_types': list(fuel_types),
                    'total_generation': total_generation,
                    'fuel_mix': fuel_percentages.to_dict()
                }
            else:
                logging.error("Failed to save generation mix data")
                return False, None
        else:
            logging.error("No generation mix data available from API")
            return False, None
            
    except Exception as generation_error:
        logging.error(f"Error during generation mix collection: {generation_error}")
        return False, None

def analyze_renewable_generation(collector, start_date, end_date):
    """
    Perform specialized analysis of renewable energy generation patterns.
    
    This function focuses specifically on renewable energy sources (solar, wind,
    hydroelectric, geothermal, biomass) to support research into renewable energy
    integration challenges and opportunities in electricity scheduling optimization.
    
    Renewable Analysis Components:
    - Renewable penetration rates over time
    - Intermittency patterns for wind and solar
    - Seasonal variations in renewable output
    - Correlation with demand patterns
    - Grid stability implications
    
    Args:
        collector (EIADataCollector): Initialized data collector instance
        start_date (datetime): Analysis period start
        end_date (datetime): Analysis period end
        
    Returns:
        tuple: (success_status, renewable_analysis_summary)
    """
    logging.info("Commencing renewable energy generation analysis")
    logging.info("Objective: Quantify renewable penetration and variability patterns")
    
    try:
        # Execute renewable-specific data collection
        renewable_data = collector.get_renewable_generation(
            region='US48',
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if renewable_data is not None and len(renewable_data) > 0:
            # Categorize renewable sources
            renewable_categories = renewable_data['renewable_category'].unique()
            total_renewable_generation = renewable_data['generation_mw'].sum()
            
            # Calculate renewable penetration statistics
            renewable_analysis = {}
            if 'renewable_penetration' in renewable_data.columns:
                renewable_analysis['average_penetration'] = renewable_data['renewable_penetration'].mean()
                renewable_analysis['max_penetration'] = renewable_data['renewable_penetration'].max()
                renewable_analysis['min_penetration'] = renewable_data['renewable_penetration'].min()
                renewable_analysis['penetration_std'] = renewable_data['renewable_penetration'].std()
            
            # Analyze renewable generation by category
            category_analysis = renewable_data.groupby('renewable_category')['generation_mw'].agg([
                'sum', 'mean', 'std', 'min', 'max'
            ]).round(2)
            
            # Save comprehensive renewable analysis
            save_success = collector.save_collected_data(renewable_data, "us_renewable_generation")
            
            if save_success:
                logging.info(f"Renewable analysis complete: {len(renewable_data):,} records processed")
                logging.info(f"Renewable categories: {', '.join(renewable_categories)}")
                logging.info(f"Total renewable generation: {total_renewable_generation:,.0f} MWh")
                
                if renewable_analysis:
                    logging.info(f"Average renewable penetration: {renewable_analysis['average_penetration']:.1%}")
                    logging.info(f"Renewable penetration range: {renewable_analysis['min_penetration']:.1%} - {renewable_analysis['max_penetration']:.1%}")
                
                logging.info("Renewable generation analysis saved to: data/raw/us_renewable_generation.csv")
                
                return True, {
                    'records': len(renewable_data),
                    'categories': list(renewable_categories),
                    'total_generation': total_renewable_generation,
                    'penetration_stats': renewable_analysis,
                    'category_analysis': category_analysis.to_dict()
                }
            else:
                logging.error("Failed to save renewable generation analysis")
                return False, None
        else:
            logging.error("No renewable generation data available")
            return False, None
            
    except Exception as renewable_error:
        logging.error(f"Error during renewable analysis: {renewable_error}")
        return False, None

def generate_synthetic_validation_data(collector):
    """
    Create synthetic electricity demand data for algorithm validation and testing.
    
    Synthetic data generation serves multiple research purposes:
    1. Algorithm validation with known patterns
    2. Stress testing of forecasting models
    3. Scenario analysis for extreme conditions
    4. Baseline comparison for model performance evaluation
    
    Synthetic Data Characteristics:
    - Realistic demand patterns with seasonal variation
    - Configurable noise levels and trend components
    - Daily and weekly cyclical patterns
    - Extreme event simulation capabilities
    
    Args:
        collector (EIADataCollector): Initialized data collector instance
        
    Returns:
        tuple: (success_status, synthetic_data_summary)
    """
    logging.info("Generating synthetic validation data for algorithm testing")
    logging.info("Purpose: Model validation and performance benchmarking")
    
    try:
        # Generate full year of synthetic hourly data
        synthetic_data = collector.create_synthetic_demand_data(
            hours=8760,  # Complete year (365 * 24 hours)
            base_demand=400,  # Base demand in MW (realistic for regional analysis)
            region='SYNTHETIC_US'
        )
        
        if synthetic_data is not None and len(synthetic_data) > 0:
            # Calculate synthetic data characteristics
            synthetic_stats = {
                'total_hours': len(synthetic_data),
                'min_demand': synthetic_data['electricity_demand_mw'].min(),
                'max_demand': synthetic_data['electricity_demand_mw'].max(),
                'mean_demand': synthetic_data['electricity_demand_mw'].mean(),
                'std_demand': synthetic_data['electricity_demand_mw'].std()
            }
            
            # Validate synthetic data quality
            demand_range = synthetic_stats['max_demand'] - synthetic_stats['min_demand']
            coefficient_of_variation = synthetic_stats['std_demand'] / synthetic_stats['mean_demand']
            
            # Save synthetic dataset for testing purposes
            save_success = collector.save_collected_data(synthetic_data, "synthetic_demand_full_year")
            
            if save_success:
                logging.info(f"Generated {len(synthetic_data):,} hours of synthetic demand data")
                logging.info(f"Synthetic demand range: {synthetic_stats['min_demand']:.0f} - {synthetic_stats['max_demand']:.0f} MW")
                logging.info(f"Demand variability (CV): {coefficient_of_variation:.3f}")
                logging.info(f"Peak-to-base ratio: {synthetic_stats['max_demand']/synthetic_stats['min_demand']:.2f}")
                logging.info("Synthetic data saved to: data/raw/synthetic_demand_full_year.csv")
                
                return True, synthetic_stats
            else:
                logging.error("Failed to save synthetic validation data")
                return False, None
        else:
            logging.error("Synthetic data generation failed")
            return False, None
            
    except Exception as synthetic_error:
        logging.error(f"Error during synthetic data generation: {synthetic_error}")
        return False, None

def generate_comprehensive_collection_report(collector, download_results):
    """
    Generate detailed documentation of the data collection process and results.
    
    The collection report provides comprehensive metadata about the data acquisition
    process, including data quality metrics, collection timestamps, API response
    statistics, and recommendations for subsequent analysis steps.
    
    Report Components:
    - Collection summary and success metrics
    - Data quality assessment and validation results
    - File inventory with sizes and locations
    - Recommended preprocessing steps
    - Known data limitations and considerations
    
    Args:
        collector (EIADataCollector): Data collector instance
        download_results (dict): Summary of all download operations
        
    Returns:
        bool: Success status of report generation
    """
    logging.info("Generating comprehensive data collection report")
    
    try:
        # Generate base collection report from collector
        collection_report = collector.generate_collection_report()
        
        # Enhance report with session-specific information
        session_summary = {
            'collection_timestamp': datetime.now().isoformat(),
            'successful_downloads': download_results['successful_count'],
            'total_attempted_downloads': download_results['total_attempted'],
            'overall_success_rate': f"{(download_results['successful_count']/download_results['total_attempted']*100):.0f}%",
            'data_scope_description': 'Comprehensive energy data: 1 year demand + 3 months generation mix + renewable analysis + synthetic validation data',
            'collection_strategy': 'Focused approach prioritizing LSTM training data and renewable energy analysis',
            'data_quality_notes': download_results.get('quality_notes', []),
            'recommended_next_steps': [
                'Execute data cleaning and validation pipeline',
                'Perform exploratory data analysis',
                'Implement feature engineering for LSTM models',
                'Conduct baseline forecasting model evaluation'
            ]
        }
        
        # Combine collector report with session summary
        collection_report['download_session'] = session_summary
        collection_report['file_inventory'] = download_results.get('file_inventory', {})
        collection_report['data_statistics'] = download_results.get('data_statistics', {})
        
        # Ensure reports directory exists
        os.makedirs('results/reports', exist_ok=True)
        
        # Save comprehensive report
        report_path = 'results/reports/eia_download_report.json'
        with open(report_path, 'w') as report_file:
            json.dump(collection_report, report_file, indent=2, default=str)
        
        logging.info(f"Collection report successfully saved to: {report_path}")
        
        # Generate human-readable summary
        logging.info("Collection Report Summary:")
        logging.info(f"  Total Downloads Attempted: {session_summary['total_attempted_downloads']}")
        logging.info(f"  Successful Downloads: {session_summary['successful_downloads']}")
        logging.info(f"  Success Rate: {session_summary['overall_success_rate']}")
        logging.info(f"  Report Location: {report_path}")
        
        return True
        
    except Exception as report_error:
        logging.error(f"Error generating collection report: {report_error}")
        return False

def execute_comprehensive_data_collection():
    """
    Main execution function that orchestrates the complete data collection workflow.
    
    This function implements the primary research data collection strategy, executing
    each data collection component in sequence while maintaining comprehensive error
    handling and progress tracking. The workflow is designed to maximize data
    acquisition success while providing detailed feedback on any issues encountered.
    
    Workflow Steps:
    1. Initialize data collector with API credentials
    2. Define collection time periods based on research requirements
    3. Execute primary demand data collection
    4. Collect complementary generation mix data
    5. Perform renewable energy analysis
    6. Generate synthetic validation data
    7. Create comprehensive documentation and reports
    8. Validate file outputs and provide next steps guidance
    
    Returns:
        int: Number of successful data collection operations
    """
    print("Energy Information Administration (EIA) Data Collection")
    print("MSc Research Project: Electricity Demand Forecasting and Renewable Integration")
    print("=" * 80)
    
    # Initialize collection tracking
    download_results = {
        'successful_count': 0,
        'total_attempted': 4,
        'data_statistics': {},
        'file_inventory': {},
        'quality_notes': []
    }
    
    # Step 1: Initialize data collector
    try:
        collector = initialize_data_collector()
        print("Data collector initialization: SUCCESS")
    except Exception as init_error:
        print(f"Data collector initialization: FAILED - {init_error}")
        return 0
    
    # Step 2: Define data collection time periods
    end_date = datetime.now()
    demand_start_date = end_date - timedelta(days=365)  # Full year for LSTM training
    generation_start_date = end_date - timedelta(days=90)  # Three months for efficiency
    
    print(f"Primary collection period (demand): {demand_start_date.date()} to {end_date.date()}")
    print(f"Secondary collection period (generation): {generation_start_date.date()} to {end_date.date()}")
    
    # Step 3: Execute electricity demand data collection
    print("\nStep 1 of 4: Electricity Demand Data Collection")
    print("-" * 50)
    demand_success, demand_summary = download_electricity_demand_data(
        collector, demand_start_date, end_date
    )
    
    if demand_success:
        download_results['successful_count'] += 1
        download_results['data_statistics']['demand'] = demand_summary
        download_results['file_inventory']['demand_file'] = {
            'path': 'data/raw/us_electricity_demand_2024.csv',
            'description': 'Hourly electricity demand data for US48 grid',
            'records': demand_summary['records']
        }
        print("Demand data collection: SUCCESS")
    else:
        print("Demand data collection: FAILED")
        download_results['quality_notes'].append("Demand data collection failed - check API connectivity")
    
    # Step 4: Execute generation mix data collection
    print("\nStep 2 of 4: Generation Mix Data Collection")
    print("-" * 50)
    generation_success, generation_summary = download_generation_mix_data(
        collector, generation_start_date, end_date
    )
    
    if generation_success:
        download_results['successful_count'] += 1
        download_results['data_statistics']['generation'] = generation_summary
        download_results['file_inventory']['generation_file'] = {
            'path': 'data/raw/us_generation_mix_3months.csv',
            'description': 'Electricity generation by fuel type',
            'records': generation_summary['records']
        }
        print("Generation mix data collection: SUCCESS")
    else:
        print("Generation mix data collection: FAILED")
        download_results['quality_notes'].append("Generation mix data collection failed")
    
    # Step 5: Execute renewable energy analysis
    print("\nStep 3 of 4: Renewable Energy Generation Analysis")
    print("-" * 50)
    renewable_success, renewable_summary = analyze_renewable_generation(
        collector, generation_start_date, end_date
    )
    
    if renewable_success:
        download_results['successful_count'] += 1
        download_results['data_statistics']['renewable'] = renewable_summary
        download_results['file_inventory']['renewable_file'] = {
            'path': 'data/raw/us_renewable_generation.csv',
            'description': 'Renewable energy generation analysis',
            'records': renewable_summary['records']
        }
        print("Renewable energy analysis: SUCCESS")
    else:
        print("Renewable energy analysis: FAILED")
        download_results['quality_notes'].append("Renewable energy analysis failed")
    
    # Step 6: Generate synthetic validation data
    print("\nStep 4 of 4: Synthetic Validation Data Generation")
    print("-" * 50)
    synthetic_success, synthetic_summary = generate_synthetic_validation_data(collector)
    
    if synthetic_success:
        download_results['successful_count'] += 1
        download_results['data_statistics']['synthetic'] = synthetic_summary
        download_results['file_inventory']['synthetic_file'] = {
            'path': 'data/raw/synthetic_demand_full_year.csv',
            'description': 'Synthetic demand data for algorithm validation',
            'records': synthetic_summary['total_hours']
        }
        print("Synthetic validation data generation: SUCCESS")
    else:
        print("Synthetic validation data generation: FAILED")
        download_results['quality_notes'].append("Synthetic data generation failed")
    
    # Step 7: Generate comprehensive collection report
    print("\nGenerating Collection Documentation")
    print("-" * 50)
    report_success = generate_comprehensive_collection_report(collector, download_results)
    
    if report_success:
        print("Collection report generation: SUCCESS")
    else:
        print("Collection report generation: FAILED")
    
    # Step 8: Provide final summary and next steps
    print("\n" + "=" * 80)
    print("DATA COLLECTION WORKFLOW COMPLETED")
    print("=" * 80)
    print(f"Successful Operations: {download_results['successful_count']}/{download_results['total_attempted']}")
    print(f"Overall Success Rate: {(download_results['successful_count']/download_results['total_attempted']*100):.0f}%")
    
    if download_results['successful_count'] > 0:
        print("\nData Files Successfully Created:")
        for file_key, file_info in download_results['file_inventory'].items():
            file_path = file_info['path']
            if os.path.exists(file_path):
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  {file_path}")
                print(f"    Description: {file_info['description']}")
                print(f"    Records: {file_info['records']:,}")
                print(f"    File Size: {file_size_mb:.1f} MB")
            else:
                print(f"  {file_path} (WARNING: File not found)")
        
        print("\nRecommended Next Steps for Research:")
        print("1. Data Quality Assessment:")
        print("   python src/data_pipeline/processors/data_cleaner.py")
        print("2. Exploratory Data Analysis:")
        print("   Open notebooks/01_data_exploration/eia_data_analysis.ipynb")
        print("3. Feature Engineering:")
        print("   python src/data_pipeline/processors/feature_engineer.py")
        print("4. LSTM Model Training:")
        print("   python src/forecasting/models/lstm_model.py")
        print("5. Model Evaluation:")
        print("   python src/forecasting/evaluation/model_evaluator.py")
        
        print("\nData Collection Successful - Ready for Analysis Phase")
        
        # Log final success metrics
        logging.info("Data collection workflow completed successfully")
        logging.info(f"Final success count: {download_results['successful_count']}/{download_results['total_attempted']}")
        
    else:
        print("\nData Collection Failed")
        print("Troubleshooting Steps:")
        print("1. Verify API key configuration in config/api_keys.yaml")
        print("2. Check internet connectivity and EIA API status")
        print("3. Review log files for detailed error information")
        print("4. Ensure sufficient disk space for data storage")
        
        logging.error("Data collection workflow failed - no successful downloads")
    
    return download_results['successful_count']

if __name__ == "__main__":
    """
    Main execution block for the EIA data collection script.
    
    This script is designed to be run as a standalone module for comprehensive
    energy data collection supporting MSc research in electricity demand forecasting
    and renewable energy integration analysis.
    """
    
    # Ensure required directories exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/metadata', exist_ok=True)
    os.makedirs('results/reports', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Execute comprehensive data collection workflow
    successful_downloads = execute_comprehensive_data_collection()
    
    # Provide final execution status
    if successful_downloads > 0:
        print(f"\nCollection completed successfully with {successful_downloads} datasets acquired")
        print("Research data is ready for analysis and model development")
        logging.info(f"Script execution completed successfully: {successful_downloads} downloads")
    else:
        print(f"\nCollection failed - no datasets were successfully acquired")
        print("Please review configuration and error logs before retrying")
        logging.error("Script execution failed - no successful downloads completed")
    
    # Exit with appropriate status code
    exit_code = 0 if successful_downloads > 0 else 1
    sys.exit(exit_code)