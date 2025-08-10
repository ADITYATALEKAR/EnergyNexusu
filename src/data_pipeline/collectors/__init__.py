"""
Data Pipeline Collectors Package
EnergyNexus MSc Project

This package contains specialized data collection modules for acquiring energy system data
from various sources including government databases, weather services, and energy market APIs.
Each collector is designed to handle the unique characteristics and requirements of different
energy data sources while providing a consistent interface for the overall system.

Data Sources Supported:
- NREL (National Renewable Energy Laboratory): Solar irradiance and wind resource data
- EIA (Energy Information Administration): Energy production and consumption statistics  
- Weather APIs: Meteorological data affecting renewable energy generation
- Grid Operators: Real-time generation and demand data
- Energy Markets: Pricing and trading data for optimization

Collection Capabilities:
- Real-time data streaming for operational forecasting
- Historical data batch processing for model training
- Data quality validation and error handling
- Automatic retry mechanisms for reliable data acquisition
- Rate limiting compliance for API usage policies
- Data format standardization across multiple sources

Author: Aditya Talekar (ec24018@qmul.ac.uk)
Supervisor: Saqib Iqbal
QMUL MSc Data Science and AI - 2024/25
"""

import os
import yaml
import logging
from pathlib import Path


"""
Data Pipeline Collectors Package
EnergyNexus MSc Project

This package contains specialized data collection modules for acquiring energy system data
from various sources including government databases, weather services, and energy market APIs.
Each collector is designed to handle the unique characteristics and requirements of different
energy data sources while providing a consistent interface for the overall system.

Data Sources Supported:
- NREL (National Renewable Energy Laboratory): Solar irradiance and wind resource data
- EIA (Energy Information Administration): Energy production and consumption statistics  
- Weather APIs: Meteorological data affecting renewable energy generation
- Grid Operators: Real-time generation and demand data
- Energy Markets: Pricing and trading data for optimization

Collection Capabilities:
- Real-time data streaming for operational forecasting
- Historical data batch processing for model training
- Data quality validation and error handling
- Automatic retry mechanisms for reliable data acquisition
- Rate limiting compliance for API usage policies
- Data format standardization across multiple sources

Author: Aditya Talekar (ec24018@qmul.ac.uk)
Supervisor: Saqib Iqbal
QMUL MSc Data Science and AI - 2024/25
"""

import os
import yaml
import logging
from pathlib import Path

# Package version for tracking data collection capabilities across thesis development
__version__ = "1.0.0"

# Package metadata for academic documentation and system tracking
__author__ = "Aditya Talekar"
__email__ = "ec24018@qmul.ac.uk"
__institution__ = "Queen Mary University of London"
__project__ = "EnergyNexus: Advanced Scheduling Algorithms for Integrated Power Systems"
__research_area__ = "Multi-Source Energy Data Collection and Integration"

def load_api_keys(config_path=None):
    """
    Load API keys from YAML configuration file.
    
    Args:
        config_path (str, optional): Path to the API keys configuration file.
                                   If None, will look for config/api_keys.yaml
    
    Returns:
        dict: Dictionary containing API keys for different services
    
    Raises:
        FileNotFoundError: If the configuration file is not found
        yaml.YAMLError: If the YAML file is malformed
    """
    if config_path is None:
        # I look for the config file in standard locations
        possible_paths = [
            Path(__file__).parent.parent.parent / "config" / "api_keys.yaml",
            Path("config/api_keys.yaml"),
            Path("../config/api_keys.yaml"),
            Path("../../config/api_keys.yaml"),
            Path(r"C:\Users\ADITYA\OneDrive\Desktop\EnergyNexus\config\api_keys.yaml")
        ]
        
        config_path = None
        for path in possible_paths:
            if path.exists():
                config_path = path
                break
    
    if config_path is None:
        raise FileNotFoundError(
            "Could not find api_keys.yaml configuration file. "
            "Please ensure it exists in one of these locations:\n"
            + "\n".join([str(p) for p in possible_paths])
        )
    
    try:
        with open(config_path, 'r') as file:
            api_keys = yaml.safe_load(file)
        
        logging.info(f"Successfully loaded API keys from {config_path}")
        return api_keys
        
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration file: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading API keys configuration: {e}")
        raise

def get_collector_with_keys(collector_type, config_path=None, **kwargs):
    """
    Initialize a data collector with API keys loaded from configuration.
    
    Args:
        collector_type (str): Type of collector ('nrel', 'eia', 'weather')
        config_path (str, optional): Path to API keys configuration file
        **kwargs: Additional arguments to pass to the collector
    
    Returns:
        BaseDataCollector: Initialized collector with API key
    
    Raises:
        ValueError: If collector_type is not supported
        KeyError: If required API key is not found in configuration
    """
    
    # I import collectors here to avoid circular imports
    try:
        from .base_collector import BaseDataCollector
        from .nrel_collector import NRELDataCollector
        from .eia_collector import EIADataCollector  
        from .weather_collector import WeatherDataCollector
    except ImportError as e:
        logging.error(f"Failed to import collector classes: {e}")
        raise ImportError(f"Collector modules not found. Please ensure all collector files exist: {e}")
    
    # I load the API keys from configuration
    try:
        api_keys = load_api_keys(config_path)
    except Exception as e:
        logging.warning(f"Could not load API keys: {e}")
        logging.warning("Initializing collector without API key - some functionality may be limited")
        api_keys = {}
    
    # I map collector types to their classes and required API key names
    collector_mapping = {
        'nrel': {
            'class': NRELDataCollector,
            'api_key_name': 'nrel_api_key'
        },
        'eia': {
            'class': EIADataCollector,
            'api_key_name': 'eia_api_key'
        },
        'weather': {
            'class': WeatherDataCollector,
            'api_key_name': 'openweather_api_key'
        }
    }
    
    if collector_type.lower() not in collector_mapping:
        raise ValueError(
            f"Unsupported collector type: {collector_type}. "
            f"Supported types: {list(collector_mapping.keys())}"
        )
    
    collector_info = collector_mapping[collector_type.lower()]
    collector_class = collector_info['class']
    api_key_name = collector_info['api_key_name']
    
    # I get the API key from the configuration
    api_key = api_keys.get(api_key_name)
    
    if api_key is None:
        logging.warning(f"API key '{api_key_name}' not found in configuration")
        logging.warning("Collector will be initialized without API key")
    
    # I initialize the collector with the API key
    try:
        if api_key:
            collector = collector_class(api_key=api_key, **kwargs)
        else:
            collector = collector_class(**kwargs)
        
        logging.info(f"Successfully initialized {collector_type} collector")
        return collector
        
    except Exception as e:
        logging.error(f"Error initializing {collector_type} collector: {e}")
        raise

# I make collector classes available for direct import
try:
    from .base_collector import BaseDataCollector
    from .nrel_collector import NRELDataCollector
    from .eia_collector import EIADataCollector  
    from .weather_collector import WeatherDataCollector
    
    # I make all collector classes available when importing the package
    __all__ = [
        "BaseDataCollector",        # Abstract base class defining collector interface
        "NRELDataCollector",        # NREL solar and wind resource data collection
        "EIADataCollector",         # EIA energy statistics and market data collection
        "WeatherDataCollector",     # Weather data collection for renewable forecasting
        "load_api_keys",           # Utility function for loading API keys
        "get_collector_with_keys"   # Helper function to initialize collectors with keys
    ]
    
except ImportError as e:
    logging.warning(f"Some collector classes could not be imported: {e}")
    logging.warning("Please ensure all collector files exist in the collectors directory")
    
    # I provide minimal functionality even if imports fail
    __all__ = [
        "load_api_keys",           # Utility function for loading API keys
        "get_collector_with_keys"   # Helper function to initialize collectors with keys
    ]

# I document the data sources and collection capabilities for thesis methodology section
__data_sources_info__ = {
    "NREL": {
        "full_name": "National Renewable Energy Laboratory",
        "organization": "U.S. Department of Energy",
        "data_types": ["Solar irradiance", "Wind resource", "Photovoltaic output", "Weather data"],
        "temporal_resolution": ["Hourly", "Daily", "Monthly", "Annual"],
        "spatial_coverage": "Global with high resolution for United States",
        "api_documentation": "https://developer.nrel.gov/docs/",
        "citation": "NREL. (2024). Developer Network API Documentation. National Renewable Energy Laboratory.",
        "usage_in_project": "Primary source for renewable energy resource assessment and historical generation patterns"
    },
    "EIA": {
        "full_name": "Energy Information Administration", 
        "organization": "U.S. Department of Energy",
        "data_types": ["Electricity generation", "Energy consumption", "Fuel prices", "Grid operations"],
        "temporal_resolution": ["Real-time", "Hourly", "Daily", "Monthly"],
        "spatial_coverage": "United States with state and regional breakdowns",
        "api_documentation": "https://www.eia.gov/opendata/",
        "citation": "EIA. (2024). Open Data API. U.S. Energy Information Administration.",
        "usage_in_project": "Energy market data, demand patterns, and conventional generation statistics"
    },
    "Weather_APIs": {
        "primary_sources": ["OpenWeatherMap", "NOAA", "MeteoGroup"],
        "data_types": ["Temperature", "Wind speed/direction", "Cloud cover", "Humidity", "Pressure"],
        "temporal_resolution": ["Current", "Hourly forecasts", "Daily forecasts"],
        "spatial_coverage": "Global coverage with location-specific data",
        "usage_in_project": "Weather feature engineering for renewable energy forecasting models"
    }
}

def create_sample_api_keys_file(filepath):
    """
    Create a sample API keys configuration file for reference.
    
    Args:
        filepath (str): Path where to create the sample file
    """
    
    sample_config = {
        'nrel_api_key': 'your_nrel_api_key_here',
        'eia_api_key': 'your_eia_api_key_here', 
        'openweather_api_key': 'your_openweather_api_key_here',
        'noaa_api_key': 'your_noaa_api_key_here',
        
        # API endpoint configurations (optional)
        'api_endpoints': {
            'nrel_base_url': 'https://developer.nrel.gov/api/',
            'eia_base_url': 'https://api.eia.gov/',
            'openweather_base_url': 'https://api.openweathermap.org/data/2.5/'
        },
        
        # Rate limiting configurations
        'rate_limits': {
            'nrel_requests_per_hour': 1000,
            'eia_requests_per_hour': 1000,
            'openweather_requests_per_minute': 60
        },
        
        # Data collection preferences
        'collection_settings': {
            'default_timeout': 30,
            'max_retries': 3,
            'retry_delay': 1,
            'cache_enabled': True
        }
    }
    
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as file:
        yaml.dump(sample_config, file, default_flow_style=False, indent=2)
    
    print(f"Sample API keys configuration created at: {filepath}")
    print("Please edit this file with your actual API keys")

# Set up logging for the package
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Package version for tracking data collection capabilities across thesis development
__version__ = "1.0.0"

# I make all collector classes available when importing the package
# This follows Python package organization best practices
__all__ = [
    "BaseDataCollector",        # Abstract base class defining collector interface
    "NRELDataCollector",        # NREL solar and wind resource data collection
    "EIADataCollector",         # EIA energy statistics and market data collection
    "WeatherDataCollector",     # Weather data collection for renewable forecasting
    "load_api_keys",           # Utility function for loading API keys
    "get_collector_with_keys"   # Helper function to initialize collectors with keys
]

# Package metadata for academic documentation and system tracking
__author__ = "Aditya Talekar"
__email__ = "ec24018@qmul.ac.uk"
__institution__ = "Queen Mary University of London"
__project__ = "EnergyNexus: Advanced Scheduling Algorithms for Integrated Power Systems"
__research_area__ = "Multi-Source Energy Data Collection and Integration"

def load_api_keys(config_path=None):
    """
    Load API keys from YAML configuration file.
    
    Args:
        config_path (str, optional): Path to the API keys configuration file.
                                   If None, will look for config/api_keys.yaml
    
    Returns:
        dict: Dictionary containing API keys for different services
    
    Raises:
        FileNotFoundError: If the configuration file is not found
        yaml.YAMLError: If the YAML file is malformed
    """
    if config_path is None:
        # I look for the config file in standard locations
        possible_paths = [
            Path(__file__).parent.parent.parent / "config" / "api_keys.yaml",
            Path("config/api_keys.yaml"),
            Path("../config/api_keys.yaml"),
            Path("../../config/api_keys.yaml"),
            Path(r"C:\Users\ADITYA\OneDrive\Desktop\EnergyNexus\config\api_keys.yaml")
        ]
        
        config_path = None
        for path in possible_paths:
            if path.exists():
                config_path = path
                break
    
    if config_path is None:
        raise FileNotFoundError(
            "Could not find api_keys.yaml configuration file. "
            "Please ensure it exists in one of these locations:\n"
            + "\n".join([str(p) for p in possible_paths])
        )
    
    try:
        with open(config_path, 'r') as file:
            api_keys = yaml.safe_load(file)
        
        logging.info(f"Successfully loaded API keys from {config_path}")
        return api_keys
        
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration file: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading API keys configuration: {e}")
        raise

def get_collector_with_keys(collector_type, config_path=None, **kwargs):
    """
    Initialize a data collector with API keys loaded from configuration.
    
    Args:
        collector_type (str): Type of collector ('nrel', 'eia', 'weather')
        config_path (str, optional): Path to API keys configuration file
        **kwargs: Additional arguments to pass to the collector
    
    Returns:
        BaseDataCollector: Initialized collector with API key
    
    Raises:
        ValueError: If collector_type is not supported
        KeyError: If required API key is not found in configuration
    """
    
    # I load the API keys from configuration
    try:
        api_keys = load_api_keys(config_path)
    except Exception as e:
        logging.warning(f"Could not load API keys: {e}")
        logging.warning("Initializing collector without API key - some functionality may be limited")
        api_keys = {}
    
    # I map collector types to their classes and required API key names
    collector_mapping = {
        'nrel': {
            'class': NRELDataCollector,
            'api_key_name': 'nrel_api_key'
        },
        'eia': {
            'class': EIADataCollector,
            'api_key_name': 'eia_api_key'
        },
        'weather': {
            'class': WeatherDataCollector,
            'api_key_name': 'openweather_api_key'
        }
    }
    
    if collector_type.lower() not in collector_mapping:
        raise ValueError(
            f"Unsupported collector type: {collector_type}. "
            f"Supported types: {list(collector_mapping.keys())}"
        )
    
    collector_info = collector_mapping[collector_type.lower()]
    collector_class = collector_info['class']
    api_key_name = collector_info['api_key_name']
    
    # I get the API key from the configuration
    api_key = api_keys.get(api_key_name)
    
    if api_key is None:
        logging.warning(f"API key '{api_key_name}' not found in configuration")
        logging.warning("Collector will be initialized without API key")
    
    # I initialize the collector with the API key
    try:
        if api_key:
            collector = collector_class(api_key=api_key, **kwargs)
        else:
            collector = collector_class(**kwargs)
        
        logging.info(f"Successfully initialized {collector_type} collector")
        return collector
        
    except Exception as e:
        logging.error(f"Error initializing {collector_type} collector: {e}")
        raise

# I document the data sources and collection capabilities for thesis methodology section
__data_sources_info__ = {
    "NREL": {
        "full_name": "National Renewable Energy Laboratory",
        "organization": "U.S. Department of Energy",
        "data_types": ["Solar irradiance", "Wind resource", "Photovoltaic output", "Weather data"],
        "temporal_resolution": ["Hourly", "Daily", "Monthly", "Annual"],
        "spatial_coverage": "Global with high resolution for United States",
        "api_documentation": "https://developer.nrel.gov/docs/",
        "citation": "NREL. (2024). Developer Network API Documentation. National Renewable Energy Laboratory.",
        "usage_in_project": "Primary source for renewable energy resource assessment and historical generation patterns"
    },
    "EIA": {
        "full_name": "Energy Information Administration", 
        "organization": "U.S. Department of Energy",
        "data_types": ["Electricity generation", "Energy consumption", "Fuel prices", "Grid operations"],
        "temporal_resolution": ["Real-time", "Hourly", "Daily", "Monthly"],
        "spatial_coverage": "United States with state and regional breakdowns",
        "api_documentation": "https://www.eia.gov/opendata/",
        "citation": "EIA. (2024). Open Data API. U.S. Energy Information Administration.",
        "usage_in_project": "Energy market data, demand patterns, and conventional generation statistics"
    },
    "Weather_APIs": {
        "primary_sources": ["OpenWeatherMap", "NOAA", "MeteoGroup"],
        "data_types": ["Temperature", "Wind speed/direction", "Cloud cover", "Humidity", "Pressure"],
        "temporal_resolution": ["Current", "Hourly forecasts", "Daily forecasts"],
        "spatial_coverage": "Global coverage with location-specific data",
        "usage_in_project": "Weather feature engineering for renewable energy forecasting models"
    }
}

# I provide comprehensive usage examples for data collection workflows
def get_usage_examples():
    """
    Return comprehensive usage examples for the data collection package.
    
    Returns:
        str: Multi-line string containing usage examples
    """
    
    return 

# Complete Data Collection Workflow for Energy System Analysis:

import pandas as pd
from datetime import datetime, timedelta
from data_pipeline.collectors import get_collector_with_keys, load_api_keys

def main_data_collection_workflow():
    '''Complete workflow for collecting energy data for analysis'''
    
    # Step 1: Load API keys from configuration
    try:
        api_keys = load_api_keys()
        print("API keys loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load API keys: {e}")
        print("Some data collection may use sample/synthetic data")
    
    # Step 2: Initialize data collectors with API keys
    print("Initializing data collectors...")
    
    try:
        nrel_collector = get_collector_with_keys('nrel')
        print("✓ NREL collector initialized")
    except Exception as e:
        print(f"✗ NREL collector failed: {e}")
        nrel_collector = None
    
    try:
        eia_collector = get_collector_with_keys('eia')
        print("✓ EIA collector initialized")
    except Exception as e:
        print(f"✗ EIA collector failed: {e}")
        eia_collector = None
    
    try:
        weather_collector = get_collector_with_keys('weather')
        print("✓ Weather collector initialized")
    except Exception as e:
        print(f"✗ Weather collector failed: {e}")
        weather_collector = None
    
    # Step 3: Define data collection parameters for your study region
    # Example: London, UK coordinates for European renewable energy analysis
    location_params = {
        'latitude': 51.5074,      # London latitude
        'longitude': -0.1278,     # London longitude
        'location_name': 'London_UK'
    }
    
    # Define time range for historical analysis and model training
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # 30 days of data for testing
    
    print(f"Collecting energy data from {start_date.date()} to {end_date.date()}")
    print(f"Location: {location_params['location_name']} ({location_params['latitude']}, {location_params['longitude']})")
    
    # Step 4: Collect solar resource data from NREL
    if nrel_collector:
        print("Collecting solar irradiance and photovoltaic data...")
        try:
            solar_data = nrel_collector.get_solar_data(
                latitude=location_params['latitude'],
                longitude=location_params['longitude'],
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                attributes=['ghi', 'dni', 'dhi', 'air_temperature', 'wind_speed']
            )
            
            if solar_data is not None:
                print(f"✓ Solar data collected: {len(solar_data)} records")
                solar_data.to_csv(f"data/raw/solar_data_{location_params['location_name']}.csv")
            else:
                print("⚠ No solar data returned - using synthetic data")
                solar_data = nrel_collector.create_sample_solar_data(hours=24*30)
        except Exception as e:
            print(f"✗ Solar data collection failed: {e}")
            solar_data = None
    else:
        print("⚠ NREL collector not available - skipping solar data collection")
        solar_data = None
    
    # Step 5: Collect wind resource data
    if nrel_collector:
        print("Collecting wind resource data...")
        try:
            wind_data = nrel_collector.get_wind_data(
                latitude=location_params['latitude'],
                longitude=location_params['longitude'],
                hub_height=80,  # Standard wind turbine hub height
                attributes=['windspeed', 'winddirection', 'temperature', 'pressure']
            )
            
            if wind_data is not None:
                print(f"✓ Wind data collected: {len(wind_data)} records")
                wind_data.to_csv(f"data/raw/wind_data_{location_params['location_name']}.csv")
            else:
                print("⚠ No wind data returned")
        except Exception as e:
            print(f"✗ Wind data collection failed: {e}")
            wind_data = None
    
    # Step 6: Collect energy market and demand data
    if eia_collector:
        print("Collecting energy market data...")
        try:
            demand_data = eia_collector.get_electricity_demand(
                region='US48',  # Continental United States
                start_date=start_date,
                end_date=end_date,
                frequency='hourly'
            )
            
            if demand_data is not None:
                print(f"✓ Demand data collected: {len(demand_data)} records")
                demand_data.to_csv("data/raw/electricity_demand_US48.csv")
            else:
                print("⚠ No demand data returned")
        except Exception as e:
            print(f"✗ Demand data collection failed: {e}")
            demand_data = None
    
    # Step 7: Collect real-time weather data for operational forecasting
    if weather_collector:
        print("Collecting current weather conditions...")
        try:
            current_weather = weather_collector.get_current_weather(
                latitude=location_params['latitude'],
                longitude=location_params['longitude']
            )
            
            if current_weather:
                print("✓ Current weather data collected")
            else:
                print("⚠ No current weather data returned")
        except Exception as e:
            print(f"✗ Current weather collection failed: {e}")
            current_weather = None
        
        # Collect weather forecasts for predictive modeling
        try:
            weather_forecast = weather_collector.get_weather_forecast(
                latitude=location_params['latitude'],
                longitude=location_params['longitude'],
                forecast_hours=72  # 3-day forecast for multi-horizon modeling
            )
            
            if weather_forecast is not None:
                print(f"✓ Weather forecast collected: {len(weather_forecast)} forecast points")
                weather_forecast.to_csv("data/raw/weather_forecast_72h.csv")
            else:
                print(" No weather forecast returned")
        except Exception as e:
            print(f"✗ Weather forecast collection failed: {e}")
            weather_forecast = None
    
    # Step 8: Create collection summary
    collection_summary = {
        'collection_date': datetime.now().isoformat(),
        'location': location_params,
        'time_range': {'start': start_date.isoformat(), 'end': end_date.isoformat()},
        'successful_collections': [],
        'failed_collections': []
    }
    
    # Check which collections succeeded
    if solar_data is not None:
        collection_summary['successful_collections'].append('solar')
    else:
        collection_summary['failed_collections'].append('solar')
    
    if 'wind_data' in locals() and wind_data is not None:
        collection_summary['successful_collections'].append('wind')
    else:
        collection_summary['failed_collections'].append('wind')
    
    if 'demand_data' in locals() and demand_data is not None:
        collection_summary['successful_collections'].append('demand')
    else:
        collection_summary['failed_collections'].append('demand')
    
    if 'weather_forecast' in locals() and weather_forecast is not None:
        collection_summary['successful_collections'].append('weather')
    else:
        collection_summary['failed_collections'].append('weather')
    
    print(f"\\nData collection completed!")
    print(f"Successful collections: {len(collection_summary['successful_collections'])}")
    print(f"Failed collections: {len(collection_summary['failed_collections'])}")
    
    # Save collection metadata
    import json
    import os
    os.makedirs("data/metadata", exist_ok=True)
    with open("data/metadata/collection_summary.json", 'w') as f:
        json.dump(collection_summary, f, indent=2, default=str)
    
    return collection_summary

# Example for operational data collection with scheduling
def setup_operational_collection():
    '''Set up automated operational data collection'''
    
    try:
        import schedule
        import time
    except ImportError:
        print("Please install 'schedule' package: pip install schedule")
        return
    
    def collect_operational_data():
        '''Collect real-time data for operational forecasting every hour'''
        
        # Initialize collectors
        try:
            weather_collector = get_collector_with_keys('weather')
            eia_collector = get_collector_with_keys('eia')
        except Exception as e:
            print(f"Error initializing collectors: {e}")
            return
        
        # Grid coordinates for your operational area
        grid_locations = [
            {'name': 'London', 'lat': 51.5074, 'lon': -0.1278}
        ]
        
        timestamp = datetime.now()
        
        for location in grid_locations:
            try:
                # Collect current weather for immediate forecasting
                current_weather = weather_collector.get_current_weather(
                    latitude=location['lat'],
                    longitude=location['lon']
                )
                
                # Collect short-term weather forecast for dispatch planning
                forecast_6h = weather_collector.get_weather_forecast(
                    latitude=location['lat'],
                    longitude=location['lon'],
                    forecast_hours=6
                )
                
                # Save operational data with timestamp
                if current_weather:
                    import os
                    os.makedirs("data/operational", exist_ok=True)
                    filename = f"data/operational/weather_{location['name']}_{timestamp.strftime('%Y%m%d_%H%M')}.json"
                    
                    import json
                    with open(filename, 'w') as f:
                        json.dump({
                            'timestamp': timestamp.isoformat(),
                            'location': location,
                            'current_weather': current_weather,
                            'forecast_6h': forecast_6h.to_dict() if forecast_6h is not None else None
                        }, f, indent=2, default=str)
                
                print(f"Operational data collected for {location['name']} at {timestamp}")
                
            except Exception as e:
                print(f"Error collecting data for {location['name']}: {e}")
    
    # Schedule automatic data collection for operational forecasting
    schedule.every().hour.at(":05").do(collect_operational_data)
    
    print("Operational data collection scheduler started...")
    print("Data will be collected hourly for real-time forecasting")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        print("\\nOperational collection stopped")

if __name__ == "__main__":
    # Run the main data collection workflow
    summary = main_data_collection_workflow()
    
    # Optionally set up operational collection
    # setup_operational_collection()


# I include comprehensive citation information for data sources
__data_citations__ = """
Data Source Citations for Academic Use:

Primary Data Sources:
1. National Renewable Energy Laboratory (NREL):
   NREL. (2024). National Solar Radiation Database (NSRDB) and Wind Integration 
   National Dataset (WIND) Toolkit. Golden, CO: National Renewable Energy Laboratory. 
   Available: https://www.nrel.gov/gis/data.html

2. U.S. Energy Information Administration (EIA):
   EIA. (2024). Electricity Data Browser and Open Data API. 
   Washington, DC: U.S. Energy Information Administration.
   Available: https://www.eia.gov/electricity/data/browser/

3. Weather Data Sources:
   OpenWeatherMap. (2024). Weather API and Historical Weather Data.
   Available: https://openweathermap.org/api
   
   NOAA. (2024). National Weather Service API and Climate Data.
   National Oceanic and Atmospheric Administration.
   Available: https://www.weather.gov/documentation/services-web-api

BibTeX Entries:
@misc{nrel2024nsrdb,
    title={National Solar Radiation Database (NSRDB) and Wind Integration National Dataset (WIND) Toolkit},
    author={{National Renewable Energy Laboratory}},
    year={2024},
    publisher={National Renewable Energy Laboratory},
    address={Golden, CO},
    url={https://www.nrel.gov/gis/data.html},
    note={Accessed: 2024}
}

@misc{eia2024electricity,
    title={Electricity Data Browser and Open Data API},
    author={{U.S. Energy Information Administration}},
    year={2024},
    publisher={U.S. Energy Information Administration},
    address={Washington, DC},
    url={https://www.eia.gov/electricity/data/browser/},
    note={Accessed: 2024}
}

Data Usage Acknowledgments:
- Solar irradiance data provided by NREL's National Solar Radiation Database (NSRDB)
- Wind resource data from NREL's Wind Integration National Dataset (WIND) Toolkit  
- Electricity market data from EIA's Open Data initiative
- Weather data from OpenWeatherMap and NOAA weather services
- All data used in compliance with respective terms of service and attribution requirements

Research Ethics and Data Usage:
- All data collection follows API terms of service and rate limiting requirements
- Personal and proprietary information is excluded from data collection
- Data is used solely for academic research purposes under educational license
- Commercial deployment would require appropriate licensing agreements
- Data quality and source reliability documented for research reproducibility

Data Collection Best Practices Implemented:
- Robust error handling and retry mechanisms for reliable data acquisition
- Rate limiting compliance to respect API usage policies
- Data validation and quality checking at collection time
- Metadata preservation for research reproducibility
- Secure API key management and credential handling
- Comprehensive logging for debugging and audit trails
"""

# I include collector interface standards for extending the system
__collector_interface__ = """
Standard Interface for Energy Data Collectors:

All collectors in this package implement the BaseDataCollector interface to ensure
consistent behavior and easy extensibility. When adding new data sources, follow
this standard interface:

Required Methods:
- __init__(api_key, **kwargs): Initialize collector with credentials
- validate_connection(): Test API connectivity and credentials
- get_data(start_date, end_date, **params): Retrieve historical data
- get_current_data(**params): Retrieve real-time data  
- save_data(data, filepath): Save collected data with metadata
- get_data_quality_report(data): Assess data completeness and quality

Error Handling Standards:
- Network timeout handling with exponential backoff retry
- API rate limit compliance with automatic throttling
- Comprehensive logging for debugging and monitoring
- Graceful fallback to cached or synthetic data when available
- Data validation with quality scoring and issue reporting

Configuration Management:
- API credentials secured through environment variables or config files
- Parameterized collection settings for different use cases
- Location and temporal range validation before collection
- Output format standardization across all collectors

Extension Guidelines:
To add support for new data sources, inherit from BaseDataCollector and implement
the required interface methods. Follow the existing patterns for error handling,
logging, and data format standardization.
"""

# Example API keys YAML structure for documentation
def create_sample_api_keys_file(filepath):
    """
    Create a sample API keys configuration file for reference.
    
    Args:
        filepath (str): Path where to create the sample file
    """
    
    sample_config = {
        'nrel_api_key': 'your_nrel_api_key_here',
        'eia_api_key': 'your_eia_api_key_here', 
        'openweather_api_key': 'your_openweather_api_key_here',
        'noaa_api_key': 'your_noaa_api_key_here',
        
        # API endpoint configurations (optional)
        'api_endpoints': {
            'nrel_base_url': 'https://developer.nrel.gov/api/',
            'eia_base_url': 'https://api.eia.gov/',
            'openweather_base_url': 'https://api.openweathermap.org/data/2.5/'
        },
        
        # Rate limiting configurations
        'rate_limits': {
            'nrel_requests_per_hour': 1000,
            'eia_requests_per_hour': 1000,
            'openweather_requests_per_minute': 60
        },
        
        # Data collection preferences
        'collection_settings': {
            'default_timeout': 30,
            'max_retries': 3,
            'retry_delay': 1,
            'cache_enabled': True
        }
    }
    
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as file:
        yaml.dump(sample_config, file, default_flow_style=False, indent=2)
    
    print(f"Sample API keys configuration created at: {filepath}")
    print("Please edit this file with your actual API keys")

# Set up logging for the package
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)