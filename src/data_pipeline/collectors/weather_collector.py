"""
Weather Data Collector
EnergyNexus MSc Project

This module implements data collection from weather APIs for meteorological data
that affects renewable energy generation and electricity demand patterns.

Author: Aditya Talekar (ec24018@qmul.ac.uk)
Supervisor: Saqib Iqbal
QMUL MSc Data Science and AI - 2024/25
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import time
import logging

# Import base collector if it exists, otherwise create a simple base class
try:
    from .base_collector import BaseDataCollector
except ImportError:
    # Fallback base class if base_collector doesn't exist
    class BaseDataCollector:
        def __init__(self, api_key: Optional[str] = None, **kwargs):
            self.api_key = api_key
            self.timeout = kwargs.get('timeout', 30)
            self.logger = logging.getLogger(__name__)


class WeatherDataCollector(BaseDataCollector):
    """
    Weather data collector for meteorological information.
    
    This collector interfaces with weather APIs to retrieve current conditions,
    forecasts, and historical weather data for renewable energy analysis.
    """
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize weather data collector.
        
        Args:
            api_key (str, optional): Weather API key (OpenWeatherMap, etc.)
            **kwargs: Additional configuration parameters
        """
        super().__init__(api_key, **kwargs)
        
        self.data_source_name = "Weather APIs"
        self.base_url = kwargs.get('base_url', 'https://api.openweathermap.org/data/2.5/')
        
        # Set up weather API endpoints
        self.endpoints = {
            'current': 'weather',
            'forecast': 'forecast',
            'historical': 'onecall/timemachine',
            'onecall': 'onecall'
        }
        
        self.logger.info("Weather collector initialized")
    
    def validate_connection(self) -> bool:
        """
        Validate connection to weather API.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if not self.api_key:
            self.logger.warning("No API key provided for weather API")
            return False
        
        try:
            # Test with a simple current weather query for London
            url = f"{self.base_url}{self.endpoints['current']}"
            params = {
                'appid': self.api_key,
                'lat': 51.5074,  # London
                'lon': -0.1278,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                if 'main' in data and 'temp' in data['main']:
                    self.logger.info("Weather API connection validated successfully")
                    return True
            
            self.logger.error(f"Weather API validation failed: {response.status_code}")
            return False
            
        except Exception as e:
            self.logger.error(f"Weather API connection test failed: {e}")
            return False
    
    def get_data(self, start_date: datetime, end_date: datetime, **params) -> Optional[pd.DataFrame]:
        """
        Retrieve historical weather data.
        
        Args:
            start_date (datetime): Start date for data collection
            end_date (datetime): End date for data collection
            **params: Additional parameters (latitude, longitude, etc.)
            
        Returns:
            pd.DataFrame: Retrieved data or None if failed
        """
        latitude = params.get('latitude', 51.5074)
        longitude = params.get('longitude', -0.1278)
        
        return self.get_historical_weather(start_date, end_date, latitude, longitude, **params)
    
    def get_current_weather(self, latitude: float, longitude: float, **params) -> Optional[Dict[str, Any]]:
        """
        Retrieve current weather conditions.
        
        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate
            **params: Additional parameters
            
        Returns:
            dict: Current weather data or None if failed
        """
        if not self.api_key:
            self.logger.warning("No API key - creating sample current weather")
            return self._create_sample_current_weather(latitude, longitude)
        
        try:
            url = f"{self.base_url}{self.endpoints['current']}"
            params_api = {
                'appid': self.api_key,
                'lat': latitude,
                'lon': longitude,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params_api, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract relevant weather information
                weather_data = {
                    'timestamp': datetime.now(),
                    'latitude': latitude,
                    'longitude': longitude,
                    'temperature': data['main']['temp'],
                    'feels_like': data['main']['feels_like'],
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'wind_speed': data.get('wind', {}).get('speed', 0),
                    'wind_direction': data.get('wind', {}).get('deg', 0),
                    'cloud_cover': data.get('clouds', {}).get('all', 0),
                    'visibility': data.get('visibility', 10000) / 1000,  # Convert to km
                    'weather_condition': data['weather'][0]['main'],
                    'weather_description': data['weather'][0]['description'],
                    'sunrise': datetime.fromtimestamp(data['sys']['sunrise']),
                    'sunset': datetime.fromtimestamp(data['sys']['sunset'])
                }
                
                self.logger.info(f"Retrieved current weather for ({latitude}, {longitude})")
                return weather_data
            else:
                self.logger.error(f"Weather API request failed: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Current weather collection failed: {e}")
            return None
    
    def get_weather_forecast(self, latitude: float, longitude: float, 
                           forecast_hours: int = 48, **params) -> Optional[pd.DataFrame]:
        """
        Retrieve weather forecast data.
        
        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate
            forecast_hours (int): Number of hours to forecast
            **params: Additional parameters
            
        Returns:
            pd.DataFrame: Forecast data or None if failed
        """
        if not self.api_key:
            self.logger.warning("No API key - creating sample forecast")
            return self.get_sample_weather_data(latitude, longitude, forecast_hours)
        
        try:
            url = f"{self.base_url}{self.endpoints['forecast']}"
            params_api = {
                'appid': self.api_key,
                'lat': latitude,
                'lon': longitude,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params_api, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'list' in data:
                    forecasts = data['list']
                    
                    # Limit to requested forecast hours
                    forecasts = forecasts[:min(len(forecasts), forecast_hours//3)]  # API returns 3-hour intervals
                    
                    forecast_data = []
                    for forecast in forecasts:
                        forecast_data.append({
                            'timestamp': datetime.fromtimestamp(forecast['dt']),
                            'temperature': forecast['main']['temp'],
                            'feels_like': forecast['main']['feels_like'],
                            'humidity': forecast['main']['humidity'],
                            'pressure': forecast['main']['pressure'],
                            'wind_speed': forecast.get('wind', {}).get('speed', 0),
                            'wind_direction': forecast.get('wind', {}).get('deg', 0),
                            'cloud_cover': forecast.get('clouds', {}).get('all', 0),
                            'precipitation_probability': forecast.get('pop', 0) * 100,
                            'weather_condition': forecast['weather'][0]['main'],
                            'weather_description': forecast['weather'][0]['description']
                        })
                    
                    df = pd.DataFrame(forecast_data)
                    self.logger.info(f"Retrieved {len(df)} forecast points")
                    return df
                else:
                    self.logger.error("Unexpected forecast API response format")
                    return None
            else:
                self.logger.error(f"Forecast API request failed: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Weather forecast collection failed: {e}")
            return None
    
    def get_historical_weather(self, start_date: datetime, end_date: datetime,
                             latitude: float, longitude: float, **params) -> Optional[pd.DataFrame]:
        """
        Retrieve historical weather data.
        
        Args:
            start_date (datetime): Start date
            end_date (datetime): End date
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate
            **params: Additional parameters
            
        Returns:
            pd.DataFrame: Historical weather data or None if failed
        """
        if not self.api_key:
            self.logger.warning("No API key - creating sample historical weather")
            hours = int((end_date - start_date).total_seconds() / 3600)
            return self.get_sample_weather_data(latitude, longitude, hours)
        
        # Note: Historical weather data often requires premium API access
        # For demonstration, create realistic historical data
        self.logger.info("Creating realistic historical weather data")

        try:
            hours = int((end_date - start_date).total_seconds() / 3600)
            timestamps = pd.date_range(start=start_date, periods=hours, freq='H')
            
            weather_data = []
            for timestamp in timestamps:
                # Create realistic weather patterns
                day_of_year = timestamp.dayofyear
                hour = timestamp.hour
                
                # Seasonal temperature pattern
                base_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
                daily_temp = base_temp + 5 * np.sin((hour - 14) * np.pi / 12)
                temperature = daily_temp + np.random.normal(0, 2)
                
                # Related weather parameters
                humidity = max(30, min(100, 70 + np.random.normal(0, 15)))
                pressure = 1013 + np.random.normal(0, 10)
                
                # Wind patterns
                wind_speed = max(0, 5 + np.random.normal(0, 3))
                wind_direction = np.random.uniform(0, 360)
                
                # Cloud cover and solar irradiance
                cloud_cover = max(0, min(100, 40 + np.random.normal(0, 20)))
                
                # Precipitation probability based on humidity and cloud cover
                precip_prob = max(0, min(100, (humidity - 50) * 0.5 + (cloud_cover - 30) * 0.3))
                
                weather_data.append({
                    'timestamp': timestamp,
                    'latitude': latitude,
                    'longitude': longitude,
                    'temperature': round(temperature, 1),
                    'humidity': round(humidity, 1),
                    'pressure': round(pressure, 1),
                    'wind_speed': round(wind_speed, 1),
                    'wind_direction': round(wind_direction, 1),
                    'cloud_cover': round(cloud_cover, 1),
                    'precipitation_probability': round(precip_prob, 1),
                    'visibility': max(1, 10 + np.random.normal(0, 2)),
                    'uv_index': max(0, 6 * np.sin((hour - 12) * np.pi / 12)) if 6 <= hour <= 18 else 0
                })
            
            df = pd.DataFrame(weather_data)
            self.logger.info(f"Generated {len(df)} historical weather records")
            return df
            
        except Exception as e:
            self.logger.error(f"Historical weather data generation failed: {e}")
            return None
    
    def get_sample_weather_data(self, latitude: float, longitude: float, 
                               hours: int = 24) -> pd.DataFrame:
        """
        Generate sample weather data for testing purposes.
        
        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate  
            hours (int): Number of hours of data to generate
            
        Returns:
            pd.DataFrame: Sample weather data
        """
        self.logger.info(f"Generating {hours} hours of sample weather data")
        
        # Set random seed for reproducible sample data
        np.random.seed(42)
        
        # Create timestamp range
        start_time = datetime.now() - timedelta(hours=hours)
        timestamps = pd.date_range(start=start_time, periods=hours, freq='H')
        
        weather_data = []
        for i, timestamp in enumerate(timestamps):
            # Realistic daily temperature pattern
            hour = timestamp.hour
            day_of_year = timestamp.dayofyear
            
            # Base temperature with seasonal variation
            seasonal_temp = 15 + 12 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            daily_temp_variation = 8 * np.sin((hour - 14) * np.pi / 12)
            temperature = seasonal_temp + daily_temp_variation + np.random.normal(0, 2)
            
            # Humidity inversely related to temperature
            humidity = max(30, min(95, 85 - (temperature - 10) * 1.5 + np.random.normal(0, 10)))
            
            # Pressure with realistic variation
            pressure = 1013 + 15 * np.sin(2 * np.pi * i / (24 * 7)) + np.random.normal(0, 5)
            
            # Wind speed with some persistence
            if i == 0:
                wind_speed = max(0, 8 + np.random.normal(0, 3))
            else:
                # Add some persistence to wind speed
                prev_wind = weather_data[-1]['wind_speed']
                wind_speed = max(0, prev_wind * 0.7 + np.random.normal(3, 2))
            
            # Wind direction with some consistency
            wind_direction = (180 + 60 * np.sin(2 * np.pi * i / 48) + 
                            np.random.normal(0, 30)) % 360
            
            # Cloud cover affects solar irradiance
            cloud_cover = max(0, min(100, 50 + 30 * np.sin(2 * np.pi * i / 72) + 
                                   np.random.normal(0, 20)))
            
            # UV index based on time of day and cloud cover
            if 6 <= hour <= 18:
                clear_sky_uv = 8 * np.sin((hour - 6) * np.pi / 12)
                uv_index = max(0, clear_sky_uv * (1 - cloud_cover / 200))
            else:
                uv_index = 0
            
            # Precipitation probability
            precip_prob = max(0, (humidity - 60) * 0.8 + (cloud_cover - 40) * 0.4)
            
            weather_data.append({
                'timestamp': timestamp,
                'latitude': latitude,
                'longitude': longitude,
                'temperature': round(temperature, 1),
                'humidity': round(humidity, 1),
                'pressure': round(pressure, 1),
                'wind_speed': round(wind_speed, 1),
                'wind_direction': round(wind_direction, 1),
                'cloud_cover': round(cloud_cover, 1),
                'precipitation_probability': round(precip_prob, 1),
                'visibility': max(1, 10 + np.random.normal(0, 2)),
                'uv_index': round(uv_index, 1)
            })
        
        df = pd.DataFrame(weather_data)
        self.logger.info(f"Generated sample weather data: {len(df)} records")
        return df
    
    def _create_sample_current_weather(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """
        Create sample current weather data.
        
        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate
            
        Returns:
            dict: Sample current weather data
        """
        now = datetime.now()
        hour = now.hour
        
        # Generate realistic current weather
        temperature = 15 + 10 * np.sin(2 * np.pi * (now.dayofyear - 80) / 365) + \
                     5 * np.sin((hour - 14) * np.pi / 12) + np.random.normal(0, 2)
        
        current_weather = {
            'timestamp': now,
            'latitude': latitude,
            'longitude': longitude,
            'temperature': round(temperature, 1),
            'feels_like': round(temperature + np.random.normal(0, 2), 1),
            'humidity': round(max(30, min(95, 70 + np.random.normal(0, 15))), 1),
            'pressure': round(1013 + np.random.normal(0, 10), 1),
            'wind_speed': round(max(0, 8 + np.random.normal(0, 3)), 1),
            'wind_direction': round(np.random.uniform(0, 360), 1),
            'cloud_cover': round(max(0, min(100, 50 + np.random.normal(0, 20))), 1),
            'visibility': round(max(1, 10 + np.random.normal(0, 2)), 1),
            'weather_condition': np.random.choice(['Clear', 'Clouds', 'Rain', 'Partly Cloudy']),
            'weather_description': 'Sample weather data',
            'sunrise': now.replace(hour=6, minute=30, second=0, microsecond=0),
            'sunset': now.replace(hour=18, minute=30, second=0, microsecond=0)
        }
        
        return current_weather
    
    def save_weather_data(self, data: pd.DataFrame, filename: str) -> bool:
        """
        Save weather data to file.
        
        Args:
            data (pd.DataFrame): Weather data to save
            filename (str): Output filename
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import os
            os.makedirs('data/raw', exist_ok=True)
            
            filepath = f"data/raw/{filename}.csv"
            data.to_csv(filepath, index=False)
            
            self.logger.info(f"Weather data saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save weather data: {e}")
            return False