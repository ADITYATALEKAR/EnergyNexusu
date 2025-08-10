"""
Base Data Collector Abstract Class
EnergyNexus MSc Project

This module defines the abstract base class for all energy data collectors, ensuring
consistent interfaces and standardized behavior across different data sources.

Author: Aditya Talekar (ec24018@qmul.ac.uk)
Supervisor: Saqib Iqbal
QMUL MSc Data Science and AI - 2024/25
"""

import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import json
import os
from typing import Optional, Dict, Any, List


class BaseDataCollector(ABC):
    """
    Abstract base class for all energy data collectors.
    
    This class defines the standard interface that all data collectors must implement,
    ensuring consistent behavior across different data sources and APIs.
    """
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the base data collector.
        
        Args:
            api_key (str, optional): API key for the data source
            **kwargs: Additional configuration parameters
        """
        self.api_key = api_key
        self.config = kwargs
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # I set up basic configuration
        self.timeout = kwargs.get('timeout', 30)
        self.max_retries = kwargs.get('max_retries', 3)
        self.retry_delay = kwargs.get('retry_delay', 1)
        
        # I initialize session for connection pooling
        self._session = None
        
        self.logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def validate_connection(self) -> bool:
        """
        Validate the connection to the data source.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_data(self, start_date: datetime, end_date: datetime, **params) -> Optional[pd.DataFrame]:
        """
        Retrieve historical data from the source.
        
        Args:
            start_date (datetime): Start date for data collection
            end_date (datetime): End date for data collection
            **params: Additional parameters specific to the data source
            
        Returns:
            pd.DataFrame: Retrieved data or None if failed
        """
        pass
    
    def get_current_data(self, **params) -> Optional[Dict[str, Any]]:
        """
        Retrieve current/real-time data from the source.
        
        Args:
            **params: Parameters specific to the data source
            
        Returns:
            dict: Current data or None if failed
        """
        self.logger.warning("get_current_data not implemented for this collector")
        return None
    
    def save_data(self, data: pd.DataFrame, filepath: str, include_metadata: bool = True) -> bool:
        """
        Save collected data to file with optional metadata.
        
        Args:
            data (pd.DataFrame): Data to save
            filepath (str): Path where to save the data
            include_metadata (bool): Whether to save metadata alongside data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # I ensure the directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # I save the data
            if filepath.endswith('.csv'):
                data.to_csv(filepath, index=False)
            elif filepath.endswith('.json'):
                data.to_json(filepath, orient='records', date_format='iso')
            else:
                # Default to CSV
                data.to_csv(filepath, index=False)
            
            self.logger.info(f"Data saved to {filepath}")
            
            # I save metadata if requested
            if include_metadata:
                metadata = self._create_metadata(data, filepath)
                metadata_path = filepath.replace('.csv', '_metadata.json').replace('.json', '_metadata.json')
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
                self.logger.info(f"Metadata saved to {metadata_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save data to {filepath}: {e}")
            return False
    
    def get_data_quality_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess data completeness and quality.
        
        Args:
            data (pd.DataFrame): Data to assess
            
        Returns:
            dict: Quality assessment report
        """
        if data is None or data.empty:
            return {
                'status': 'error',
                'message': 'No data provided for quality assessment'
            }
        
        # I calculate basic quality metrics
        total_rows = len(data)
        total_columns = len(data.columns)
        missing_values = data.isnull().sum().sum()
        missing_percentage = (missing_values / (total_rows * total_columns)) * 100
        
        # I check for duplicate rows
        duplicate_rows = data.duplicated().sum()
        duplicate_percentage = (duplicate_rows / total_rows) * 100
        
        # I analyze data types
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        datetime_columns = data.select_dtypes(include=['datetime64']).columns.tolist()
        
        # I calculate completeness by column
        column_completeness = {}
        for col in data.columns:
            non_null_count = data[col].count()
            completeness = (non_null_count / total_rows) * 100
            column_completeness[col] = completeness
        
        quality_score = max(0, 100 - missing_percentage - duplicate_percentage)
        
        return {
            'status': 'success',
            'overall_quality_score': quality_score,
            'total_rows': total_rows,
            'total_columns': total_columns,
            'missing_values': {
                'count': missing_values,
                'percentage': missing_percentage
            },
            'duplicate_rows': {
                'count': duplicate_rows,
                'percentage': duplicate_percentage
            },
            'column_completeness': column_completeness,
            'data_types': {
                'numeric_columns': numeric_columns,
                'datetime_columns': datetime_columns
            },
            'assessment_timestamp': datetime.now().isoformat()
        }
    
    def _create_metadata(self, data: pd.DataFrame, filepath: str) -> Dict[str, Any]:
        """
        Create metadata for the collected data.
        
        Args:
            data (pd.DataFrame): The collected data
            filepath (str): Path where data is saved
            
        Returns:
            dict: Metadata dictionary
        """
        return {
            'collector_type': self.__class__.__name__,
            'collection_timestamp': datetime.now().isoformat(),
            'data_source': getattr(self, 'data_source_name', 'Unknown'),
            'file_path': filepath,
            'data_shape': {
                'rows': len(data),
                'columns': len(data.columns)
            },
            'columns': data.columns.tolist(),
            'data_types': data.dtypes.astype(str).to_dict(),
            'date_range': {
                'start': str(data.index.min()) if hasattr(data.index, 'min') else None,
                'end': str(data.index.max()) if hasattr(data.index, 'max') else None
            } if not data.empty else None,
            'quality_report': self.get_data_quality_report(data),
            'collection_config': {
                'api_key_used': self.api_key is not None,
                'timeout': self.timeout,
                'max_retries': self.max_retries
            }
        }
    
    def create_sample_data(self, hours: int = 24*7, **params) -> pd.DataFrame:
        """
        Create sample/synthetic data for testing purposes.
        
        Args:
            hours (int): Number of hours of data to generate
            **params: Additional parameters for sample data generation
            
        Returns:
            pd.DataFrame: Sample data
        """
        self.logger.info(f"Creating sample data for {hours} hours")
        
        # I create a basic time series
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=hours),
            periods=hours,
            freq='H'
        )
        
        # I create basic sample data structure
        sample_data = pd.DataFrame({
            'timestamp': timestamps,
            'value': np.random.normal(100, 20, hours),
            'quality_flag': np.random.choice(['good', 'fair', 'poor'], hours, p=[0.8, 0.15, 0.05])
        })
        
        self.logger.info(f"Sample data created with shape: {sample_data.shape}")
        return sample_data
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._session:
            self._session.close()
            self.logger.info("Session closed")
    
    def __repr__(self):
        """String representation of the collector."""
        return f"{self.__class__.__name__}(api_key={'***' if self.api_key else None})"