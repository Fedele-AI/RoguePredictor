"""
Data loading and preprocessing for rouge wave analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import warnings
import os

class WaveDataLoader:
    """Handles loading and preprocessing of wave data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.required_columns = [
            'timestamp', 'latitude', 'longitude', 
            'wave_height', 'wave_period', 'wind_speed', 'wind_direction'
        ]
    
    def load_data(self, data_path: str, max_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Load wave data from CSV file
        
        Args:
            data_path: Path to the CSV file
            max_samples: Maximum number of samples to load
            
        Returns:
            DataFrame containing wave data
        """
        try:
            if not Path(data_path).exists():
                self.logger.warning(f"Data file {data_path} not found. Creating sample data.")
                return self._create_sample_data(max_samples or 1000)
            
            self.logger.info(f"Loading data from {data_path}")
            data = pd.read_csv(data_path)
            
            if max_samples and len(data) > max_samples:
                data = data.sample(n=max_samples, random_state=42)
                self.logger.info(f"Sampled {max_samples} records from dataset")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            self.logger.info("Creating sample data instead")
            return self._create_sample_data(max_samples or 1000)
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess wave data for model input
        
        Args:
            data: Raw wave data DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        self.logger.info("Preprocessing wave data")
        
        # Create a copy to avoid modifying original data
        processed = data.copy()
        
        # Ensure required columns exist
        processed = self._ensure_required_columns(processed)
        
        # Convert timestamp to datetime
        processed['timestamp'] = pd.to_datetime(processed['timestamp'])
        
        # Extract time-based features
        processed['hour'] = processed['timestamp'].dt.hour
        processed['day_of_week'] = processed['timestamp'].dt.dayofweek
        processed['month'] = processed['timestamp'].dt.month
        
        # Handle missing values
        processed = self._handle_missing_values(processed)
        
        # Normalize numerical features
        processed = self._normalize_features(processed)
        
        # Add derived features
        processed = self._add_derived_features(processed)
        
        # Create geospatial features
        processed = self._create_geospatial_features(processed)
        
        self.logger.info(f"Preprocessing complete. Final shape: {processed.shape}")
        return processed
    
    def _ensure_required_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required columns exist, create if missing"""
        for col in self.required_columns:
            if col not in data.columns:
                if col == 'timestamp':
                    data[col] = pd.date_range(start='2023-01-01', periods=len(data), freq='H')
                elif col in ['latitude', 'longitude']:
                    data[col] = np.random.uniform(-90, 90, len(data)) if col == 'latitude' else np.random.uniform(-180, 180, len(data))
                elif col == 'wave_height':
                    data[col] = np.random.exponential(2, len(data))  # Exponential distribution for wave heights
                elif col == 'wave_period':
                    data[col] = np.random.normal(8, 2, len(data))  # Normal distribution for wave periods
                elif col == 'wind_speed':
                    data[col] = np.random.weibull(2, len(data))  # Weibull distribution for wind speeds
                elif col == 'wind_direction':
                    data[col] = np.random.uniform(0, 360, len(data))  # Uniform distribution for wind direction
                
                self.logger.info(f"Created missing column: {col}")
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Fill numerical columns with median
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if data[col].isnull().any():
                median_val = data[col].median()
                data[col].fillna(median_val, inplace=True)
                self.logger.info(f"Filled missing values in {col} with median: {median_val}")
        
        # Fill categorical columns with mode
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if data[col].isnull().any():
                mode_val = data[col].mode()[0] if not data[col].mode().empty else 'Unknown'
                data[col].fillna(mode_val, inplace=True)
                self.logger.info(f"Filled missing values in {col} with mode: {mode_val}")
        
        return data
    
    def _normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features to [0, 1] range"""
        numerical_cols = ['wave_height', 'wave_period', 'wind_speed', 'wind_direction']
        
        for col in numerical_cols:
            if col in data.columns:
                min_val = data[col].min()
                max_val = data[col].max()
                if max_val > min_val:
                    data[f'{col}_normalized'] = (data[col] - min_val) / (max_val - min_val)
                else:
                    data[f'{col}_normalized'] = 0.5
        
        return data
    
    def _add_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add derived features that might be useful for prediction"""
        # Wave steepness (wave height / wave period)
        if 'wave_height' in data.columns and 'wave_period' in data.columns:
            data['wave_steepness'] = data['wave_height'] / (data['wave_period'] + 1e-8)
        
        # Wind-wave interaction
        if 'wind_speed' in data.columns and 'wave_height' in data.columns:
            data['wind_wave_ratio'] = data['wind_speed'] / (data['wave_height'] + 1e-8)
        
        # Seasonal features
        if 'month' in data.columns:
            data['is_summer'] = data['month'].isin([6, 7, 8]).astype(int)
            data['is_winter'] = data['month'].isin([12, 1, 2]).astype(int)
        
        return data
    
    def _create_geospatial_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create geospatial features"""
        if 'latitude' in data.columns and 'longitude' in data.columns:
            # Convert to radians for calculations
            lat_rad = np.radians(data['latitude'])
            lon_rad = np.radians(data['longitude'])
            
            # Create grid-based features
            data['lat_grid'] = np.round(data['latitude'], 1)
            data['lon_grid'] = np.round(data['longitude'], 1)
            
            # Distance from equator
            data['distance_from_equator'] = np.abs(data['latitude'])
            
            # Ocean basin classification (simplified)
            data['ocean_basin'] = data['longitude'].apply(self._classify_ocean_basin)
        
        return data
    
    def _classify_ocean_basin(self, longitude: float) -> str:
        """Classify ocean basin based on longitude"""
        if -180 <= longitude < -60:
            return 'Pacific'
        elif -60 <= longitude < 20:
            return 'Atlantic'
        elif 20 <= longitude < 180:
            return 'Indian'
        else:
            return 'Unknown'
    
    def _create_sample_data(self, n_samples: int) -> pd.DataFrame:
        """Create sample wave data for testing"""
        self.logger.info(f"Creating sample dataset with {n_samples} records")
        
        np.random.seed(42)
        
        # Generate timestamps
        timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='h')
        
        # Generate realistic wave data
        data = {
            'timestamp': timestamps,
            'latitude': np.random.uniform(30, 60, n_samples),  # Mid-latitudes
            'longitude': np.random.uniform(-80, -40, n_samples),  # North Atlantic
            'wave_height': np.random.exponential(2, n_samples),  # Exponential distribution
            'wave_period': np.random.normal(8, 2, n_samples),   # Normal distribution
            'wind_speed': np.random.weibull(2, n_samples),      # Weibull distribution
            'wind_direction': np.random.uniform(0, 360, n_samples)
        }
        
        # Add some rouge wave events (unusually high waves)
        rouge_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        data['wave_height'][rouge_indices] *= np.random.uniform(2, 4, len(rouge_indices))
        
        df = pd.DataFrame(data)
        
        # Save sample data
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/sample_wave_data.csv', index=False)
        self.logger.info("Sample data saved to data/sample_wave_data.csv")
        
        return df 