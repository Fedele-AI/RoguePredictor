#!/usr/bin/env python3
"""
Comprehensive tests for data_loader.py
Tests data loading, preprocessing,            # Should have derived features
        expected_features = ['wave_steepness', 'wind_wave_ratio']
        for feature in expected_features:
            self.assertIn(feature, data_with_features.columns)# Should have derived features
        expected_features = ['wave_steepness', 'wind_wave_ratio', 'is_summer', 'is_winter']
        for feature in expected_features:
            self.assertIn(feature, data_with_features.columns) all utility methods
"""

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
from pathlib import Path

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rouge_wave_predictor.data_loader import WaveDataLoader

class TestWaveDataLoader(unittest.TestCase):
    """Test cases for WaveDataLoader class"""

    def setUp(self):
        """Set up test fixtures"""
        self.loader = WaveDataLoader()
        self.sample_data = self.loader._create_sample_data(50)

    def test_initialization(self):
        """Test WaveDataLoader initialization"""
        self.assertIsInstance(self.loader, WaveDataLoader)
        self.assertIsNotNone(self.loader.logger)

    def test_create_sample_data(self):
        """Test sample data creation"""
        n_samples = 25
        data = self.loader._create_sample_data(n_samples)

        self.assertEqual(len(data), n_samples)
        self.assertIn('timestamp', data.columns)
        self.assertIn('latitude', data.columns)
        self.assertIn('longitude', data.columns)
        self.assertIn('wave_height', data.columns)
        self.assertIn('wave_period', data.columns)
        self.assertIn('wind_speed', data.columns)
        self.assertIn('wind_direction', data.columns)

        # Check data types
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(data['timestamp']))
        self.assertTrue(pd.api.types.is_numeric_dtype(data['wave_height']))
        self.assertTrue(pd.api.types.is_numeric_dtype(data['latitude']))

    def test_preprocess_data(self):
        """Test data preprocessing"""
        processed = self.loader.preprocess_data(self.sample_data)

        # Should have more columns after preprocessing
        self.assertGreater(len(processed.columns), len(self.sample_data.columns))

        # Should not have missing values
        self.assertFalse(processed.isnull().any().any())

        # Should have derived features
        expected_features = ['wave_steepness', 'wind_wave_ratio', 'ocean_basin', 'hour', 'day_of_week']
        for feature in expected_features:
            self.assertIn(feature, processed.columns)

    def test_ensure_required_columns(self):
        """Test ensuring required columns exist"""
        # Test with missing columns
        incomplete_data = self.sample_data.drop(['wave_height'], axis=1)
        completed_data = self.loader._ensure_required_columns(incomplete_data)

        self.assertIn('wave_height', completed_data.columns)
        self.assertEqual(len(completed_data), len(incomplete_data))

    def test_handle_missing_values(self):
        """Test missing value handling"""
        # Create data with missing values
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0:5, 'wave_height'] = np.nan

        filled_data = self.loader._handle_missing_values(data_with_missing)

        # Should not have NaN values
        self.assertFalse(filled_data['wave_height'].isnull().any())

    def test_normalize_features(self):
        """Test feature normalization"""
        normalized = self.loader._normalize_features(self.sample_data)

        # Check that normalized columns are created and have values between 0 and 1
        normalized_cols = ['wave_height_normalized', 'wave_period_normalized', 'wind_speed_normalized', 'wind_direction_normalized']
        for col in normalized_cols:
            if col in normalized.columns:
                self.assertTrue((normalized[col] >= 0).all())
                self.assertTrue((normalized[col] <= 1).all())

    def test_add_derived_features(self):
        """Test adding derived features"""
        data_with_features = self.loader._add_derived_features(self.sample_data)

        # Should have new features
        expected_features = ['wave_steepness', 'wind_wave_ratio']
        for feature in expected_features:
            self.assertIn(feature, data_with_features.columns)

        # Wave steepness should be positive
        self.assertTrue((data_with_features['wave_steepness'] > 0).all())

    def test_create_geospatial_features(self):
        """Test geospatial feature creation"""
        data_with_geo = self.loader._create_geospatial_features(self.sample_data)

        # Should have geospatial features
        self.assertIn('ocean_basin', data_with_geo.columns)
        self.assertIn('lat_grid', data_with_geo.columns)
        self.assertIn('lon_grid', data_with_geo.columns)
        self.assertIn('distance_from_equator', data_with_geo.columns)

        # Ocean basin should be string
        self.assertTrue(data_with_geo['ocean_basin'].dtype == 'object')

    def test_classify_ocean_basin(self):
        """Test ocean basin classification"""
        # Test different longitudes
        test_cases = [
            (-170, 'Pacific'),
            (-100, 'Pacific'),
            (-30, 'Atlantic'),
            (10, 'Atlantic'),
            (50, 'Indian'),
            (150, 'Indian')
        ]

        for longitude, expected_basin in test_cases:
            basin = self.loader._classify_ocean_basin(longitude)
            self.assertEqual(basin, expected_basin)

    def test_load_data_from_csv(self):
        """Test loading data from CSV file"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            loaded_data = self.loader.load_data(temp_file, max_samples=10)
            self.assertEqual(len(loaded_data), 10)
            self.assertIn('timestamp', loaded_data.columns)
        finally:
            os.unlink(temp_file)

    def test_load_data_max_samples(self):
        """Test max_samples parameter"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            loaded_data = self.loader.load_data(temp_file, max_samples=5)
            self.assertEqual(len(loaded_data), 5)
        finally:
            os.unlink(temp_file)

if __name__ == '__main__':
    unittest.main()
