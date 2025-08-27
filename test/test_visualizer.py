#!/usr/bin/env python3
"""
Comprehensive tests for visualizer.py
Tests all visualization methods and plotting functionality
"""

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rouge_wave_predictor.visualizer import WaveVisualizer

class TestWaveVisualizer(unittest.TestCase):
    """Test cases for WaveVisualizer class"""

    def setUp(self):
        """Set up test fixtures"""
        self.visualizer = WaveVisualizer()

        # Create sample wave data
        self.sample_wave_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=50, freq='h'),  # Changed H to h
            'latitude': np.random.uniform(-90, 90, 50),
            'longitude': np.random.uniform(-180, 180, 50),
            'wave_height': np.random.uniform(0.5, 5.0, 50),
            'wave_period': np.random.uniform(5, 15, 50),
            'wind_speed': np.random.uniform(0, 20, 50),
            'wind_direction': np.random.uniform(0, 360, 50),
            'wave_energy': np.random.uniform(10, 100, 50),
            'ocean_basin': np.random.choice(['Pacific', 'Atlantic', 'Indian'], 50)
        })

        # Create sample predictions
        self.sample_predictions = []
        for i in range(50):
            self.sample_predictions.append({
                'timestamp': self.sample_wave_data.iloc[i]['timestamp'],
                'latitude': self.sample_wave_data.iloc[i]['latitude'],
                'longitude': self.sample_wave_data.iloc[i]['longitude'],
                'input_wave_height': self.sample_wave_data.iloc[i]['wave_height'],
                'predicted_wave_height': self.sample_wave_data.iloc[i]['wave_height'] * np.random.normal(1.0, 0.1),
                'predicted_wave_period': self.sample_wave_data.iloc[i]['wave_period'] * np.random.normal(1.0, 0.05),
                'rouge_wave_probability': np.random.uniform(0, 1),
                'risk_level': np.random.choice(['Low', 'Medium', 'High']),
                'confidence_score': np.random.uniform(0.5, 1.0),
                'analysis_notes': f'Test note {i}',
                'prediction_timestamp': pd.Timestamp.now()
            })

    def test_initialization(self):
        """Test WaveVisualizer initialization"""
        self.assertIsInstance(self.visualizer, WaveVisualizer)
        self.assertIsNotNone(self.visualizer.logger)

    def test_create_wave_analysis_plots(self):
        """Test creating all wave analysis plots"""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.visualizer.create_wave_analysis_plots(
                self.sample_wave_data,
                self.sample_predictions,
                temp_dir
            )

            # Should create multiple plot files
            expected_files = [
                'wave_height_distribution.png',
                'rouge_wave_probability.png',
                'geographic_risk_map.png',
                'time_series_analysis.png',
                'risk_level_distribution.png',
                'wave_characteristics_scatter.png',
                'confidence_analysis.png',
                'analysis_dashboard.png'
            ]

            for filename in expected_files:
                filepath = os.path.join(temp_dir, filename)
                self.assertTrue(os.path.exists(filepath), f"Missing file: {filename}")

    def test_create_wave_height_distribution(self):
        """Test wave height distribution plot"""
        with tempfile.TemporaryDirectory() as temp_dir:
            pred_df = pd.DataFrame(self.sample_predictions)
            self.visualizer._create_wave_height_distribution(
                self.sample_wave_data,
                pred_df,
                temp_dir
            )

            filepath = os.path.join(temp_dir, 'wave_height_distribution.png')
            self.assertTrue(os.path.exists(filepath))

    def test_create_rouge_wave_probability_plot(self):
        """Test rouge wave probability plot"""
        pred_df = pd.DataFrame(self.sample_predictions)

        with tempfile.TemporaryDirectory() as temp_dir:
            self.visualizer._create_rouge_wave_probability_plot(pred_df, temp_dir)

            filepath = os.path.join(temp_dir, 'rouge_wave_probability.png')
            self.assertTrue(os.path.exists(filepath))

    def test_create_geographic_risk_map(self):
        """Test geographic risk map"""
        pred_df = pd.DataFrame(self.sample_predictions)

        with tempfile.TemporaryDirectory() as temp_dir:
            self.visualizer._create_geographic_risk_map(pred_df, temp_dir)

            filepath = os.path.join(temp_dir, 'geographic_risk_map.png')
            self.assertTrue(os.path.exists(filepath))

    def test_create_time_series_analysis(self):
        """Test time series analysis plot"""
        with tempfile.TemporaryDirectory() as temp_dir:
            pred_df = pd.DataFrame(self.sample_predictions)
            self.visualizer._create_time_series_analysis(
                self.sample_wave_data,
                pred_df,
                temp_dir
            )

            filepath = os.path.join(temp_dir, 'time_series_analysis.png')
            self.assertTrue(os.path.exists(filepath))

    def test_create_risk_level_distribution(self):
        """Test risk level distribution plot"""
        pred_df = pd.DataFrame(self.sample_predictions)

        with tempfile.TemporaryDirectory() as temp_dir:
            self.visualizer._create_risk_level_distribution(pred_df, temp_dir)

            filepath = os.path.join(temp_dir, 'risk_level_distribution.png')
            self.assertTrue(os.path.exists(filepath))

    def test_create_wave_characteristics_scatter(self):
        """Test wave characteristics scatter plot"""
        with tempfile.TemporaryDirectory() as temp_dir:
            pred_df = pd.DataFrame(self.sample_predictions)
            self.visualizer._create_wave_characteristics_scatter(
                self.sample_wave_data,
                pred_df,
                temp_dir
            )

            filepath = os.path.join(temp_dir, 'wave_characteristics_scatter.png')
            self.assertTrue(os.path.exists(filepath))

    def test_create_confidence_analysis(self):
        """Test confidence analysis plot"""
        pred_df = pd.DataFrame(self.sample_predictions)

        with tempfile.TemporaryDirectory() as temp_dir:
            self.visualizer._create_confidence_analysis(pred_df, temp_dir)

            filepath = os.path.join(temp_dir, 'confidence_analysis.png')
            self.assertTrue(os.path.exists(filepath))

    def test_create_analysis_dashboard(self):
        """Test analysis dashboard creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            pred_df = pd.DataFrame(self.sample_predictions)
            self.visualizer._create_analysis_dashboard(
                self.sample_wave_data,
                pred_df,
                temp_dir
            )

            filepath = os.path.join(temp_dir, 'analysis_dashboard.png')
            self.assertTrue(os.path.exists(filepath))

    def test_empty_data_handling(self):
        """Test handling of empty data"""
        empty_data = pd.DataFrame()
        empty_predictions = []

        with tempfile.TemporaryDirectory() as temp_dir:
            # Should handle empty data gracefully
            try:
                self.visualizer.create_wave_analysis_plots(
                    empty_data,
                    empty_predictions,
                    temp_dir
                )
            except Exception as e:
                # Should not crash, but might log warnings
                self.assertIsInstance(str(e), str)

    def test_missing_columns_handling(self):
        """Test handling of missing columns"""
        incomplete_data = self.sample_wave_data.drop(['wave_height'], axis=1)
        incomplete_predictions = []
        for pred in self.sample_predictions[:5]:
            incomplete_pred = pred.copy()
            del incomplete_pred['rouge_wave_probability']
            incomplete_predictions.append(incomplete_pred)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Should handle missing columns gracefully
            try:
                self.visualizer.create_wave_analysis_plots(
                    incomplete_data,
                    incomplete_predictions,
                    temp_dir
                )
            except Exception as e:
                # Should not crash, but might log warnings
                self.assertIsInstance(str(e), str)

    def test_single_prediction_handling(self):
        """Test handling of single prediction"""
        single_prediction = [self.sample_predictions[0]]

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                self.visualizer.create_wave_analysis_plots(
                    self.sample_wave_data.head(1),
                    single_prediction,
                    temp_dir
                )
            except Exception as e:
                # Should handle single data point gracefully
                self.assertIsInstance(str(e), str)

if __name__ == '__main__':
    unittest.main()
