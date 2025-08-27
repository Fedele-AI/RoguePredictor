#!/usr/bin/env python3
"""
Comprehensive tests for model_handler.py
Tests model loading, pipeline creation, and all utility methods
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rouge_wave_predictor.model_handler import GeospatialModelHandler

class TestGeospatialModelHandler(unittest.TestCase):
    """Test cases for GeospatialModelHandler class"""

    def setUp(self):
        """Set up test fixtures"""
        self.model_name = "test-model"
        self.handler = GeospatialModelHandler(self.model_name)

    def test_initialization(self):
        """Test GeospatialModelHandler initialization"""
        self.assertIsInstance(self.handler, GeospatialModelHandler)
        self.assertEqual(self.handler.model_name, self.model_name)
        self.assertIsNone(self.handler.model)
        self.assertIsNone(self.handler.tokenizer)

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    def test_load_specified_model_success(self, mock_model, mock_tokenizer):
        """Test loading a specified model successfully"""
        # Mock successful model loading
        mock_model_instance = Mock()
        mock_tokenizer_instance = Mock()
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value = mock_tokenizer_instance

        model, tokenizer = self.handler._load_specified_model()

        self.assertEqual(model, mock_model_instance)
        self.assertEqual(tokenizer, mock_tokenizer_instance)

        # Verify calls
        mock_model.assert_called_once_with(self.model_name, trust_remote_code=True, cache_dir='./model_cache')
        mock_tokenizer.assert_called_once_with(self.model_name, trust_remote_code=True, cache_dir='./model_cache')

    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_load_specified_model_failure(self, mock_model):
        """Test handling of model loading failure"""
        # Mock model loading failure
        mock_model.side_effect = Exception("Model not found")

        model, tokenizer = self.handler._load_specified_model()

        self.assertIsNone(model)
        self.assertIsNone(tokenizer)
        self.assertIsNone(self.handler.model)
        self.assertIsNone(self.handler.tokenizer)

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    def test_load_fallback_model(self, mock_model, mock_tokenizer):
        """Test loading fallback models"""
        # Mock successful fallback loading
        mock_model_instance = Mock()
        mock_tokenizer_instance = Mock()
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value = mock_tokenizer_instance

        model = self.handler._load_fallback_model()

        self.assertEqual(model, mock_model_instance)
        self.assertEqual(self.handler.model, mock_model_instance)
        self.assertEqual(self.handler.tokenizer, mock_tokenizer_instance)

    def test_load_model_with_fallback(self):
        """Test load_model with fallback enabled"""
        with patch.object(self.handler, '_load_specified_model') as mock_specified:
            with patch.object(self.handler, '_load_fallback_model') as mock_fallback:
                # Specified model fails, fallback succeeds
                mock_specified.return_value = None
                mock_fallback.return_value = Mock()

                model = self.handler.load_model(use_fallback=True)

                mock_specified.assert_called_once()
                mock_fallback.assert_called_once()
                self.assertIsNotNone(model)

    def test_load_model_without_fallback(self):
        """Test load_model without fallback"""
        with patch.object(self.handler, '_load_specified_model') as mock_specified:
            mock_specified.return_value = (None, None)

            model = self.handler.load_model(use_fallback=False)

            mock_specified.assert_called_once()
            self.assertIsNone(model)

    def test_get_model_info_no_model(self):
        """Test getting model info when no model is loaded"""
        info = self.handler.get_model_info()

        expected_keys = ['model_name', 'model_loaded', 'tokenizer_loaded', 'pipeline_available']
        for key in expected_keys:
            self.assertIn(key, info)
        
        self.assertEqual(info['model_name'], 'test-model')
        self.assertFalse(info['model_loaded'])
        self.assertFalse(info['tokenizer_loaded'])
        self.assertFalse(info['pipeline_available'])

    def test_get_model_info_with_model(self):
        """Test getting model info when model is loaded"""
        # Mock model with config
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.model_type = 'test_type'
        mock_model.config.n_parameters = 1000000
        # Mock the parameters method
        mock_model.parameters.return_value = [Mock(numel=lambda: 100) for _ in range(10)]

        mock_tokenizer = Mock()
        mock_tokenizer.__class__.__name__ = 'TestTokenizer'

        self.handler.model = mock_model
        self.handler.tokenizer = mock_tokenizer

        info = self.handler.get_model_info()

        self.assertEqual(info['model_name'], self.model_name)
        self.assertEqual(info['model_type'], 'Mock')
        self.assertEqual(info['model_parameters'], 1000)  # 10 params * 100 each
        self.assertEqual(info['tokenizer_type'], 'TestTokenizer')

    def test_prepare_input_for_wave_analysis(self):
        """Test preparing input for wave analysis"""
        wave_data = {
            'wave_height': 3.5,
            'wave_period': 12.0,
            'wind_speed': 15.0,
            'latitude': 45.0,
            'longitude': -125.0,
            'ocean_basin': 'Pacific'
        }

        input_text = self.handler.prepare_input_for_wave_analysis(wave_data)

        self.assertIsInstance(input_text, str)
        self.assertIn('3.5', input_text)
        self.assertIn('Pacific', input_text)
        self.assertIn('wave analysis', input_text.lower())

    def test_is_model_ready_no_model(self):
        """Test is_model_ready when no model is loaded"""
        self.assertFalse(self.handler.is_model_ready())

    def test_is_model_ready_with_model(self):
        """Test is_model_ready when model is loaded"""
        self.handler.model = Mock()
        self.handler.tokenizer = Mock()

        self.assertTrue(self.handler.is_model_ready())

    @patch('rouge_wave_predictor.model_handler.pipeline')
    def test_create_wave_analysis_pipeline(self, mock_pipeline):
        """Test creating wave analysis pipeline"""
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance

        self.handler.model = Mock()
        self.handler.tokenizer = Mock()

        pipeline = self.handler.create_wave_analysis_pipeline()

        self.assertEqual(pipeline, mock_pipeline_instance)
        mock_pipeline.assert_called_once()

    def test_create_wave_analysis_pipeline_no_model(self):
        """Test creating pipeline when no model is loaded"""
        pipeline = self.handler.create_wave_analysis_pipeline()

        self.assertIsNone(pipeline)

    def test_fallback_model_list(self):
        """Test that fallback models list is comprehensive"""
        # This tests the fallback_models list in the class
        # We can't directly access it, but we can test the behavior

        with patch.object(self.handler, '_load_specified_model') as mock_specified:
            with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
                with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                    # Make specified model fail
                    mock_specified.return_value = None

                    # Make fallback models succeed on first try
                    mock_model_instance = Mock()
                    mock_tokenizer_instance = Mock()
                    mock_model.return_value = mock_model_instance
                    mock_tokenizer.return_value = mock_tokenizer_instance

                    model = self.handler.load_model(use_fallback=True)

                    # Should have loaded a fallback model
                    self.assertIsNotNone(model)

    def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test with invalid model name
        handler = GeospatialModelHandler("invalid-model-name")

        with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
            mock_model.side_effect = Exception("Invalid model")

            model = handler.load_model(use_fallback=False)
            self.assertIsNone(model)

    def test_model_info_edge_cases(self):
        """Test model info with edge cases"""
        # Test with model that has no config
        mock_model = Mock()
        mock_model.config = None
        # Mock will raise an exception when parameters() is called
        mock_model.parameters.side_effect = Exception("No parameters")

        self.handler.model = mock_model
        self.handler.tokenizer = Mock()

        info = self.handler.get_model_info()

        self.assertEqual(info['model_type'], 'Unknown')
        self.assertEqual(info['model_parameters'], 'Unknown')

    def test_prepare_input_edge_cases(self):
        """Test input preparation with edge cases"""
        # Test with missing keys
        incomplete_data = {'wave_height': 3.0}
        input_text = self.handler.prepare_input_for_wave_analysis(incomplete_data)

        self.assertIsInstance(input_text, str)
        self.assertIn('3.0', input_text)

        # Test with empty data
        empty_data = {}
        input_text = self.handler.prepare_input_for_wave_analysis(empty_data)

        self.assertIsInstance(input_text, str)

if __name__ == '__main__':
    unittest.main()
