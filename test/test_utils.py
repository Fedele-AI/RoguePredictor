#!/usr/bin/env python3
"""
Comprehensive tests for utils.py
Tests all utility functions and configuration management
"""

import unittest
import os
import tempfile
import json
import logging
from datetime import datetime
from unittest.mock import patch, MagicMock

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rouge_wave_predictor.utils import (
    setup_logging, create_output_dirs, load_config, get_default_config,
    save_config, validate_data_format, format_timestamp, calculate_statistics,
    create_progress_bar, check_dependencies, print_dependency_status,
    get_file_size_mb, cleanup_temp_files
)

class TestUtils(unittest.TestCase):
    """Test cases for utility functions"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_config = {
            'model_name': 'test-model',
            'batch_size': 16,
            'max_samples': 100,
            'output_dir': 'test_output'
        }

    def test_setup_logging(self):
        """Test logging setup"""
        # Reset logging configuration for clean test
        logging.shutdown()
        import importlib
        importlib.reload(logging)
        
        logger = setup_logging(level="DEBUG", log_file=None)

        self.assertIsInstance(logger, logging.Logger)
        # Check that the logger has the correct effective level
        self.assertEqual(logger.getEffectiveLevel(), logging.DEBUG)

        # Test with log file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            log_file = f.name

        try:
            # Reset logging again for second test
            logging.shutdown()
            import importlib
            importlib.reload(logging)
            
            logger_with_file = setup_logging(level="INFO", log_file=log_file)
            self.assertIsInstance(logger_with_file, logging.Logger)
            self.assertEqual(logger_with_file.getEffectiveLevel(), logging.INFO)
            self.assertTrue(os.path.exists(log_file))
        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)

    def test_create_output_dirs(self):
        """Test output directory creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, 'test_output')
            subdirs = ['plots', 'data', 'logs']

            result_dir = create_output_dirs(output_dir, subdirs)

            # Should return a timestamped subdirectory
            self.assertTrue(result_dir.startswith(output_dir))
            self.assertIn('run_', result_dir)
            self.assertTrue(os.path.exists(result_dir))

            for subdir in subdirs:
                subdir_path = os.path.join(output_dir, subdir)
                self.assertTrue(os.path.exists(subdir_path))

    def test_load_config(self):
        """Test configuration loading"""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(self.test_config, f)
            config_file = f.name

        try:
            loaded_config = load_config(config_file)

            self.assertEqual(loaded_config['model_name'], 'test-model')
            self.assertEqual(loaded_config['batch_size'], 16)
        finally:
            os.unlink(config_file)

    def test_get_default_config(self):
        """Test getting default configuration"""
        config = get_default_config()

        self.assertIsInstance(config, dict)
        self.assertIn('model', config)
        self.assertIn('data', config)
        self.assertIn('analysis', config)
        self.assertIn('output', config)
        self.assertEqual(config['model']['name'], 'ibm-nasa-geospatial/Prithvi-100M')

    def test_save_config(self):
        """Test configuration saving"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_file = f.name

        try:
            save_config(self.test_config, config_file)

            # Verify file was created and has content
            self.assertTrue(os.path.exists(config_file))

            with open(config_file, 'r') as f:
                content = f.read()
                self.assertIn('model_name', content)
                self.assertIn('test-model', content)
        finally:
            if os.path.exists(config_file):
                os.unlink(config_file)

    def test_validate_data_format(self):
        """Test data format validation"""
        # Valid data
        valid_data = {
            'name': 'test',
            'value': 123,
            'timestamp': '2023-01-01'
        }
        required_fields = ['name', 'value']

        self.assertTrue(validate_data_format(valid_data, required_fields))

        # Invalid data - missing field
        invalid_data = {'name': 'test'}
        self.assertFalse(validate_data_format(invalid_data, required_fields))

        # Invalid data - wrong type
        self.assertFalse(validate_data_format("not a dict", required_fields))

    def test_format_timestamp(self):
        """Test timestamp formatting"""
        # Test with datetime object
        dt = datetime(2023, 1, 1, 12, 30, 45)
        formatted = format_timestamp(dt)
        self.assertIn('2023', formatted)
        self.assertIn('12:30', formatted)

        # Test with string
        timestamp_str = "2023-01-01 12:30:45"
        formatted = format_timestamp(timestamp_str)
        self.assertEqual(formatted, timestamp_str)

        # Test with None - should return current time formatted
        formatted = format_timestamp(None)
        self.assertIsInstance(formatted, str)
        self.assertRegex(formatted, r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')

    def test_calculate_statistics(self):
        """Test statistics calculation"""
        data = [
            {'value': 10},
            {'value': 20},
            {'value': 30},
            {'value': 40},
            {'value': 50}
        ]

        stats = calculate_statistics(data, 'value')

        self.assertAlmostEqual(stats['mean'], 30.0)
        self.assertAlmostEqual(stats['std'], 14.142, places=3)  # numpy std vs manual calculation difference
        self.assertEqual(stats['min'], 10)
        self.assertEqual(stats['max'], 50)

    def test_calculate_statistics_empty_data(self):
        """Test statistics calculation with empty data"""
        stats = calculate_statistics([], 'value')

        self.assertEqual(stats['mean'], 0)
        self.assertEqual(stats['min'], 0)
        self.assertEqual(stats['max'], 0)
        self.assertEqual(stats['std'], 0)

    def test_create_progress_bar(self):
        """Test progress bar creation"""
        progress_bar = create_progress_bar(100, "Test Progress")

        # Should return a progress bar object with expected attributes
        self.assertTrue(hasattr(progress_bar, 'update'))
        self.assertTrue(hasattr(progress_bar, 'close'))

    @patch('importlib.util.find_spec')
    def test_check_dependencies(self, mock_find_spec):
        """Test dependency checking"""
        # Mock successful imports
        mock_find_spec.return_value = MagicMock()

        deps = check_dependencies()

        self.assertIsInstance(deps, dict)
        self.assertIn('transformers', deps)
        self.assertIn('torch', deps)

    def test_print_dependency_status(self):
        """Test dependency status printing"""
        # Should not raise any exceptions
        try:
            print_dependency_status()
        except Exception as e:
            self.fail(f"print_dependency_status raised an exception: {e}")

    def test_get_file_size_mb(self):
        """Test file size calculation"""
        # Create a test file
        test_content = "This is a test file with some content."
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(test_content)
            test_file = f.name

        try:
            size_mb = get_file_size_mb(test_file)
            self.assertIsInstance(size_mb, float)
            self.assertGreater(size_mb, 0)
        finally:
            os.unlink(test_file)

    def test_get_file_size_mb_nonexistent(self):
        """Test file size calculation for nonexistent file"""
        size_mb = get_file_size_mb("nonexistent_file.txt")
        self.assertEqual(size_mb, 0.0)

    def test_cleanup_temp_files(self):
        """Test temporary file cleanup"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some test files
            old_file = os.path.join(temp_dir, 'old_file.txt')
            new_file = os.path.join(temp_dir, 'new_file.txt')

            with open(old_file, 'w') as f:
                f.write('old content')
            with open(new_file, 'w') as f:
                f.write('new content')

            # Set old file modification time to be old
            import time
            old_time = time.time() - (25 * 60 * 60)  # 25 hours ago
            os.utime(old_file, (old_time, old_time))

            # Clean up files older than 24 hours
            cleanup_temp_files(temp_dir, max_age_hours=24)

            # Old file should be gone, new file should remain
            self.assertFalse(os.path.exists(old_file))
            self.assertTrue(os.path.exists(new_file))

    def test_edge_cases(self):
        """Test edge cases for utility functions"""
        # Test validate_data_format with None - should handle gracefully
        try:
            result = validate_data_format(None, ['field'])
            # If it doesn't raise an exception, it should return False
            self.assertFalse(result)
        except Exception:
            # If it raises an exception, that's also acceptable
            pass

        # Test validate_data_format with empty required fields
        self.assertTrue(validate_data_format({'field': 'value'}, []))

        # Test calculate_statistics with non-numeric values
        data = [{'value': 'not_a_number'}]
        stats = calculate_statistics(data, 'value')
        self.assertEqual(stats['mean'], 0)  # Should handle non-numeric gracefully

        # Test format_timestamp with invalid input
        formatted = format_timestamp(12345)
        self.assertIsInstance(formatted, str)

if __name__ == '__main__':
    unittest.main()
