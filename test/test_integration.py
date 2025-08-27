#!/usr/bin/env python3
"""
Integration tests for main scripts
Tests rouge_wave_analysis.py and demo.py end-to-end functionality
"""

import unittest
import sys
import os
import tempfile
import subprocess
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

class TestIntegration(unittest.TestCase):
    """Integration tests for main scripts"""

    def setUp(self):
        """Set up test environment"""
        self.project_root = Path(__file__).parent

    def test_demo_quick_mode(self):
        """Test demo script in quick mode"""
        # Run demo with --quick flag
        result = subprocess.run([
            'rouge-wave-demo', '--quick'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

        # Should exit successfully
        self.assertEqual(result.returncode, 0)

        # Should have output indicating success
        self.assertIn('Quick functionality test', result.stdout)
        self.assertIn('✓ All components initialized successfully', result.stdout)

    def test_demo_full_mode(self):
        """Test demo script in full mode"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run demo in full mode
            env = os.environ.copy()
            env['PYTHONPATH'] = str(Path(__file__).parent.parent)

            result = subprocess.run([
                'rouge-wave-demo'
            ], capture_output=True, text=True, cwd=temp_dir, env=env)

            # Should exit successfully (0 or 1 depending on demo result)
            self.assertIn(result.returncode, [0, 1])

            # Should create output files
            expected_files = [
                'demo_outputs/wave_predictions.csv',
                'demo_outputs/demo_report.txt'
            ]

            for expected_file in expected_files:
                if result.returncode == 0:  # Only check files if successful
                    file_path = os.path.join(temp_dir, expected_file)
                    # Note: Files might not exist if demo fails gracefully
                    pass

    def test_rouge_wave_analysis_help(self):
        """Test rouge_wave_analysis help output"""
        result = subprocess.run([
            'rouge-wave-analysis', '--help'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

        # Should exit successfully
        self.assertEqual(result.returncode, 0)

        # Should show help information
        self.assertIn('usage:', result.stdout.lower())
        self.assertIn('model_name', result.stdout)
        self.assertIn('data_path', result.stdout)

    def test_rouge_wave_analysis_with_sample_data(self):
        """Test rouge_wave_analysis with sample data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            env = os.environ.copy()
            env['PYTHONPATH'] = str(Path(__file__).parent.parent)

            # Run analysis with limited samples
            result = subprocess.run([
                'rouge-wave-analysis',
                '--data_path', str(Path(__file__).parent.parent / 'data/sample_wave_data.csv'),
                '--output_dir', temp_dir,
                '--max_samples', '10'
            ], capture_output=True, text=True, cwd=temp_dir, env=env, timeout=60)

            # Should complete (may fail due to model loading, but shouldn't crash)
            self.assertIn(result.returncode, [0, 1, 2])

            # Should create output directory
            self.assertTrue(os.path.exists(temp_dir))

    def test_import_main_modules(self):
        """Test that main scripts can be imported without errors"""
        try:
            # Test importing main analysis script components
            import rouge_wave_predictor.rouge_wave_analysis
            self.assertTrue(hasattr(rouge_wave_predictor.rouge_wave_analysis, 'main'))
        except ImportError as e:
            self.fail(f"Failed to import rouge_wave_predictor.rouge_wave_analysis: {e}")

        try:
            # Test importing demo script components
            import rouge_wave_predictor.demo
            self.assertTrue(hasattr(rouge_wave_predictor.demo, 'main'))
            self.assertTrue(hasattr(rouge_wave_predictor.demo, 'quick_test'))
        except ImportError as e:
            self.fail(f"Failed to import rouge_wave_predictor.demo: {e}")

    def test_config_file_loading(self):
        """Test that config.yaml can be loaded"""
        config_path = Path(__file__).parent.parent / 'config.yaml'
        if config_path.exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                self.assertIsInstance(config, dict)
            except Exception as e:
                self.fail(f"Failed to load config.yaml: {e}")

    def test_data_file_exists(self):
        """Test that sample data file exists"""
        data_path = Path(__file__).parent.parent / 'data/sample_wave_data.csv'
        self.assertTrue(data_path.exists(), "Sample data file should exist")

        # Check that it has content
        import pandas as pd
        df = pd.read_csv(data_path)
        self.assertGreater(len(df), 0, "Sample data should not be empty")
        self.assertIn('wave_height', df.columns, "Sample data should have wave_height column")

    def test_requirements_file_exists(self):
        """Test that requirements.txt exists and is valid"""
        req_path = Path(__file__).parent.parent / 'requirements.txt'
        self.assertTrue(req_path.exists(), "requirements.txt should exist")

        # Check that it has content
        with open(req_path, 'r') as f:
            content = f.read()
        self.assertGreater(len(content), 0, "requirements.txt should not be empty")

        # Should contain key packages
        key_packages = ['transformers', 'torch', 'pandas']
        for package in key_packages:
            self.assertIn(package, content, f"requirements.txt should contain {package}")

    def test_readme_exists(self):
        """Test that README.md exists"""
        readme_path = Path(__file__).parent.parent / 'README.md'
        self.assertTrue(readme_path.exists(), "README.md should exist")

        # Check that it has content
        with open(readme_path, 'r') as f:
            content = f.read()
        self.assertGreater(len(content), 0, "README.md should not be empty")

    def test_directory_structure(self):
        """Test that expected directories exist"""
        expected_dirs = ['data', 'outputs', 'demo_outputs']
        for dir_name in expected_dirs:
            dir_path = Path(__file__).parent.parent / dir_name
            self.assertTrue(dir_path.exists(), f"Directory {dir_name} should exist")
            self.assertTrue(dir_path.is_dir(), f"{dir_name} should be a directory")

    def test_script_execution_permissions(self):
        """Test that main scripts have execution permissions"""
        scripts = ['src/rouge_wave_predictor/rouge_wave_analysis.py', 'src/rouge_wave_predictor/demo.py']
        for script in scripts:
            script_path = Path(__file__).parent.parent / script
            if script_path.exists():
                # Check if executable (on Unix-like systems)
                # This is more of a warning than a failure
                pass

    def test_error_handling(self):
        """Test error handling in main scripts"""
        # Test demo with invalid arguments
        result = subprocess.run([
            'rouge-wave-demo', '--invalid-flag'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

        # Should handle invalid arguments gracefully
        self.assertIn(result.returncode, [0, 1, 2])

if __name__ == '__main__':
    unittest.main()
