#!/usr/bin/env python3
"""
System test script for Rouge Wave Analysis
Tests basic functionality and imports
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing module imports...")
    
    try:
        # Test core modules
        from data_loader import WaveDataLoader
        print("✓ data_loader imported successfully")
        
        from model_handler import GeospatialModelHandler
        print("✓ model_handler imported successfully")
        
        from predictor import WavePredictor
        print("✓ predictor imported successfully")
        
        from visualizer import WaveVisualizer
        print("✓ visualizer imported successfully")
        
        from utils import setup_logging, check_dependencies
        print("✓ utils imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during import: {e}")
        return False

def test_data_loader():
    """Test data loader functionality"""
    print("\nTesting data loader...")
    
    try:
        from data_loader import WaveDataLoader
        
        loader = WaveDataLoader()
        print("✓ WaveDataLoader initialized")
        
        # Test sample data creation
        sample_data = loader._create_sample_data(10)
        print(f"✓ Created sample data with {len(sample_data)} records")
        
        # Test preprocessing
        processed_data = loader.preprocess_data(sample_data)
        print(f"✓ Data preprocessing successful, {processed_data.shape[1]} features")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return False

def test_model_handler():
    """Test model handler functionality"""
    print("\nTesting model handler...")
    
    try:
        from model_handler import GeospatialModelHandler
        
        handler = GeospatialModelHandler()
        print("✓ GeospatialModelHandler initialized")
        
        # Test model info
        info = handler.get_model_info()
        print(f"✓ Model info retrieved: {info['model_name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model handler test failed: {e}")
        return False

def test_predictor():
    """Test predictor functionality"""
    print("\nTesting predictor...")
    
    try:
        from predictor import WavePredictor
        
        # Test with None model (fallback mode)
        predictor = WavePredictor(None, None)
        print("✓ WavePredictor initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Predictor test failed: {e}")
        return False

def test_visualizer():
    """Test visualizer functionality"""
    print("\nTesting visualizer...")
    
    try:
        from visualizer import WaveVisualizer
        
        visualizer = WaveVisualizer()
        print("✓ WaveVisualizer initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualizer test failed: {e}")
        return False

def test_utils():
    """Test utility functions"""
    print("\nTesting utility functions...")
    
    try:
        from utils import check_dependencies, print_dependency_status
        
        # Test dependency checking
        deps = check_dependencies()
        print(f"✓ Dependency check completed, {len(deps)} dependencies checked")
        
        return True
        
    except Exception as e:
        print(f"❌ Utils test failed: {e}")
        return False

def run_all_tests():
    """Run all system tests"""
    print("=" * 50)
    print("ROUGE WAVE ANALYSIS SYSTEM TEST")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Data Loader", test_data_loader),
        ("Model Handler", test_model_handler),
        ("Predictor", test_predictor),
        ("Visualizer", test_visualizer),
        ("Utilities", test_utils)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} : {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 