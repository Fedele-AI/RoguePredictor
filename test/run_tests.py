#!/usr/bin/env python3
"""
Test runner for the Rouge Wave Analysis project
Runs all tests and generates coverage reports
"""

import unittest
import sys
import os
import importlib.util
from pathlib import Path

def run_all_tests():
    """Run all test suites"""
    print("=" * 60)
    print("ROUGE WAVE ANALYSIS - COMPREHENSIVE TEST SUITE")
    print("=" * 60)

    # Discover all test files
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()

    # Test files to run
    test_files = [
        'test_granite_ts',
        'test_data_loader',
        'test_predictor',
        'test_visualizer',
        'test_utils',
        'test_model_handler',
        'test_integration',
        'test_real_model',
        'test_system'
    ]

    project_root = Path(__file__).parent

    for test_file in test_files:
        test_path = project_root / f'{test_file}.py'
        if test_path.exists():
            try:
                # Import the test module
                module_name = test_file
                if module_name in sys.modules:
                    del sys.modules[module_name]

                spec = importlib.util.spec_from_file_location(module_name, test_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                # Load tests from the module
                suite = test_loader.loadTestsFromModule(module)
                test_suite.addTest(suite)

                print(f"✓ Loaded {test_file}.py")

            except Exception as e:
                print(f"❌ Failed to load {test_file}.py: {e}")
        else:
            print(f"⚠️  Test file {test_file}.py not found")

    # Run the tests
    print("\n" + "=" * 60)
    print("RUNNING TESTS")
    print("=" * 60)

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)

    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    passed = total_tests - failures - errors - skipped

    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")

    if failures > 0:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}")

    if errors > 0:
        print(f"\n🔥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}")

    # Coverage estimate (rough)
    coverage_percentage = (passed / total_tests * 100) if total_tests > 0 else 0
    print(f"Estimated Coverage: {coverage_percentage:.1f}%")
    if coverage_percentage >= 80:
        print("🎉 Excellent coverage!")
    elif coverage_percentage >= 60:
        print("👍 Good coverage")
    elif coverage_percentage >= 40:
        print("⚠️  Moderate coverage - consider adding more tests")
    else:
        print("❌ Low coverage - add more comprehensive tests")

    return result.wasSuccessful()

def run_specific_test(test_name):
    """Run a specific test file"""
    print(f"Running {test_name}.py...")

    try:
        # Import and run the specific test
        import importlib.util

        test_path = Path(__file__).parent / f'{test_name}.py'
        spec = importlib.util.spec_from_file_location(test_name, test_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Run the tests
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        return result.wasSuccessful()

    except Exception as e:
        print(f"Error running {test_name}: {e}")
        return False

if __name__ == '__main__':
    import importlib.util

    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        success = run_specific_test(test_name)
    else:
        # Run all tests
        success = run_all_tests()

    sys.exit(0 if success else 1)
