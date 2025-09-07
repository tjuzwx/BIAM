"""
Test runner script for BIAM project
"""

import unittest
import sys
import os
import subprocess

def run_unit_tests():
    """
    Run unit tests
    """
    print("Running unit tests...")
    
    # Add project root to path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_performance_tests():
    """
    Run performance tests
    """
    print("Running performance tests...")
    
    try:
        # Run performance tests
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/test_biam_performance.py', 
            '-v', '--benchmark-only'
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"Error running performance tests: {e}")
        return False

def run_linting():
    """
    Run code linting
    """
    print("Running code linting...")
    
    try:
        # Run flake8
        result = subprocess.run([
            sys.executable, '-m', 'flake8', 
            '.', '--count', '--select=E9,F63,F7,F82', 
            '--show-source', '--statistics'
        ], capture_output=True, text=True)
        
        if result.stdout:
            print("Linting issues found:")
            print(result.stdout)
        
        return result.returncode == 0
    except Exception as e:
        print(f"Error running linting: {e}")
        return False

def run_type_checking():
    """
    Run type checking
    """
    print("Running type checking...")
    
    try:
        # Run mypy
        result = subprocess.run([
            sys.executable, '-m', 'mypy', 
            '.', '--ignore-missing-imports'
        ], capture_output=True, text=True)
        
        if result.stdout:
            print("Type checking results:")
            print(result.stdout)
        
        return result.returncode == 0
    except Exception as e:
        print(f"Error running type checking: {e}")
        return False

def run_code_formatting_check():
    """
    Check code formatting
    """
    print("Checking code formatting...")
    
    try:
        # Run black check
        result = subprocess.run([
            sys.executable, '-m', 'black', 
            '.', '--check', '--diff'
        ], capture_output=True, text=True)
        
        if result.stdout:
            print("Formatting issues found:")
            print(result.stdout)
        
        return result.returncode == 0
    except Exception as e:
        print(f"Error checking code formatting: {e}")
        return False

def main():
    """
    Main test runner
    """
    print("=" * 60)
    print("BIAM Project Test Suite")
    print("=" * 60)
    
    # Run all tests
    tests_passed = []
    
    # Unit tests
    tests_passed.append(("Unit Tests", run_unit_tests()))
    
    # Performance tests
    tests_passed.append(("Performance Tests", run_performance_tests()))
    
    # Code quality checks
    tests_passed.append(("Linting", run_linting()))
    tests_passed.append(("Type Checking", run_type_checking()))
    tests_passed.append(("Code Formatting", run_code_formatting_check()))
    
    # Print results
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in tests_passed:
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("All tests PASSED! ✅")
        return 0
    else:
        print("Some tests FAILED! ❌")
        return 1

if __name__ == '__main__':
    sys.exit(main())
