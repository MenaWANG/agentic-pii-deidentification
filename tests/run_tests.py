"""
Modern pytest-based test runner for PII deidentification framework.
"""
import pytest
import sys
from pathlib import Path


def run_pytest_tests():
    """Run all tests using pytest."""
    print("🧪 PII Deidentification Framework - Modern Test Suite")
    print("=" * 60)
    
    # Change to the project root directory
    project_root = Path(__file__).parent.parent
    
    try:
        # Run pytest directly using pytest.main()
        args = [
            "-v",                    # Verbose output
            "--tb=short",           # Short traceback format
            "--cov=src",            # Coverage for src directory
            "--cov-report=term-missing",  # Show missing lines
            "--cov-report=html",    # Generate HTML coverage report
            str(project_root / "tests/"),  # Test directory
        ]
        
        result = pytest.main(args)
        
        print("\n" + "=" * 60)
        if result == 0:
            print("🎉 ALL TESTS PASSED! 🎉")
            print("📊 Coverage report generated in htmlcov/index.html")
        else:
            print("💥 SOME TESTS FAILED!")
            
        return result == 0
        
    except ImportError:
        print("❌ pytest not found. Please install with: pip install pytest pytest-cov")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def run_specific_markers(marker):
    """Run tests with specific pytest markers."""
    project_root = Path(__file__).parent.parent
    
    print(f"🧪 Running tests with marker: {marker}")
    
    try:
        args = [
            "-v",
            "-m", marker,
            str(project_root / "tests/"),
        ]
        
        result = pytest.main(args)
        return result == 0
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run PII deidentification tests")
    parser.add_argument("-m", "--marker", help="Run tests with specific marker (unit, integration, baseline, etc.)")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report", default=True)
    
    args = parser.parse_args()
    
    if args.marker:
        success = run_specific_markers(args.marker)
    else:
        success = run_pytest_tests()
    
    exit(0 if success else 1) 