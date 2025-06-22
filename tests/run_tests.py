"""
Modern pytest-based test runner for PII deidentification framework.
"""
import subprocess
import sys
from pathlib import Path


def run_pytest_tests():
    """Run all tests using pytest."""
    print("🧪 PII Deidentification Framework - Modern Test Suite")
    print("=" * 60)
    
    # Change to the project root directory
    project_root = Path(__file__).parent.parent
    
    try:
        # Run pytest with coverage and proper configuration
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "-v",                    # Verbose output
            "--tb=short",           # Short traceback format
            "--cov=src",            # Coverage for src directory
            "--cov-report=term-missing",  # Show missing lines
            "--cov-report=html",    # Generate HTML coverage report
            "tests/",               # Test directory
        ], 
        cwd=project_root,
        capture_output=False,  # Show output in real-time
        text=True
        )
        
        print("\n" + "=" * 60)
        if result.returncode == 0:
            print("🎉 ALL TESTS PASSED! 🎉")
            print("📊 Coverage report generated in htmlcov/index.html")
        else:
            print("💥 SOME TESTS FAILED!")
            
        return result.returncode == 0
        
    except FileNotFoundError:
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
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "-v",
            "-m", marker,
            "tests/",
        ], 
        cwd=project_root,
        capture_output=False,
        text=True
        )
        
        return result.returncode == 0
        
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