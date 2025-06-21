"""
Main test runner for PII deidentification framework.
"""
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_all_tests():
    """Run all test suites."""
    print("🧪 PII Deidentification Framework - Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: PII Extraction Tests
    print("\n1️⃣  Running PII Extraction Tests...")
    try:
        from test_pii_extraction import run_tests as run_pii_tests
        pii_result = run_pii_tests()
        test_results.append(("PII Extraction", pii_result))
    except Exception as e:
        print(f"❌ Failed to run PII extraction tests: {e}")
        test_results.append(("PII Extraction", False))
    
    # Future tests can be added here
    # print("\n2️⃣  Running Matching Logic Tests...")
    # print("\n3️⃣  Running Evaluation Metrics Tests...")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal Tests: {len(test_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        return True
    else:
        print(f"\n💥 {failed} TEST(S) FAILED!")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1) 