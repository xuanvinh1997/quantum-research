"""Run all tests."""
import sys

# Import test modules
import test_ising
import test_h2
import test_vqa


def main():
    """Run all test suites."""
    print("\n" + "=" * 70)
    print(" " * 20 + "VQA TEST SUITE")
    print("=" * 70 + "\n")

    all_passed = True

    # Run Ising tests
    try:
        test_ising.run_all_tests()
    except Exception as e:
        print(f"\n✗ Ising tests failed: {e}")
        all_passed = False

    print("\n")

    # Run H2 tests
    try:
        test_h2.run_all_tests()
    except Exception as e:
        print(f"\n✗ H2 tests failed: {e}")
        all_passed = False

    print("\n")

    # Run VQA tests
    try:
        test_vqa.run_all_tests()
    except Exception as e:
        print(f"\n✗ VQA tests failed: {e}")
        all_passed = False

    # Final summary
    print("\n" + "=" * 70)
    if all_passed:
        print(" " * 20 + "ALL TESTS PASSED ✓")
    else:
        print(" " * 20 + "SOME TESTS FAILED ✗")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
