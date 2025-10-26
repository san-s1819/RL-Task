"""
Test script to verify the dropout grading function works correctly.
This includes both correct and incorrect implementations.
"""

import numpy as np
from main2 import grade_dropout_implementation


def test_correct_implementation():
    """Test a correct dropout implementation"""
    correct_code = """
import numpy as np

def dropout(x, p, training):
    if not training:
        return x
    
    if p == 0:
        return x
    
    if p == 1:
        return np.zeros_like(x)
    
    # Generate mask: keep with probability (1-p)
    mask = np.random.rand(*x.shape) > p
    
    # Apply mask and scale
    return x * mask / (1 - p)
"""
    
    success, message, _ = grade_dropout_implementation(correct_code)
    print("Test: Correct Implementation")
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    print(f"  Message: {message}\n")
    return success


def test_no_scaling():
    """Test implementation that forgets to scale"""
    no_scaling_code = """
import numpy as np

def dropout(x, p, training):
    if not training:
        return x
    
    # Missing scaling!
    mask = np.random.rand(*x.shape) > p
    return x * mask
"""
    
    success, message, _ = grade_dropout_implementation(no_scaling_code)
    print("Test: No Scaling (Common Mistake)")
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    print(f"  Message: {message}\n")
    return not success  # Should fail


def test_wrong_scaling():
    """Test implementation with wrong scaling factor"""
    wrong_scaling_code = """
import numpy as np

def dropout(x, p, training):
    if not training:
        return x
    
    # Wrong scaling factor (using p instead of 1/(1-p))
    mask = np.random.rand(*x.shape) > p
    return x * mask * p
"""
    
    success, message, _ = grade_dropout_implementation(wrong_scaling_code)
    print("Test: Wrong Scaling Factor")
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    print(f"  Message: {message}\n")
    return not success  # Should fail


def test_no_train_eval_distinction():
    """Test implementation that applies dropout in eval mode"""
    no_distinction_code = """
import numpy as np

def dropout(x, p, training):
    # Always applies dropout!
    mask = np.random.rand(*x.shape) > p
    return x * mask / (1 - p)
"""
    
    success, message, _ = grade_dropout_implementation(no_distinction_code)
    print("Test: No Train/Eval Distinction")
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    print(f"  Message: {message}\n")
    return not success  # Should fail


def test_missing_function():
    """Test code without dropout function"""
    missing_function_code = """
import numpy as np

# No dropout function defined!
x = np.ones((10, 10))
"""
    
    success, message, _ = grade_dropout_implementation(missing_function_code)
    print("Test: Missing Function")
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    print(f"  Message: {message}\n")
    return not success  # Should fail


def test_alternative_correct_implementation():
    """Test alternative correct implementation using binomial"""
    alternative_code = """
import numpy as np

def dropout(x, p, training):
    if not training:
        return x
    
    # Alternative: use binomial distribution
    keep_prob = 1 - p
    mask = np.random.binomial(1, keep_prob, size=x.shape)
    
    if keep_prob == 0:
        return np.zeros_like(x)
    
    return x * mask / keep_prob
"""
    
    success, message, _ = grade_dropout_implementation(alternative_code)
    print("Test: Alternative Correct Implementation (Binomial)")
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    print(f"  Message: {message}\n")
    return success


def main():
    print("=" * 60)
    print("Testing Dropout Grading Function")
    print("=" * 60)
    print()
    
    tests = [
        ("Correct Implementation", test_correct_implementation),
        ("Alternative Implementation", test_alternative_correct_implementation),
        ("No Scaling Error", test_no_scaling),
        ("Wrong Scaling Error", test_wrong_scaling),
        ("No Train/Eval Distinction", test_no_train_eval_distinction),
        ("Missing Function", test_missing_function),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"  Exception: {e}\n")
            results.append((name, False))
    
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    main()

