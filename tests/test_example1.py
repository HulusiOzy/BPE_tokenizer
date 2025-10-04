"""
Test Example 1 from page 4 of the paper and verify core functionality.

This test verifies that the implementation correctly handles the formal
definitions and examples from "Formalizing BPE Tokenization".
"""

import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

from bpe_tokenizer.core import Tokenization, BytePairDictionary, T_empty


def log_test(test_name, passed, details=""):
    """Log test results with details."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status}: {test_name}")
    if details:
        print(f"    {details}")
    if not passed:
        print(f"    Expected behavior not met")
    print()


def test_tokenization_functionality():
    """Test core Tokenization class functionality."""
    print("=== Testing Tokenization Class ===")
    
    # Test 1: Valid tokenization creation and concatenation
    try:
        tokens = ["hello", "world"]
        tau = Tokenization(tokens)
        concatenated = tau.concatenate()
        length = len(tau)
        
        expected_concat = "helloworld"
        expected_length = 2
        
        passed = (concatenated == expected_concat and length == expected_length)
        log_test("Tokenization creation and concatenation", 
                passed, 
                f"Created {tau}, concatenated to '{concatenated}', length {length}")
    except Exception as e:
        log_test("Tokenization creation and concatenation", False, f"Exception: {e}")
    
    # Test 2: Empty tokenization handling
    try:
        empty_tau = Tokenization([])
        empty_concat = empty_tau.concatenate()
        empty_length = len(empty_tau)
        
        passed = (empty_concat == "" and empty_length == 0)
        log_test("Empty tokenization handling", 
                passed,
                f"Empty tokenization: {empty_tau}, concat: '{empty_concat}', length: {empty_length}")
    except Exception as e:
        log_test("Empty tokenization handling", False, f"Exception: {e}")
    
    # Test 3: Non-empty token validation (Σ+ constraint)
    try:
        # This should raise an error due to empty token
        try:
            invalid_tau = Tokenization(["a", "", "c"])
            log_test("Empty token validation", False, "Should have rejected empty token")
        except ValueError:
            log_test("Empty token validation", True, "Correctly rejected empty token")
    except Exception as e:
        log_test("Empty token validation", False, f"Unexpected exception: {e}")
    
    # Test 4: Token access and iteration
    try:
        tokens = ["ab", "cd", "ef"]
        tau = Tokenization(tokens)
        
        # Test indexing
        first_token = tau[0]
        last_token = tau[2]
        
        # Test iteration
        iterated_tokens = list(tau)
        
        passed = (first_token == "ab" and 
                 last_token == "ef" and 
                 iterated_tokens == tokens)
        log_test("Token access and iteration", 
                passed,
                f"Access: first='{first_token}', last='{last_token}', iteration={iterated_tokens}")
    except Exception as e:
        log_test("Token access and iteration", False, f"Exception: {e}")


def test_dictionary_functionality():
    """Test BytePairDictionary class functionality."""
    print("=== Testing BytePairDictionary Class ===")
    
    # Test 1: Dictionary creation and priority handling
    try:
        rules = [("a", "b"), ("c", "d"), ("ab", "c")]
        D = BytePairDictionary(rules)
        
        # Test priority assignment (lower index = higher priority)
        priority_ab = D.get_priority("a", "b")
        priority_cd = D.get_priority("c", "d") 
        priority_abc = D.get_priority("ab", "c")
        priority_missing = D.get_priority("x", "y")
        
        passed = (priority_ab == 0 and 
                 priority_cd == 1 and 
                 priority_abc == 2 and 
                 priority_missing == -1)
        log_test("Dictionary priority ordering", 
                passed,
                f"Priorities: a≀b={priority_ab}, c≀d={priority_cd}, ab≀c={priority_abc}, x≀y={priority_missing}")
    except Exception as e:
        log_test("Dictionary priority ordering", False, f"Exception: {e}")
    
    # Test 2: Rule existence checking
    try:
        rules = [("a", "b"), ("c", "d")]
        D = BytePairDictionary(rules)
        
        can_apply_ab = D.can_apply("a", "b")
        can_apply_cd = D.can_apply("c", "d")
        can_apply_missing = D.can_apply("x", "y")
        
        passed = (can_apply_ab and can_apply_cd and not can_apply_missing)
        log_test("Rule existence checking", 
                passed,
                f"Can apply: a≀b={can_apply_ab}, c≀d={can_apply_cd}, x≀y={can_apply_missing}")
    except Exception as e:
        log_test("Rule existence checking", False, f"Exception: {e}")
    
    # Test 3: Invalid rule validation
    try:
        # This should raise an error due to invalid rule format
        try:
            invalid_D = BytePairDictionary([("a", "b", "c")])  # Too many elements
            log_test("Invalid rule validation", False, "Should have rejected invalid rule format")
        except ValueError:
            log_test("Invalid rule validation", True, "Correctly rejected invalid rule format")
    except Exception as e:
        log_test("Invalid rule validation", False, f"Unexpected exception: {e}")


def test_t_empty_functionality():
    """Test T_empty function functionality."""
    print("=== Testing T_empty Function ===")
    
    # Test 1: Character-level tokenization property
    try:
        w = "hello"
        tau = T_empty(w)
        
        # Each character should be its own token
        tokens = tau.tokens
        concatenated = tau.concatenate()
        length = len(tau)
        
        expected_tokens = ["h", "e", "l", "l", "o"]
        expected_concat = w
        expected_length = len(w)
        
        passed = (tokens == expected_tokens and 
                 concatenated == expected_concat and 
                 length == expected_length)
        log_test("Character-level tokenization", 
                passed,
                f"T_empty('{w}') = {tau}, tokens={tokens}")
    except Exception as e:
        log_test("Character-level tokenization", False, f"Exception: {e}")
    
    # Test 2: Empty string handling
    try:
        empty_tau = T_empty("")
        empty_length = len(empty_tau)
        empty_concat = empty_tau.concatenate()
        
        passed = (empty_length == 0 and empty_concat == "")
        log_test("Empty string tokenization", 
                passed,
                f"T_empty('') = {empty_tau}, length={empty_length}")
    except Exception as e:
        log_test("Empty string tokenization", False, f"Exception: {e}")
    
    # Test 3: Concatenation property π(T∅(w)) = w
    try:
        test_strings = ["a", "abc", "hello world", "123!@#"]
        all_passed = True
        
        for w in test_strings:
            tau = T_empty(w)
            if tau.concatenate() != w:
                all_passed = False
                break
        
        log_test("Concatenation property π(T∅(w)) = w", 
                all_passed,
                f"Tested strings: {test_strings}")
    except Exception as e:
        log_test("Concatenation property π(T∅(w)) = w", False, f"Exception: {e}")


def test_example_1_from_paper():
    """Test Example 1 from page 4 of the paper."""
    print("=== Testing Example 1 from Paper (Page 4) ===")
    
    try:
        # Create dictionary D = [a≀b, a≀bc, b≀c, ab≀c]
        D = BytePairDictionary([
            ('a', 'b'),
            ('a', 'bc'),
            ('b', 'c'),
            ('ab', 'c')
        ])
        
        # Create initial tokenization of "abcbcab"
        w = "abcbcab"
        initial = T_empty(w)
        
        # Verify properties mentioned in the example
        concatenation_correct = (initial.concatenate() == w)
        length_correct = (len(initial) == 7)  # 7 characters
        
        # Verify dictionary priorities are as expected
        priority_checks = [
            D.get_priority('a', 'b') == 0,     # Highest priority
            D.get_priority('a', 'bc') == 1,
            D.get_priority('b', 'c') == 2,
            D.get_priority('ab', 'c') == 3     # Lowest priority
        ]
        
        passed = concatenation_correct and length_correct and all(priority_checks)
        
        log_test("Example 1 setup verification", 
                passed,
                f"Dictionary: {D}")
        log_test("Example 1 initial tokenization", 
                passed,
                f"T∅('{w}') = {initial}, length={len(initial)}")
        log_test("Example 1 priority verification", 
                passed,
                f"Priorities: {[D.get_priority('a', 'b'), D.get_priority('a', 'bc'), D.get_priority('b', 'c'), D.get_priority('ab', 'c')]}")
        
    except Exception as e:
        log_test("Example 1 from paper", False, f"Exception: {e}")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("=== Testing Edge Cases ===")
    
    # Test 1: Single character strings
    try:
        single_char = T_empty("x")
        passed = (len(single_char) == 1 and 
                 single_char[0] == "x" and 
                 single_char.concatenate() == "x")
        log_test("Single character tokenization", 
                passed,
                f"T_empty('x') = {single_char}")
    except Exception as e:
        log_test("Single character tokenization", False, f"Exception: {e}")
    
    # Test 2: Empty dictionary
    try:
        empty_dict = BytePairDictionary([])
        passed = (len(empty_dict) == 0 and 
                 not empty_dict.can_apply("a", "b") and 
                 empty_dict.get_priority("a", "b") == -1)
        log_test("Empty dictionary handling", 
                passed,
                f"Empty dict: {empty_dict}")
    except Exception as e:
        log_test("Empty dictionary handling", False, f"Exception: {e}")
    
    # Test 3: Unicode and special characters
    try:
        unicode_str = "café"
        tau = T_empty(unicode_str)
        passed = (tau.concatenate() == unicode_str and len(tau) == 4)
        log_test("Unicode character handling", 
                passed,
                f"T_empty('{unicode_str}') = {tau}")
    except Exception as e:
        log_test("Unicode character handling", False, f"Exception: {e}")


def run_all_tests():
    """Run all test suites."""
    print("BPE Tokenization Library - Core Functionality Tests")
    print("=" * 55)
    print()
    
    test_tokenization_functionality()
    test_dictionary_functionality()
    test_t_empty_functionality()
    test_example_1_from_paper()
    test_edge_cases()
    
    print("=" * 55)
    print("All tests completed!")


if __name__ == "__main__":
    run_all_tests()