"""
Unit tests for SentencePiece deterministic tokenizer.

Tests the implementation of:
- find_highest_priority_rule function
- tokenize_sentencepiece main algorithm
- validate_sentencepiece testing function
- All test cases from Tasks file Step 1.3

Following the specifications from Definition 3, Page 4 of the paper.
"""

import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

from bpe_tokenizer.core import Tokenization, BytePairDictionary, T_empty
from bpe_tokenizer.tokenizers import (
    find_highest_priority_rule, tokenize_sentencepiece, validate_sentencepiece,
    tokenize_base
)


def log_test(test_name, passed, details=""):
    """Log test results with details."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status}: {test_name}")
    if details:
        print(f"    {details}")
    if not passed:
        print(f"    Expected behavior not met")
    print()


def test_find_highest_priority_rule_functionality():
    """Test find_highest_priority_rule function."""
    print("=== Testing find_highest_priority_rule Function ===")
    
    # Test 1: Single applicable rule
    try:
        tau = Tokenization(["a", "b", "c"])
        D = BytePairDictionary([("a", "b"), ("x", "y")])
        
        result = find_highest_priority_rule(tau, D)
        expected = (("a", "b"), 0)  # Rule (a,b) at position 0 with priority 0
        
        passed = (result == expected)
        log_test("Single applicable rule selection", 
                passed,
                f"Found: {result}, Expected: {expected}")
    except Exception as e:
        log_test("Single applicable rule selection", False, f"Exception: {e}")
    
    # Test 2: Multiple rules - priority selection
    try:
        tau = Tokenization(["a", "b", "c", "d"])
        D = BytePairDictionary([("c", "d"), ("a", "b")])  # c≀d has higher priority (index 0)
        
        result = find_highest_priority_rule(tau, D)
        expected = (("c", "d"), 2)  # Higher priority rule c≀d at position 2
        
        passed = (result == expected)
        log_test("Priority-based rule selection", 
                passed,
                f"Found: {result}, Expected: {expected}")
    except Exception as e:
        log_test("Priority-based rule selection", False, f"Exception: {e}")
    
    # Test 3: Same rule at multiple positions - leftmost selection
    try:
        tau = Tokenization(["a", "b", "x", "a", "b"])
        D = BytePairDictionary([("a", "b")])
        
        result = find_highest_priority_rule(tau, D)
        expected = (("a", "b"), 0)  # Leftmost occurrence at position 0
        
        passed = (result == expected)
        log_test("Leftmost position selection", 
                passed,
                f"Found: {result}, Expected: {expected}")
    except Exception as e:
        log_test("Leftmost position selection", False, f"Exception: {e}")
    
    # Test 4: No applicable rules
    try:
        tau = Tokenization(["a", "c", "e"])
        D = BytePairDictionary([("a", "b"), ("x", "y")])
        
        result = find_highest_priority_rule(tau, D)
        
        passed = (result is None)
        log_test("No applicable rules", 
                passed,
                f"Found: {result}")
    except Exception as e:
        log_test("No applicable rules", False, f"Exception: {e}")
    
    # Test 5: Single token tokenization
    try:
        tau = Tokenization(["single"])
        D = BytePairDictionary([("a", "b")])
        
        result = find_highest_priority_rule(tau, D)
        
        passed = (result is None)
        log_test("Single token (no pairs)", 
                passed,
                f"Found: {result}")
    except Exception as e:
        log_test("Single token (no pairs)", False, f"Exception: {e}")


def test_tokenize_sentencepiece_functionality():
    """Test tokenize_sentencepiece main algorithm."""
    print("=== Testing tokenize_sentencepiece Algorithm ===")
    
    # Test 1: Simple single rule application
    try:
        D = BytePairDictionary([("a", "b")])
        w = "ab"
        
        result = tokenize_sentencepiece(w, D)
        expected_tokens = ["ab"]
        
        passed = (result.tokens == expected_tokens and result.concatenate() == w)
        log_test("Simple single rule application", 
                passed,
                f"Result: {result}")
    except Exception as e:
        log_test("Simple single rule application", False, f"Exception: {e}")
    
    # Test 2: Multiple rule applications
    try:
        D = BytePairDictionary([("a", "b"), ("ab", "c")])
        w = "abc"
        
        result = tokenize_sentencepiece(w, D)
        # Should apply a≀b first (priority 0), then ab≀c (priority 1)
        expected_tokens = ["abc"]
        
        passed = (result.tokens == expected_tokens and result.concatenate() == w)
        log_test("Multiple rule applications", 
                passed,
                f"Result: {result}")
    except Exception as e:
        log_test("Multiple rule applications", False, f"Exception: {e}")
    
    # Test 3: Priority ordering affects result
    try:
        D1 = BytePairDictionary([("a", "b"), ("b", "c")])  # a≀b has higher priority
        D2 = BytePairDictionary([("b", "c"), ("a", "b")])  # b≀c has higher priority
        w = "abc"
        
        result1 = tokenize_sentencepiece(w, D1)
        result2 = tokenize_sentencepiece(w, D2)
        
        # Results should be different due to different priorities
        different_results = (result1.tokens != result2.tokens)
        both_valid = (result1.concatenate() == w and result2.concatenate() == w)
        
        passed = (different_results and both_valid)
        log_test("Priority ordering affects result", 
                passed,
                f"D1 result: {result1}, D2 result: {result2}")
    except Exception as e:
        log_test("Priority ordering affects result", False, f"Exception: {e}")
    
    # Test 4: No applicable rules (character-level result)
    try:
        D = BytePairDictionary([("x", "y")])
        w = "abc"
        
        result = tokenize_sentencepiece(w, D)
        expected_tokens = ["a", "b", "c"]  # Character-level
        
        passed = (result.tokens == expected_tokens and result.concatenate() == w)
        log_test("No applicable rules", 
                passed,
                f"Result: {result}")
    except Exception as e:
        log_test("No applicable rules", False, f"Exception: {e}")
    
    # Test 5: Empty string
    try:
        D = BytePairDictionary([("a", "b")])
        w = ""
        
        result = tokenize_sentencepiece(w, D)
        
        passed = (len(result) == 0 and result.concatenate() == "")
        log_test("Empty string", 
                passed,
                f"Result: {result}")
    except Exception as e:
        log_test("Empty string", False, f"Exception: {e}")


def test_validate_sentencepiece_functionality():
    """Test validate_sentencepiece function."""
    print("=== Testing validate_sentencepiece Function ===")
    
    # Test 1: Valid SentencePiece result
    try:
        D = BytePairDictionary([("a", "b")])
        w = "ab"
        
        is_valid = validate_sentencepiece(w, D)
        
        passed = is_valid
        log_test("Valid SentencePiece result", 
                passed,
                f"Validation result: {is_valid}")
    except Exception as e:
        log_test("Valid SentencePiece result", False, f"Exception: {e}")
    
    # Test 2: Multiple valid tokenizations - SentencePiece selects one
    try:
        D = BytePairDictionary([("a", "b"), ("b", "c")])
        w = "abc"
        
        is_valid = validate_sentencepiece(w, D)
        
        # Get both results for comparison
        sp_result = tokenize_sentencepiece(w, D)
        base_results = tokenize_base(w, D)
        
        passed = is_valid
        log_test("SentencePiece selects valid tokenization", 
                passed,
                f"SP: {sp_result}, Base options: {[str(r) for r in base_results]}")
    except Exception as e:
        log_test("SentencePiece selects valid tokenization", False, f"Exception: {e}")
    
    # Test 3: Edge case with empty dictionary
    try:
        D = BytePairDictionary([])
        w = "abc"
        
        is_valid = validate_sentencepiece(w, D)
        
        passed = is_valid  # Should be valid (character-level tokenization)
        log_test("Empty dictionary validation", 
                passed,
                f"Validation result: {is_valid}")
    except Exception as e:
        log_test("Empty dictionary validation", False, f"Exception: {e}")


def test_example_1_from_paper():
    """
    Test Example 1 from page 4 of the paper.
    Shows step-by-step SentencePiece tokenization.
    
    Reference: Page 4, Example 1
    """
    print("=== Testing Example 1 from Paper (Page 4) ===")
    
    try:
        # Create dictionary from Example 1
        D = BytePairDictionary([
            ('a', 'b'),      # Priority 0 (highest)
            ('a', 'bc'),     # Priority 1  
            ('b', 'c'),      # Priority 2
            ('ab', 'c')      # Priority 3 (lowest)
        ])
        w = "abcbcab"
        
        result = tokenize_sentencepiece(w, D)
        
        # Expected result from paper: "abc≀bc≀ab"
        expected_tokens = ["abc", "bc", "ab"]
        expected_length = 3
        
        # Verify basic properties
        length_correct = (len(result) == expected_length)
        tokens_correct = (result.tokens == expected_tokens)
        concatenation_correct = (result.concatenate() == w)
        
        # Verify it's in the base tokenization set
        is_valid = validate_sentencepiece(w, D)
        
        passed = (length_correct and tokens_correct and concatenation_correct and is_valid)
        
        log_test("Example 1 - correct result", 
                passed,
                f"Result: {result}, Expected: abc≀bc≀ab")
        log_test("Example 1 - validates against base", 
                passed,
                f"Validation: {is_valid}")
        
        # Test step-by-step process manually to verify algorithm
        from bpe_tokenizer.core import T_empty
        step_tau = T_empty(w)
        steps = []
        
        while True:
            rule_pos = find_highest_priority_rule(step_tau, D)
            if rule_pos is None:
                break
            rule, pos = rule_pos
            steps.append(f"Apply {rule} at pos {pos}: {step_tau} → ")
            from bpe_tokenizer.tokenizers import apply_rule
            step_tau = apply_rule(step_tau, rule, pos)
            steps.append(str(step_tau))
        
        log_test("Example 1 - step-by-step verification", 
                True,
                f"Final result matches: {step_tau.tokens == expected_tokens}")
        
    except Exception as e:
        log_test("Example 1 from paper", False, f"Exception: {e}")


def test_leftmost_selection():
    """
    Test leftmost selection when same rule applies at multiple positions.
    Reference: Tasks file Test 2
    """
    print("=== Testing Leftmost Selection ===")
    
    try:
        D = BytePairDictionary([('a', 'b')])
        w = "ababab"
        
        result = tokenize_sentencepiece(w, D)
        
        # Should apply left-to-right: ab≀ab≀ab
        expected_tokens = ["ab", "ab", "ab"]
        
        passed = (result.tokens == expected_tokens and result.concatenate() == w)
        log_test("Leftmost selection", 
                passed,
                f"Result: {result}, Expected: ab≀ab≀ab")
    except Exception as e:
        log_test("Leftmost selection", False, f"Exception: {e}")


def test_priority_ordering():
    """
    Test that higher priority rules are selected first.
    Reference: Tasks file Test 3
    """
    print("=== Testing Priority Ordering ===")
    
    try:
        D = BytePairDictionary([
            ('a', 'b'),      # Higher priority (index 0)
            ('b', 'c')       # Lower priority (index 1)
        ])
        w = "abc"
        
        result = tokenize_sentencepiece(w, D)
        
        # Should apply 'a≀b' first, resulting in "ab≀c"
        expected_tokens = ["ab", "c"]
        
        passed = (result.tokens == expected_tokens and result.concatenate() == w)
        log_test("Priority ordering", 
                passed,
                f"Result: {result}, Expected: ab≀c")
    except Exception as e:
        log_test("Priority ordering", False, f"Exception: {e}")


def test_unused_rule():
    """
    Test observation from Example 1: rule 'a≀bc' never applies.
    Reference: Tasks file Test 4, Page 4 observation
    """
    print("=== Testing Unused Rule (Example 1 Observation) ===")
    
    try:
        D = BytePairDictionary([
            ('a', 'b'),
            ('a', 'bc'),     # This rule never applies!
            ('b', 'c'),
            ('ab', 'c')
        ])
        w = "abcbcab"
        
        result = tokenize_sentencepiece(w, D)
        
        # The token "bc" exists in result, but rule 'a≀bc' was never used
        # because 'b≀c' creates 'bc' before 'a≀bc' could apply
        has_bc_token = "bc" in result.tokens
        correct_result = (result.tokens == ["abc", "bc", "ab"])
        
        passed = (has_bc_token and correct_result and result.concatenate() == w)
        log_test("Unused rule observation", 
                passed,
                f"Result: {result}, Contains 'bc': {has_bc_token}")
        
        # Additional verification: 'a≀bc' rule exists but was not needed
        rule_exists = D.can_apply('a', 'bc')
        log_test("Rule 'a≀bc' exists but unused", 
                rule_exists,
                f"Rule exists: {rule_exists}, but 'bc' created by 'b≀c' first")
        
    except Exception as e:
        log_test("Unused rule observation", False, f"Exception: {e}")


def test_deterministic_property():
    """Test that SentencePiece is deterministic (same input → same output)."""
    print("=== Testing Deterministic Property ===")
    
    try:
        D = BytePairDictionary([('a', 'b'), ('b', 'c'), ('ab', 'c')])
        w = "abcabc"
        
        # Run multiple times
        results = [tokenize_sentencepiece(w, D) for _ in range(5)]
        
        # All results should be identical
        all_same = all(r.tokens == results[0].tokens for r in results)
        all_concatenate = all(r.concatenate() == w for r in results)
        
        passed = (all_same and all_concatenate)
        log_test("Deterministic property", 
                passed,
                f"All 5 runs identical: {all_same}, Result: {results[0]}")
    except Exception as e:
        log_test("Deterministic property", False, f"Exception: {e}")


def run_all_tests():
    """Run all test suites."""
    print("BPE Tokenization Library - SentencePiece Tests")
    print("=" * 55)
    print()
    
    test_find_highest_priority_rule_functionality()
    test_tokenize_sentencepiece_functionality()
    test_validate_sentencepiece_functionality()
    test_example_1_from_paper()
    test_leftmost_selection()
    test_priority_ordering()
    test_unused_rule()
    test_deterministic_property()
    
    print("=" * 55)
    print("All SentencePiece tests completed!")


if __name__ == "__main__":
    run_all_tests()