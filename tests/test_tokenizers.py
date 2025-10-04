"""
Unit tests for base tokenizer functionality.

Tests the implementation of:
- apply_rule function
- find_applicable_rules function  
- is_terminal helper
- tokenize_base function (non-deterministic)

Following the specifications from Tasks file Step 1.2.
"""

import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

from bpe_tokenizer.core import Tokenization, BytePairDictionary, T_empty
from bpe_tokenizer.tokenizers import apply_rule, find_applicable_rules, is_terminal, tokenize_base


def log_test(test_name, passed, details=""):
    """Log test results with details."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status}: {test_name}")
    if details:
        print(f"    {details}")
    if not passed:
        print(f"    Expected behavior not met")
    print()


def test_apply_rule_functionality():
    """Test apply_rule function."""
    print("=== Testing apply_rule Function ===")
    
    # Test 1: Basic rule application
    try:
        tau = Tokenization(["a", "b", "c"])
        rule = ("a", "b")
        position = 0
        
        result = apply_rule(tau, rule, position)
        expected_tokens = ["ab", "c"]
        
        passed = (result.tokens == expected_tokens and 
                 result.concatenate() == "abc")
        log_test("Basic rule application", 
                passed,
                f"Applied {rule} at pos {position}: {tau} → {result}")
    except Exception as e:
        log_test("Basic rule application", False, f"Exception: {e}")
    
    # Test 2: Rule application in middle
    try:
        tau = Tokenization(["x", "a", "b", "y"])
        rule = ("a", "b")
        position = 1
        
        result = apply_rule(tau, rule, position)
        expected_tokens = ["x", "ab", "y"]
        
        passed = (result.tokens == expected_tokens)
        log_test("Middle position rule application", 
                passed,
                f"Applied {rule} at pos {position}: {tau} → {result}")
    except Exception as e:
        log_test("Middle position rule application", False, f"Exception: {e}")
    
    # Test 3: Rule application at end
    try:
        tau = Tokenization(["x", "a", "b"])
        rule = ("a", "b")
        position = 1
        
        result = apply_rule(tau, rule, position)
        expected_tokens = ["x", "ab"]
        
        passed = (result.tokens == expected_tokens)
        log_test("End position rule application", 
                passed,
                f"Applied {rule} at pos {position}: {tau} → {result}")
    except Exception as e:
        log_test("End position rule application", False, f"Exception: {e}")
    
    # Test 4: Invalid position handling
    try:
        tau = Tokenization(["a", "b"])
        rule = ("a", "b")
        
        # Test invalid positions
        invalid_positions = [-1, 1, 2]  # Only position 0 is valid for length 2
        errors_caught = 0
        
        for pos in invalid_positions:
            try:
                apply_rule(tau, rule, pos)
            except (IndexError, ValueError):
                errors_caught += 1
        
        passed = (errors_caught == len(invalid_positions))
        log_test("Invalid position handling", 
                passed,
                f"Correctly rejected {errors_caught}/{len(invalid_positions)} invalid positions")
    except Exception as e:
        log_test("Invalid position handling", False, f"Exception: {e}")
    
    # Test 5: Rule mismatch handling
    try:
        tau = Tokenization(["a", "c"])  # No "b" to match rule
        rule = ("a", "b")
        position = 0
        
        try:
            apply_rule(tau, rule, position)
            log_test("Rule mismatch handling", False, "Should have rejected mismatched rule")
        except ValueError:
            log_test("Rule mismatch handling", True, "Correctly rejected mismatched rule")
    except Exception as e:
        log_test("Rule mismatch handling", False, f"Unexpected exception: {e}")


def test_find_applicable_rules_functionality():
    """Test find_applicable_rules function."""
    print("=== Testing find_applicable_rules Function ===")
    
    # Test 1: Single applicable rule
    try:
        tau = Tokenization(["a", "b", "c"])
        D = BytePairDictionary([("a", "b"), ("x", "y")])
        
        applicable = find_applicable_rules(tau, D)
        expected = [(("a", "b"), 0)]  # Rule (a,b) at position 0
        
        passed = (applicable == expected)
        log_test("Single applicable rule", 
                passed,
                f"Found applicable rules: {applicable}")
    except Exception as e:
        log_test("Single applicable rule", False, f"Exception: {e}")
    
    # Test 2: Multiple applicable rules
    try:
        tau = Tokenization(["a", "b", "c", "d"])
        D = BytePairDictionary([("a", "b"), ("c", "d"), ("b", "c")])
        
        applicable = find_applicable_rules(tau, D)
        # Should find: (a,b) at 0, (b,c) at 1, (c,d) at 2
        expected_rules = {("a", "b"), ("b", "c"), ("c", "d")}
        found_rules = {rule for rule, pos in applicable}
        
        passed = (found_rules == expected_rules and len(applicable) == 3)
        log_test("Multiple applicable rules", 
                passed,
                f"Found applicable rules: {applicable}")
    except Exception as e:
        log_test("Multiple applicable rules", False, f"Exception: {e}")
    
    # Test 3: No applicable rules
    try:
        tau = Tokenization(["a", "c", "e"])  # No adjacent pairs match dictionary
        D = BytePairDictionary([("a", "b"), ("b", "c")])
        
        applicable = find_applicable_rules(tau, D)
        
        passed = (len(applicable) == 0)
        log_test("No applicable rules", 
                passed,
                f"Found applicable rules: {applicable}")
    except Exception as e:
        log_test("No applicable rules", False, f"Exception: {e}")
    
    # Test 4: Empty tokenization
    try:
        tau = Tokenization([])
        D = BytePairDictionary([("a", "b")])
        
        applicable = find_applicable_rules(tau, D)
        
        passed = (len(applicable) == 0)
        log_test("Empty tokenization", 
                passed,
                f"Found applicable rules: {applicable}")
    except Exception as e:
        log_test("Empty tokenization", False, f"Exception: {e}")


def test_is_terminal_functionality():
    """Test is_terminal function."""
    print("=== Testing is_terminal Function ===")
    
    # Test 1: Terminal tokenization
    try:
        tau = Tokenization(["ab", "cd"])  # No rules apply
        D = BytePairDictionary([("a", "b"), ("c", "d")])  # Rules already applied
        
        terminal = is_terminal(tau, D)
        
        passed = terminal
        log_test("Terminal tokenization", 
                passed,
                f"Tokenization {tau} is terminal: {terminal}")
    except Exception as e:
        log_test("Terminal tokenization", False, f"Exception: {e}")
    
    # Test 2: Non-terminal tokenization
    try:
        tau = Tokenization(["a", "b", "c"])  # Rule (a,b) can apply
        D = BytePairDictionary([("a", "b")])
        
        terminal = is_terminal(tau, D)
        
        passed = not terminal
        log_test("Non-terminal tokenization", 
                passed,
                f"Tokenization {tau} is terminal: {terminal}")
    except Exception as e:
        log_test("Non-terminal tokenization", False, f"Exception: {e}")
    
    # Test 3: Empty dictionary (always terminal)
    try:
        tau = Tokenization(["a", "b", "c"])
        D = BytePairDictionary([])  # No rules
        
        terminal = is_terminal(tau, D)
        
        passed = terminal
        log_test("Empty dictionary (always terminal)", 
                passed,
                f"With empty dictionary, tokenization is terminal: {terminal}")
    except Exception as e:
        log_test("Empty dictionary (always terminal)", False, f"Exception: {e}")


def test_tokenize_base_functionality():
    """Test tokenize_base function."""
    print("=== Testing tokenize_base Function ===")
    
    # Test 1: Simple single-path tokenization (from tasks)
    try:
        D = BytePairDictionary([('a', 'b')])
        w = "ab"
        results = tokenize_base(w, D)
        
        # Should have one result: "ab" as single token
        expected_count = 1
        expected_concatenation = "ab"
        expected_length = 1
        
        passed = (len(results) == expected_count and 
                 results[0].concatenate() == expected_concatenation and
                 len(results[0]) == expected_length)
        log_test("Simple single-path tokenization", 
                passed,
                f"Results: {[str(r) for r in results]}")
    except Exception as e:
        log_test("Simple single-path tokenization", False, f"Exception: {e}")
    
    # Test 2: Multiple rule applications
    try:
        D = BytePairDictionary([('a', 'b'), ('b', 'c')])
        w = "abc"
        results = tokenize_base(w, D)
        
        # All results should concatenate back to original string
        all_correct = all(r.concatenate() == w for r in results)
        # All should be terminal
        all_terminal = all(is_terminal(r, D) for r in results)
        
        passed = (len(results) >= 1 and all_correct and all_terminal)
        log_test("Multiple rule applications", 
                passed,
                f"Found {len(results)} results: {[str(r) for r in results]}")
    except Exception as e:
        log_test("Multiple rule applications", False, f"Exception: {e}")
    
    # Test 3: No applicable rules
    try:
        D = BytePairDictionary([('x', 'y')])  # Rule doesn't match string
        w = "abc"
        results = tokenize_base(w, D)
        
        # Should return character-level tokenization
        expected_result = T_empty(w)
        
        passed = (len(results) == 1 and 
                 results[0].tokens == expected_result.tokens)
        log_test("No applicable rules", 
                passed,
                f"Result: {results[0]} (character-level)")
    except Exception as e:
        log_test("No applicable rules", False, f"Exception: {e}")
    
    # Test 4: Empty string
    try:
        D = BytePairDictionary([('a', 'b')])
        w = ""
        results = tokenize_base(w, D)
        
        # Should return empty tokenization
        passed = (len(results) == 1 and 
                 len(results[0]) == 0 and
                 results[0].concatenate() == "")
        log_test("Empty string tokenization", 
                passed,
                f"Result: {results[0]}")
    except Exception as e:
        log_test("Empty string tokenization", False, f"Exception: {e}")
    
    # Test 5: Verify all results are terminal
    try:
        D = BytePairDictionary([('a', 'b'), ('c', 'd')])
        w = "abcd"
        results = tokenize_base(w, D)
        
        all_terminal = all(is_terminal(r, D) for r in results)
        all_concatenate = all(r.concatenate() == w for r in results)
        
        passed = (all_terminal and all_concatenate and len(results) >= 1)
        log_test("Terminal condition verification", 
                passed,
                f"All {len(results)} results are terminal and concatenate correctly")
    except Exception as e:
        log_test("Terminal condition verification", False, f"Exception: {e}")


def test_integration_with_example_from_tasks():
    """Test examples provided in the tasks file."""
    print("=== Testing Examples from Tasks File ===")
    
    # Test from tasks: test_simple_base
    try:
        D = BytePairDictionary([('a', 'b')])
        w = "ab"
        results = tokenize_base(w, D)
        
        test_passed = (len(results) == 1 and
                      results[0].concatenate() == "ab" and
                      len(results[0]) == 1)  # Single token "ab"
        
        log_test("Task example: test_simple_base", 
                test_passed,
                f"Results: {[str(r) for r in results]}")
    except Exception as e:
        log_test("Task example: test_simple_base", False, f"Exception: {e}")
    
    # Test from tasks: test_multiple_paths  
    try:
        D = BytePairDictionary([('a', 'b'), ('b', 'c')])
        w = "abc"
        results = tokenize_base(w, D)
        
        # Should find tokenizations, all concatenating to original
        all_correct = all(r.concatenate() == w for r in results)
        
        test_passed = (len(results) >= 1 and all_correct)
        log_test("Task example: test_multiple_paths", 
                test_passed,
                f"Found {len(results)} valid tokenizations")
    except Exception as e:
        log_test("Task example: test_multiple_paths", False, f"Exception: {e}")
    
    # Test from tasks: test_terminal_condition
    try:
        D = BytePairDictionary([('a', 'b')])
        w = "abc"
        results = tokenize_base(w, D)
        
        all_terminal = all(is_terminal(result, D) for result in results)
        
        test_passed = all_terminal
        log_test("Task example: test_terminal_condition", 
                test_passed,
                f"All {len(results)} results are terminal")
    except Exception as e:
        log_test("Task example: test_terminal_condition", False, f"Exception: {e}")


def run_all_tests():
    """Run all test suites."""
    print("BPE Tokenization Library - Base Tokenizer Tests")
    print("=" * 55)
    print()
    
    test_apply_rule_functionality()
    test_find_applicable_rules_functionality()
    test_is_terminal_functionality() 
    test_tokenize_base_functionality()
    test_integration_with_example_from_tasks()
    
    print("=" * 55)
    print("All base tokenizer tests completed!")


if __name__ == "__main__":
    run_all_tests()