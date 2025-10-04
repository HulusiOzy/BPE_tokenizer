"""
Unit tests for dictionary validation functionality.

Tests the implementation of:
- is_proper_dictionary function
- get_rule_dependencies function
- validate_training_construction function
- All test cases from Tasks file Step 2.1

Following Definition 5, Page 5 and related theoretical results.
"""

import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

from bpe_tokenizer.core import BytePairDictionary
from bpe_tokenizer.validation import (
    is_proper_dictionary, get_rule_dependencies, validate_training_construction,
    find_unused_rules, analyze_dictionary_properties
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


def test_is_proper_dictionary_functionality():
    """Test is_proper_dictionary function."""
    print("=== Testing is_proper_dictionary Function ===")
    
    # Test 1: Empty dictionary (trivially proper)
    try:
        D = BytePairDictionary([])
        result = is_proper_dictionary(D)
        
        passed = result
        log_test("Empty dictionary (trivially proper)", 
                passed,
                f"Empty dictionary is proper: {result}")
    except Exception as e:
        log_test("Empty dictionary (trivially proper)", False, f"Exception: {e}")
    
    # Test 2: Single character rules only (always proper)
    try:
        D = BytePairDictionary([('a', 'b'), ('c', 'd'), ('e', 'f')])
        result = is_proper_dictionary(D)
        
        passed = result
        log_test("Single character rules (always proper)", 
                passed,
                f"Single char rules are proper: {result}")
    except Exception as e:
        log_test("Single character rules (always proper)", False, f"Exception: {e}")
    
    # Test 3: Proper dictionary with dependencies
    try:
        D = BytePairDictionary([
            ('a', 'b'),      # Creates 'ab'
            ('b', 'c'),      # Creates 'bc'
            ('ab', 'c'),     # Uses 'ab' from rule 0
            ('a', 'bc')      # Uses 'bc' from rule 1
        ])
        result = is_proper_dictionary(D)
        
        passed = result
        log_test("Proper dictionary with dependencies", 
                passed,
                f"Dictionary: {D}")
    except Exception as e:
        log_test("Proper dictionary with dependencies", False, f"Exception: {e}")
    
    # Test 4: Improper dictionary (missing dependency)
    try:
        D = BytePairDictionary([
            ('ab', 'c'),     # Uses 'ab' but no rule creates it
            ('a', 'b')       # This comes after, so can't help rule 0
        ])
        result = is_proper_dictionary(D)
        
        passed = not result  # Should be False (improper)
        log_test("Improper dictionary (missing dependency)", 
                passed,
                f"Dictionary is properly detected as improper: {not result}")
    except Exception as e:
        log_test("Improper dictionary (missing dependency)", False, f"Exception: {e}")
    
    # Test 5: Example 3 from paper (improper)
    try:
        D = BytePairDictionary([
            ('ab', 'a'),     # Creates 'aba' but no rule creates 'ab'
            ('a', 'b')       # This creates 'ab' but comes after
        ])
        result = is_proper_dictionary(D)
        
        passed = not result  # Should be False (improper)
        log_test("Example 3 from paper (improper)", 
                passed,
                f"Example 3 dictionary correctly detected as improper: {not result}")
    except Exception as e:
        log_test("Example 3 from paper (improper)", False, f"Exception: {e}")


def test_get_rule_dependencies_functionality():
    """Test get_rule_dependencies function."""
    print("=== Testing get_rule_dependencies Function ===")
    
    # Test 1: No dependencies
    try:
        D = BytePairDictionary([('a', 'b'), ('c', 'd')])
        deps = get_rule_dependencies(D)
        
        expected = {0: [], 1: []}
        passed = (deps == expected)
        log_test("No dependencies", 
                passed,
                f"Dependencies: {deps}")
    except Exception as e:
        log_test("No dependencies", False, f"Exception: {e}")
    
    # Test 2: Simple chain dependency
    try:
        D = BytePairDictionary([
            ('a', 'b'),      # Creates 'ab'
            ('ab', 'c')      # Depends on rule 0
        ])
        deps = get_rule_dependencies(D)
        
        expected = {0: [], 1: [0]}
        passed = (deps == expected)
        log_test("Simple chain dependency", 
                passed,
                f"Dependencies: {deps}")
    except Exception as e:
        log_test("Simple chain dependency", False, f"Exception: {e}")
    
    # Test 3: Multiple dependencies
    try:
        D = BytePairDictionary([
            ('a', 'b'),      # Creates 'ab'
            ('c', 'd'),      # Creates 'cd'
            ('ab', 'cd')     # Depends on both rules 0 and 1
        ])
        deps = get_rule_dependencies(D)
        
        # Rule 2 should depend on both rules 0 and 1
        rule_2_deps = set(deps[2])
        expected_deps = {0, 1}
        passed = (rule_2_deps == expected_deps)
        log_test("Multiple dependencies", 
                passed,
                f"Rule 2 dependencies: {deps[2]}")
    except Exception as e:
        log_test("Multiple dependencies", False, f"Exception: {e}")
    
    # Test 4: Long dependency chain
    try:
        D = BytePairDictionary([
            ('a', 'b'),      # Creates 'ab'
            ('ab', 'c'),     # Creates 'abc'
            ('abc', 'd'),    # Creates 'abcd'
            ('abcd', 'e')    # Creates 'abcde'
        ])
        deps = get_rule_dependencies(D)
        
        expected = {
            0: [],
            1: [0],
            2: [1],
            3: [2]
        }
        passed = (deps == expected)
        log_test("Long dependency chain", 
                passed,
                f"Dependencies: {deps}")
    except Exception as e:
        log_test("Long dependency chain", False, f"Exception: {e}")


def test_validate_training_construction_functionality():
    """Test validate_training_construction function."""
    print("=== Testing validate_training_construction Function ===")
    
    # Test 1: Simple proper dictionary with useful rules
    try:
        D = BytePairDictionary([('a', 'b')])
        corpus = "ababab"
        
        result = validate_training_construction(D, corpus)
        
        is_proper = result['is_proper']
        all_useful = result['all_rules_useful']
        construction_valid = result['construction_valid']
        
        passed = (is_proper and all_useful and construction_valid)
        log_test("Simple proper dictionary", 
                passed,
                f"Proper: {is_proper}, Useful: {all_useful}, Valid: {construction_valid}")
    except Exception as e:
        log_test("Simple proper dictionary", False, f"Exception: {e}")
    
    # Test 2: Improper dictionary
    try:
        D = BytePairDictionary([
            ('ab', 'c'),     # Improper - no rule creates 'ab'
            ('a', 'b')
        ])
        corpus = "abc"
        
        result = validate_training_construction(D, corpus)
        
        is_proper = result['is_proper']
        construction_valid = result['construction_valid']
        
        passed = (not is_proper and not construction_valid)
        log_test("Improper dictionary validation", 
                passed,
                f"Properly detected as improper and invalid: {not is_proper}")
    except Exception as e:
        log_test("Improper dictionary validation", False, f"Exception: {e}")
    
    # Test 3: Empty corpus
    try:
        D = BytePairDictionary([('a', 'b')])
        corpus = ""
        
        result = validate_training_construction(D, corpus)
        
        # Should still validate dictionary structure
        has_results = all(key in result for key in ['is_proper', 'all_rules_useful', 'construction_valid'])
        
        passed = has_results
        log_test("Empty corpus handling", 
                passed,
                f"Result keys present: {has_results}")
    except Exception as e:
        log_test("Empty corpus handling", False, f"Exception: {e}")


def test_example_1_proper():
    """
    Example 1 dictionary from page 4 - analyze its properness.
    Note: The tasks file suggests this might NOT be proper due to rule ordering.
    """
    print("=== Testing Example 1 Dictionary Analysis ===")
    
    try:
        D = BytePairDictionary([
            ('a', 'b'),      # Creates 'ab'
            ('a', 'bc'),     # Needs 'bc' - but 'bc' is created later!
            ('b', 'c'),      # Creates 'bc'
            ('ab', 'c')      # Uses 'ab' from rule 0
        ])
        
        is_proper = is_proper_dictionary(D)
        deps = get_rule_dependencies(D)
        
        # Rule 1 ('a', 'bc') needs 'bc' which is created by rule 2
        # This violates the proper ordering requirement
        rule_1_needs_bc = 'bc' in [u + v for u, v in D.rules[:1]]  # Should be False
        
        passed = not is_proper  # Dictionary should be improper
        log_test("Example 1 dictionary is improper", 
                passed,
                f"Dictionary: {D}")
        log_test("Example 1 dependency analysis", 
                True,  # Always show this
                f"Dependencies: {deps}")
        
    except Exception as e:
        log_test("Example 1 analysis", False, f"Exception: {e}")


def test_proper_ordering():
    """
    Dictionary with correct dependency ordering.
    Reference: Tasks file Test 2
    """
    print("=== Testing Proper Ordering ===")
    
    try:
        D = BytePairDictionary([
            ('a', 'b'),      # Creates 'ab'
            ('b', 'c'),      # Creates 'bc'
            ('a', 'bc'),     # Now 'bc' exists (from rule 1)
            ('ab', 'c')      # 'ab' exists (from rule 0)
        ])
        
        is_proper = is_proper_dictionary(D)
        deps = get_rule_dependencies(D)
        
        # Check expected dependencies
        rule_2_deps = deps.get(2, [])
        rule_3_deps = deps.get(3, [])
        
        expected_rule_2_deps = [1]  # Rule 2 depends on rule 1 for 'bc'
        expected_rule_3_deps = [0]  # Rule 3 depends on rule 0 for 'ab'
        
        deps_correct = (rule_2_deps == expected_rule_2_deps and 
                       rule_3_deps == expected_rule_3_deps)
        
        passed = (is_proper and deps_correct)
        log_test("Proper ordering validation", 
                passed,
                f"Proper: {is_proper}, Dependencies correct: {deps_correct}")
        log_test("Dependency details", 
                True,
                f"Rule 2 deps: {rule_2_deps}, Rule 3 deps: {rule_3_deps}")
        
    except Exception as e:
        log_test("Proper ordering", False, f"Exception: {e}")


def test_example_3_improper():
    """
    Example 3 from page 5 - improper dictionary.
    Reference: Tasks file Test 3
    """
    print("=== Testing Example 3 (Improper Dictionary) ===")
    
    try:
        D = BytePairDictionary([
            ('ab', 'a'),     # Creates 'aba' but no rule creates 'ab'!
            ('a', 'b')       # Creates 'ab' but comes after rule 0
        ])
        
        is_proper = is_proper_dictionary(D)
        
        passed = not is_proper
        log_test("Example 3 improper detection", 
                passed,
                f"Correctly detected as improper: {not is_proper}")
        
    except Exception as e:
        log_test("Example 3 improper", False, f"Exception: {e}")


def test_single_char_rules():
    """
    Dictionaries with only single-character merges are always proper.
    Reference: Tasks file Test 4
    """
    print("=== Testing Single Character Rules ===")
    
    try:
        D = BytePairDictionary([
            ('a', 'b'),
            ('c', 'd'),
            ('e', 'f')
        ])
        
        is_proper = is_proper_dictionary(D)
        deps = get_rule_dependencies(D)
        
        # All rules should have no dependencies
        no_dependencies = all(len(deps[i]) == 0 for i in range(len(D)))
        
        passed = (is_proper and no_dependencies)
        log_test("Single char rules are proper", 
                passed,
                f"Proper: {is_proper}, No dependencies: {no_dependencies}")
        
    except Exception as e:
        log_test("Single char rules", False, f"Exception: {e}")


def test_chain_dependencies():
    """
    Test long dependency chains.
    Reference: Tasks file Test 5
    """
    print("=== Testing Chain Dependencies ===")
    
    try:
        D = BytePairDictionary([
            ('a', 'b'),      # Creates 'ab'
            ('ab', 'c'),     # Creates 'abc'
            ('abc', 'd'),    # Creates 'abcd'
            ('abcd', 'e')    # Creates 'abcde'
        ])
        
        is_proper = is_proper_dictionary(D)
        deps = get_rule_dependencies(D)
        
        # Check chain dependencies
        expected_deps = {
            0: [],
            1: [0],
            2: [1], 
            3: [2]
        }
        
        deps_correct = (deps == expected_deps)
        
        passed = (is_proper and deps_correct)
        log_test("Chain dependencies", 
                passed,
                f"Proper: {is_proper}, Dependencies: {deps}")
        
    except Exception as e:
        log_test("Chain dependencies", False, f"Exception: {e}")


def test_invalid_construction():
    """
    Dictionary that's proper but couldn't be trained naturally.
    Reference: Tasks file Test 6
    """
    print("=== Testing Invalid Construction ===")
    
    try:
        D = BytePairDictionary([
            ('b', 'c'),      # Creates 'bc'
            ('a', 'b'),      # Creates 'ab'
            ('c', 'd'),      # Creates 'cd'
            ('ab', 'cd')     # Proper but can't apply (a≀b and c≀d never adjacent)
        ])
        
        corpus = "abcd"  # In this corpus, 'ab' and 'cd' are never adjacent
        result = validate_training_construction(D, corpus)
        
        is_proper = result['is_proper']
        all_useful = result['all_rules_useful']
        
        # Dictionary should be proper but not all rules useful
        passed = (is_proper and not all_useful)
        log_test("Proper but invalid construction", 
                passed,
                f"Proper: {is_proper}, All useful: {all_useful}")
        
    except Exception as e:
        log_test("Invalid construction", False, f"Exception: {e}")


def test_comprehensive_analysis():
    """Test the comprehensive analyze_dictionary_properties function."""
    print("=== Testing Comprehensive Analysis ===")
    
    try:
        D = BytePairDictionary([
            ('a', 'b'),      # Creates 'ab'
            ('b', 'c'),      # Creates 'bc'
            ('ab', 'c')      # Uses 'ab' from rule 0
        ])
        
        analysis = analyze_dictionary_properties(D, "abc")
        
        # Check that all expected keys are present
        expected_keys = [
            'is_proper', 'dictionary_size', 'rules_with_dependencies',
            'max_dependency_chain_length', 'dependency_graph', 'unused_rules',
            'construction_validation', 'total_token_length'
        ]
        
        has_all_keys = all(key in analysis for key in expected_keys)
        
        passed = has_all_keys
        log_test("Comprehensive analysis structure", 
                passed,
                f"Has all expected keys: {has_all_keys}")
        
        if has_all_keys:
            log_test("Analysis details", 
                    True,
                    f"Proper: {analysis['is_proper']}, Size: {analysis['dictionary_size']}")
        
    except Exception as e:
        log_test("Comprehensive analysis", False, f"Exception: {e}")


def run_all_tests():
    """Run all test suites."""
    print("BPE Tokenization Library - Validation Tests")
    print("=" * 55)
    print()
    
    test_is_proper_dictionary_functionality()
    test_get_rule_dependencies_functionality()
    test_validate_training_construction_functionality()
    test_example_1_proper()
    test_proper_ordering()
    test_example_3_improper()
    test_single_char_rules()
    test_chain_dependencies()
    test_invalid_construction()
    test_comprehensive_analysis()
    
    print("=" * 55)
    print("All validation tests completed!")


if __name__ == "__main__":
    run_all_tests()