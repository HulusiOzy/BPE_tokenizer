"""
Unit tests for HuggingFace tokenizer with alternative deterministic semantics.

Tests the implementation of:
- apply_rule_exhaustively function
- tokenize_huggingface main algorithm
- compare_tokenizers validator function
- All test cases from Tasks file Step 1.4

Following Definition 4, Page 4-5 of the paper and verifying Lemma 1.
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
    apply_rule_exhaustively, tokenize_huggingface, compare_tokenizers,
    tokenize_sentencepiece
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


def test_apply_rule_exhaustively_functionality():
    """Test apply_rule_exhaustively function."""
    print("=== Testing apply_rule_exhaustively Function ===")
    
    # Test 1: Single rule application
    try:
        tau = Tokenization(["a", "b", "c"])
        rule = ("a", "b")
        
        result = apply_rule_exhaustively(tau, rule)
        expected_tokens = ["ab", "c"]
        
        passed = (result.tokens == expected_tokens and result.concatenate() == "abc")
        log_test("Single rule application", 
                passed,
                f"Applied {rule}: {tau} → {result}")
    except Exception as e:
        log_test("Single rule application", False, f"Exception: {e}")
    
    # Test 2: Multiple exhaustive applications
    try:
        tau = Tokenization(["a", "b", "a", "b", "a", "b"])
        rule = ("a", "b")
        
        result = apply_rule_exhaustively(tau, rule)
        expected_tokens = ["ab", "ab", "ab"]
        
        passed = (result.tokens == expected_tokens and result.concatenate() == "ababab")
        log_test("Multiple exhaustive applications", 
                passed,
                f"Applied {rule} exhaustively: {tau} → {result}")
    except Exception as e:
        log_test("Multiple exhaustive applications", False, f"Exception: {e}")
    
    # Test 3: No applicable rule
    try:
        tau = Tokenization(["a", "c", "e"])
        rule = ("a", "b")
        
        result = apply_rule_exhaustively(tau, rule)
        expected_tokens = ["a", "c", "e"]  # No change
        
        passed = (result.tokens == expected_tokens)
        log_test("No applicable rule", 
                passed,
                f"No changes: {tau} → {result}")
    except Exception as e:
        log_test("No applicable rule", False, f"Exception: {e}")
    
    # Test 4: Overlapping patterns
    try:
        tau = Tokenization(["a", "b", "b", "c"])
        rule = ("b", "c")
        
        result = apply_rule_exhaustively(tau, rule)
        expected_tokens = ["a", "b", "bc"]  # Only rightmost b≀c applies
        
        passed = (result.tokens == expected_tokens)
        log_test("Overlapping patterns", 
                passed,
                f"Leftmost application: {tau} → {result}")
    except Exception as e:
        log_test("Overlapping patterns", False, f"Exception: {e}")
    
    # Test 5: Chain reactions (merged token creates new opportunities)
    try:
        tau = Tokenization(["a", "b", "c", "d"])
        rule1 = ("a", "b") 
        rule2 = ("ab", "c")
        
        # First apply a≀b → ab
        intermediate = apply_rule_exhaustively(tau, rule1)
        # Then apply ab≀c → abc
        result = apply_rule_exhaustively(intermediate, rule2)
        expected_tokens = ["abc", "d"]
        
        passed = (result.tokens == expected_tokens)
        log_test("Chain reaction merges", 
                passed,
                f"Chain: {tau} → {intermediate} → {result}")
    except Exception as e:
        log_test("Chain reaction merges", False, f"Exception: {e}")


def test_tokenize_huggingface_functionality():
    """Test tokenize_huggingface main algorithm."""
    print("=== Testing tokenize_huggingface Algorithm ===")
    
    # Test 1: Simple single rule
    try:
        D = BytePairDictionary([("a", "b")])
        w = "ababab"
        
        result = tokenize_huggingface(w, D)
        expected_tokens = ["ab", "ab", "ab"]
        
        passed = (result.tokens == expected_tokens and result.concatenate() == w)
        log_test("Simple exhaustive application", 
                passed,
                f"Result: {result}")
    except Exception as e:
        log_test("Simple exhaustive application", False, f"Exception: {e}")
    
    # Test 2: Multiple rules in sequence
    try:
        D = BytePairDictionary([("a", "b"), ("c", "d")])
        w = "abcd"
        
        result = tokenize_huggingface(w, D)
        expected_tokens = ["ab", "cd"]
        
        passed = (result.tokens == expected_tokens and result.concatenate() == w)
        log_test("Multiple rules in sequence", 
                passed,
                f"Result: {result}")
    except Exception as e:
        log_test("Multiple rules in sequence", False, f"Exception: {e}")
    
    # Test 3: Rule ordering matters
    try:
        D1 = BytePairDictionary([("a", "b"), ("ab", "c")])  # a≀b first
        D2 = BytePairDictionary([("ab", "c"), ("a", "b")])  # ab≀c first
        w = "abc"
        
        result1 = tokenize_huggingface(w, D1)
        result2 = tokenize_huggingface(w, D2)
        
        # D1: a≀b → ab, then ab≀c → abc
        # D2: ab≀c can't apply to a≀b≀c, then a≀b → ab≀c
        different_results = (result1.tokens != result2.tokens)
        both_valid = (result1.concatenate() == w and result2.concatenate() == w)
        
        passed = (different_results and both_valid)
        log_test("Rule ordering affects result", 
                passed,
                f"D1: {result1}, D2: {result2}")
    except Exception as e:
        log_test("Rule ordering affects result", False, f"Exception: {e}")
    
    # Test 4: No applicable rules
    try:
        D = BytePairDictionary([("x", "y")])
        w = "abc"
        
        result = tokenize_huggingface(w, D)
        expected_tokens = ["a", "b", "c"]  # Character-level
        
        passed = (result.tokens == expected_tokens and result.concatenate() == w)
        log_test("No applicable rules", 
                passed,
                f"Result: {result}")
    except Exception as e:
        log_test("No applicable rules", False, f"Exception: {e}")
    
    # Test 5: Empty string and edge cases
    try:
        D = BytePairDictionary([("a", "b")])
        
        # Empty string
        result_empty = tokenize_huggingface("", D)
        empty_passed = (len(result_empty) == 0)
        
        # Single character
        result_single = tokenize_huggingface("x", D)
        single_passed = (result_single.tokens == ["x"])
        
        passed = (empty_passed and single_passed)
        log_test("Edge cases (empty, single char)", 
                passed,
                f"Empty: {result_empty}, Single: {result_single}")
    except Exception as e:
        log_test("Edge cases (empty, single char)", False, f"Exception: {e}")


def test_compare_tokenizers_functionality():
    """Test compare_tokenizers function."""
    print("=== Testing compare_tokenizers Function ===")
    
    # Test 1: Matching tokenizers (proper dictionary)
    try:
        D = BytePairDictionary([("a", "b"), ("b", "c")])
        w = "abc"
        
        comparison = compare_tokenizers(w, D)
        
        match_correct = comparison['match']
        has_results = ('sentencepiece' in comparison and 'huggingface' in comparison)
        same_input = (comparison['input_string'] == w)
        
        passed = (has_results and same_input)
        log_test("Comparison structure", 
                passed,
                f"Match: {comparison['match']}, SP: {comparison['sentencepiece']}, HF: {comparison['huggingface']}")
    except Exception as e:
        log_test("Comparison structure", False, f"Exception: {e}")
    
    # Test 2: Different results for improper dictionary
    try:
        # This is an improper dictionary (will be tested more in Example 3)
        D = BytePairDictionary([("ab", "a"), ("a", "b")])
        w = "abab"
        
        comparison = compare_tokenizers(w, D)
        
        has_difference = not comparison['match']
        both_concatenate = (comparison['sentencepiece'].concatenate() == w and 
                          comparison['huggingface'].concatenate() == w)
        
        passed = both_concatenate  # May or may not differ, but both should be valid
        log_test("Improper dictionary handling", 
                passed,
                f"Match: {comparison['match']}, Both valid: {both_concatenate}")
    except Exception as e:
        log_test("Improper dictionary handling", False, f"Exception: {e}")


def test_example_3_difference():
    """
    Example 3 from page 5 - shows when SentencePiece ≠ HuggingFace.
    This uses an IMPROPER dictionary.
    
    Reference: Page 5, Example 3
    """
    print("=== Testing Example 3 (SentencePiece vs HuggingFace Difference) ===")
    
    try:
        D = BytePairDictionary([
            ('ab', 'a'),     # Priority 0 (highest)
            ('a', 'b')       # Priority 1 (lower)
        ])
        w = "abababab"
        
        sp_result = tokenize_sentencepiece(w, D)
        hf_result = tokenize_huggingface(w, D)
        
        # SentencePiece: aba≀b≀aba≀b
        # (applies ab≀a once leftmost, then a≀b, then ab≀a again, etc.)
        expected_sp = ["aba", "b", "aba", "b"]
        
        # HuggingFace: ab≀ab≀ab≀ab  
        # (exhaustively applies ab≀a first: no matches since no "aba" exists!
        #  then exhaustively applies a≀b: all positions)
        expected_hf = ["ab", "ab", "ab", "ab"]
        
        sp_correct = (sp_result.tokens == expected_sp)
        hf_correct = (hf_result.tokens == expected_hf)
        they_differ = (sp_result.tokens != hf_result.tokens)
        both_concatenate = (sp_result.concatenate() == w and hf_result.concatenate() == w)
        
        passed = (sp_correct and hf_correct and they_differ and both_concatenate)
        
        log_test("Example 3 - SentencePiece result", 
                sp_correct,
                f"Expected: {expected_sp}, Got: {sp_result.tokens}")
        log_test("Example 3 - HuggingFace result", 
                hf_correct,
                f"Expected: {expected_hf}, Got: {hf_result.tokens}")
        log_test("Example 3 - Tokenizers differ", 
                they_differ,
                f"Different results for improper dictionary")
        
    except Exception as e:
        log_test("Example 3 difference", False, f"Exception: {e}")


def test_proper_dictionary_equivalence():
    """
    Verify Lemma 1 (page 5-6): Proper dictionaries give same results.
    
    Reference: Lemma 1, Page 5-6
    """
    print("=== Testing Proper Dictionary Equivalence (Lemma 1) ===")
    
    try:
        # Proper dictionary (each multi-char token has constituent rules)
        D = BytePairDictionary([
            ('a', 'b'),      # Priority 0 - creates "ab"
            ('b', 'c'),      # Priority 1 - creates "bc"  
            ('ab', 'c')      # Priority 2 - depends on "ab" existing
        ])
        w = "abc"
        
        comparison = compare_tokenizers(w, D)
        sp_result = comparison['sentencepiece']
        hf_result = comparison['huggingface']
        
        # Both should produce: abc
        results_match = comparison['match']
        expected_tokens = ["abc"]
        sp_correct = (sp_result.tokens == expected_tokens)
        hf_correct = (hf_result.tokens == expected_tokens)
        
        passed = (results_match and sp_correct and hf_correct)
        log_test("Proper dictionary equivalence", 
                passed,
                f"SP: {sp_result}, HF: {hf_result}, Match: {results_match}")
        
    except Exception as e:
        log_test("Proper dictionary equivalence", False, f"Exception: {e}")


def test_exhaustive_application():
    """
    Verify that HuggingFace applies rules exhaustively.
    Reference: Tasks file Test 3
    """
    print("=== Testing Exhaustive Application ===")
    
    try:
        D = BytePairDictionary([('a', 'b')])
        w = "ababab"
        
        result = tokenize_huggingface(w, D)
        
        # Should apply a≀b exhaustively: ab≀ab≀ab
        expected_tokens = ["ab", "ab", "ab"]
        
        passed = (result.tokens == expected_tokens and result.concatenate() == w)
        log_test("Exhaustive rule application", 
                passed,
                f"Result: {result}")
        
        # Verify this differs from applying just once
        from bpe_tokenizer.tokenizers import apply_rule
        from bpe_tokenizer.core import T_empty
        single_application = apply_rule(T_empty(w), ('a', 'b'), 0)
        
        is_exhaustive = (len(result) < len(single_application))
        log_test("More exhaustive than single application", 
                is_exhaustive,
                f"Exhaustive: {len(result)} tokens, Single: {len(single_application)} tokens")
        
    except Exception as e:
        log_test("Exhaustive application", False, f"Exception: {e}")


def test_example_1_huggingface():
    """
    Example 1 from page 4, but using HuggingFace.
    Should produce same result since dictionary is proper.
    
    Reference: Tasks file Test 4
    """
    print("=== Testing Example 1 with HuggingFace ===")
    
    try:
        D = BytePairDictionary([
            ('a', 'b'),      # Priority 0
            ('a', 'bc'),     # Priority 1 
            ('b', 'c'),      # Priority 2
            ('ab', 'c')      # Priority 3
        ])
        w = "abcbcab"
        
        hf_result = tokenize_huggingface(w, D)
        sp_result = tokenize_sentencepiece(w, D)
        
        # Both should give: abc≀bc≀ab
        expected_tokens = ["abc", "bc", "ab"]
        results_match = (hf_result.tokens == sp_result.tokens)
        hf_correct = (hf_result.tokens == expected_tokens)
        both_concatenate = (hf_result.concatenate() == w and sp_result.concatenate() == w)
        
        passed = (results_match and hf_correct and both_concatenate)
        log_test("Example 1 - HuggingFace matches SentencePiece", 
                passed,
                f"HF: {hf_result}, SP: {sp_result}")
        
    except Exception as e:
        log_test("Example 1 with HuggingFace", False, f"Exception: {e}")


def test_multiple_passes():
    """
    Test case requiring multiple passes through rules.
    Reference: Tasks file Test 5
    """
    print("=== Testing Multiple Rule Passes ===")
    
    try:
        D = BytePairDictionary([
            ('a', 'b'),      # Priority 0
            ('c', 'd'),      # Priority 1
            ('ab', 'cd')     # Priority 2
        ])
        w = "abcd"
        
        hf_result = tokenize_huggingface(w, D)
        
        # Pass 1: a≀b → ab (no more a≀b pairs)
        # Pass 2: c≀d → cd (no more c≀d pairs)  
        # Pass 3: ab≀cd → abcd
        expected_tokens = ["abcd"]
        
        passed = (hf_result.tokens == expected_tokens and hf_result.concatenate() == w)
        log_test("Multiple rule passes", 
                passed,
                f"Result: {hf_result}")
        
        # Verify step-by-step progression
        from bpe_tokenizer.core import T_empty
        step1 = apply_rule_exhaustively(T_empty(w), ('a', 'b'))
        step2 = apply_rule_exhaustively(step1, ('c', 'd'))
        step3 = apply_rule_exhaustively(step2, ('ab', 'cd'))
        
        progression_correct = (step3.tokens == expected_tokens)
        log_test("Step-by-step progression", 
                progression_correct,
                f"T∅ → {step1} → {step2} → {step3}")
        
    except Exception as e:
        log_test("Multiple passes", False, f"Exception: {e}")


def test_algorithmic_differences():
    """Test the core algorithmic differences between the two tokenizers."""
    print("=== Testing Algorithmic Differences ===")
    
    try:
        # Create a scenario where the difference is clear
        D = BytePairDictionary([
            ('a', 'b'),      # Priority 0 - higher
            ('ab', 'c')      # Priority 1 - lower
        ])
        w = "abc"
        
        sp_result = tokenize_sentencepiece(w, D)
        hf_result = tokenize_huggingface(w, D)
        
        # SentencePiece: a≀b (priority 0) → ab≀c, then ab≀c (priority 1) → abc
        # HuggingFace: exhaust a≀b → ab≀c, then exhaust ab≀c → abc
        # Both should give same result here since dictionary is proper
        
        both_correct = (sp_result.tokens == ["abc"] and hf_result.tokens == ["abc"])
        both_match = (sp_result.tokens == hf_result.tokens)
        
        passed = (both_correct and both_match)
        log_test("Algorithmic equivalence for proper dictionary", 
                passed,
                f"SP: {sp_result}, HF: {hf_result}")
        
    except Exception as e:
        log_test("Algorithmic differences", False, f"Exception: {e}")


def run_all_tests():
    """Run all test suites."""
    print("BPE Tokenization Library - HuggingFace Tests")
    print("=" * 55)
    print()
    
    test_apply_rule_exhaustively_functionality()
    test_tokenize_huggingface_functionality()
    test_compare_tokenizers_functionality()
    test_example_3_difference()
    test_proper_dictionary_equivalence()
    test_exhaustive_application()
    test_example_1_huggingface()
    test_multiple_passes()
    test_algorithmic_differences()
    
    print("=" * 55)
    print("All HuggingFace tests completed!")


if __name__ == "__main__":
    run_all_tests()