# tests/test_comprehensive_validation.py

import pytest
import random
import string
import time
from typing import List, Tuple

# Import your implementations
from bpe_tokenizer.core import Tokenization, BytePairDictionary, T_empty
from bpe_tokenizer.tokenizers import (
    tokenize_base,
    tokenize_sentencepiece,
    tokenize_huggingface,
    is_terminal
)

# ============================================
# TEST 1: Synthetic Validation (Small Known Cases)
# ============================================

def test_synthetic_single_path():
    """Base tokenizer with only one valid result."""
    D = BytePairDictionary([('a', 'b')])
    w = "ab"
    results = tokenize_base(w, D)

    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    assert results[0].tokens == ["ab"], f"Expected ['ab'], got {results[0].tokens}"
    print("✓ Test 1.1 passed: Single path tokenization")

def test_synthetic_multiple_paths():
    """Base tokenizer with multiple valid paths."""
    D = BytePairDictionary([('a', 'b'), ('b', 'c')])
    w = "abc"
    results = tokenize_base(w, D)

    # Possible results: ["ab", "c"] or ["a", "bc"]
    token_lists = [r.tokens for r in results]
    assert ["ab", "c"] in token_lists, "Missing ['ab', 'c']"
    assert ["a", "bc"] in token_lists, "Missing ['a', 'bc']"
    print(f"✓ Test 1.2 passed: Found {len(results)} valid tokenizations")

def test_synthetic_all_terminal():
    """Verify all base results are terminal."""
    D = BytePairDictionary([('a', 'b'), ('c', 'd')])
    w = "abcd"
    results = tokenize_base(w, D)

    for i, result in enumerate(results):
        assert is_terminal(result, D), f"Result {i} is not terminal: {result.tokens}"
    print(f"✓ Test 1.3 passed: All {len(results)} results are terminal")

# ============================================
# TEST 2: Cross-Validation (Deterministic ∈ Base)
# ============================================

def test_cross_validation_paper_examples():
    """Validate paper examples: SP and HF results must be in base set."""

    # Example 1 (Page 4)
    D1 = BytePairDictionary([
        ('a', 'b'),
        ('a', 'bc'),
        ('b', 'c'),
        ('ab', 'c')
    ])
    w1 = "abcbcab"

    base1 = tokenize_base(w1, D1)
    sp1 = tokenize_sentencepiece(w1, D1)
    hf1 = tokenize_huggingface(w1, D1)

    assert sp1 in base1, f"SP result {sp1.tokens} not in base set"
    assert hf1 in base1, f"HF result {hf1.tokens} not in base set"
    print(f"✓ Test 2.1 passed: Example 1 validated ({len(base1)} base tokenizations)")

    # Example 3 (Page 5) - Improper dictionary
    D2 = BytePairDictionary([('ab', 'a'), ('a', 'b')])
    w2 = "abababab"

    base2 = tokenize_base(w2, D2)
    sp2 = tokenize_sentencepiece(w2, D2)
    hf2 = tokenize_huggingface(w2, D2)

    assert sp2 in base2, f"SP result {sp2.tokens} not in base set"
    assert hf2 in base2, f"HF result {hf2.tokens} not in base set"
    assert sp2 != hf2, "SP and HF should differ for improper dictionary!"
    print(f"✓ Test 2.2 passed: Example 3 validated (SP ≠ HF as expected)")

def test_cross_validation_systematic():
    """Systematic cross-validation on various dictionaries."""
    test_cases = [
        # (rules, string, description)
        ([('a', 'b')], "ababab", "Simple repetition"),
        ([('a', 'b'), ('b', 'c')], "abcabc", "Two rules"),
        ([('a', 'b'), ('c', 'd'), ('ab', 'cd')], "abcd", "Chain dependency"),
    ]

    for rules, w, desc in test_cases:
        D = BytePairDictionary(rules)
        base = tokenize_base(w, D)
        sp = tokenize_sentencepiece(w, D)
        hf = tokenize_huggingface(w, D)

        assert sp in base, f"{desc}: SP not in base"
        assert hf in base, f"{desc}: HF not in base"
        print(f"✓ Test 2.3 passed: {desc}")

# ============================================
# TEST 3: Property-Based Testing (Random Cases)
# ============================================

def generate_random_proper_dictionary(num_rules: int = 5, alphabet: str = 'abc'):
    """Generate random dictionary with only single-char merges (always proper)."""
    rules = []
    chars = list(alphabet)

    for _ in range(num_rules):
        a, b = random.sample(chars, 2)
        rules.append((a, b))

    return BytePairDictionary(rules)

def test_property_based_random():
    """Property: Deterministic results always in base set."""
    num_tests = 50  # Adjust based on performance
    failures = []

    for i in range(num_tests):
        D = generate_random_proper_dictionary(num_rules=random.randint(2, 6))
        w = ''.join(random.choices('abc', k=random.randint(5, 10)))

        try:
            base = tokenize_base(w, D)
            sp = tokenize_sentencepiece(w, D)
            hf = tokenize_huggingface(w, D)

            assert sp in base, f"Test {i}: SP not in base"
            assert hf in base, f"Test {i}: HF not in base"
        except AssertionError as e:
            failures.append((i, str(e), D, w))

    if failures:
        print(f"✗ Test 3 FAILED: {len(failures)}/{num_tests} cases failed")
        for i, msg, D, w in failures[:3]:  # Show first 3 failures
            print(f"  Case {i}: {msg}")
            print(f"    Dict: {D.rules}, String: '{w}'")
        assert False, f"{len(failures)} property-based tests failed"

    print(f"✓ Test 3 passed: All {num_tests} random cases validated")

# ============================================
# TEST 4: Real Dictionary Sample
# ============================================

def test_real_dictionary_gpt2_sample():
    """Test with GPT-2-style dictionary structure."""
    # Simplified GPT-2 merge rules (Ġ = space)
    D = BytePairDictionary([
        ("Ġ", "t"),
        ("Ġ", "a"),
        ("h", "e"),
        ("i", "n"),
        ("r", "e"),
        ("Ġt", "he"),
        ("o", "n"),
        ("Ġa", "n"),
    ])

    test_cases = [
        ("Ġthe", ["Ġthe"]),          # Should merge to " the"
        ("Ġan", ["Ġan"]),            # Should merge to " an"
        ("hein", ["he", "in"]),      # "he" and "in"
    ]

    for w, expected in test_cases:
        base = tokenize_base(w, D)
        sp = tokenize_sentencepiece(w, D)

        assert sp in base, f"SP not in base for '{w}'"
        assert sp.tokens == expected, f"Expected {expected}, got {sp.tokens}"
        print(f"✓ Test 4 passed: '{w}' → {sp.tokens}")

# ============================================
# TEST 5: Exhaustiveness Validation
# ============================================

def test_exhaustiveness():
    """Verify base tokenizer finds ALL valid tokenizations."""
    D = BytePairDictionary([('a', 'b'), ('b', 'c')])
    w = "abc"

    # Manually enumerate all possible token combinations that are achievable
    # ["a", "b", "c"] - No merges, but non-terminal (can merge a+b or b+c)
    # ["ab", "c"] - Merged a≀b, terminal (ab cannot merge with c)
    # ["a", "bc"] - Merged b≀c, terminal (a cannot merge with bc)
    # ["abc"] - Not achievable: no rule creates "abc" token

    base_results = tokenize_base(w, D)
    base_token_lists = [r.tokens for r in base_results]

    # All base results should be terminal
    for result in base_results:
        assert is_terminal(result, D), f"Base result {result.tokens} is not terminal!"

    # Check expected results are present
    expected_results = [
        ["ab", "c"],   # Merged a≀b
        ["a", "bc"],   # Merged b≀c
    ]

    for expected in expected_results:
        assert expected in base_token_lists, f"Expected tokenization {expected} not found"
        print(f"  ✓ Found: {expected}")

    # Verify no unexpected results
    for result in base_token_lists:
        assert result in expected_results, f"Unexpected tokenization {result}"

    print(f"✓ Test 5 passed: Exhaustiveness validated ({len(base_results)} terminal tokenizations)")

# ============================================
# TEST 6: Performance Benchmark
# ============================================

def test_performance_benchmark():
    """Measure exponential growth of base tokenizer."""
    D = BytePairDictionary([('a', 'b'), ('b', 'c'), ('c', 'a')])

    results = []
    for length in [4, 6, 8, 10]:  # Keep small to avoid timeout
        w = 'abc' * (length // 3) + 'abc'[:length % 3]

        start = time.time()
        base = tokenize_base(w, D)
        elapsed = time.time() - start

        results.append({
            'length': length,
            'num_tokenizations': len(base),
            'time_ms': elapsed * 1000
        })

        print(f"  Length {length:2d}: {len(base):4d} tokenizations in {elapsed*1000:6.2f}ms")

    # Verify exponential growth
    for i in range(1, len(results)):
        prev_time = results[i-1]['time_ms']
        curr_time = results[i]['time_ms']
        assert curr_time > prev_time, "Time should increase with length"

    print(f"✓ Test 6 passed: Performance scales as expected")

# ============================================
# RUN ALL TESTS
# ============================================

def run_all_validation_tests():
    """Execute complete validation suite."""
    print("\n" + "="*60)
    print("COMPREHENSIVE BPE TOKENIZER VALIDATION")
    print("="*60 + "\n")

    tests = [
        ("Synthetic Validation", [
            test_synthetic_single_path,
            test_synthetic_multiple_paths,
            test_synthetic_all_terminal
        ]),
        ("Cross-Validation", [
            test_cross_validation_paper_examples,
            test_cross_validation_systematic
        ]),
        ("Property-Based Testing", [
            test_property_based_random
        ]),
        ("Real Dictionary Sample", [
            test_real_dictionary_gpt2_sample
        ]),
        ("Exhaustiveness", [
            test_exhaustiveness
        ]),
        ("Performance Benchmark", [
            test_performance_benchmark
        ]),
    ]

    total_passed = 0
    total_tests = 0

    for suite_name, suite_tests in tests:
        print(f"\n{'─'*60}")
        print(f"Running: {suite_name}")
        print(f"{'─'*60}")

        for test_func in suite_tests:
            total_tests += 1
            try:
                test_func()
                total_passed += 1
            except Exception as e:
                print(f"✗ {test_func.__name__} FAILED: {e}")

    print(f"\n{'='*60}")
    print(f"VALIDATION COMPLETE: {total_passed}/{total_tests} tests passed")
    print(f"{'='*60}\n")

    return total_passed == total_tests

if __name__ == "__main__":
    success = run_all_validation_tests()
    exit(0 if success else 1)
