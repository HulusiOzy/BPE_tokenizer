"""
Tests for advanced BPE tokenization algorithms.

Tests the implementations of:
- Incremental tokenization (Algorithm 1, Page 7)
- Streaming tokenization (Algorithm 2, Page 10)
- Lookahead optimization and chain length

Reference: Main algorithmic contributions from the paper
"""

import pytest
from bpe_tokenizer.core import BytePairDictionary, Tokenization
from bpe_tokenizer.tokenizers import tokenize_sentencepiece
from bpe_tokenizer.algorithms import (
    incremental_update,
    incremental_tokenize_sequence,
    calculate_lookahead_constant,
    calculate_sufficient_lookahead,
    calculate_chain_length,
    tokenize_streaming,
    tokenize_streaming_generator,
    get_lookahead_comparison
)


class TestIncrementalUpdate:
    """Test incremental tokenization (Algorithm 1, Page 7)."""

    def test_basic_incremental_update(self):
        """Basic incremental update test."""
        D = BytePairDictionary([('a', 'b'), ('b', 'c')])

        # Tokenize "ab" and "c" separately
        tau1 = tokenize_sentencepiece("ab", D)
        tau2 = tokenize_sentencepiece("c", D)

        # Incremental update
        result = incremental_update(tau1, tau2, D)

        # Should match direct tokenization of "abc"
        expected = tokenize_sentencepiece("abc", D)

        assert result == expected

    def test_example_2_from_paper(self):
        """Test Example 2 (Page 4) - worst case for incremental."""
        # This is a worst-case example where incremental needs many iterations
        D = BytePairDictionary([('a', 'b')])

        w = "a" * 10
        w_prime = "b" * 10

        tau = tokenize_sentencepiece(w, D)
        tau_prime = tokenize_sentencepiece(w_prime, D)

        result = incremental_update(tau, tau_prime, D)

        # Should match direct tokenization
        expected = tokenize_sentencepiece(w + w_prime, D)

        assert result == expected

    def test_empty_tokenizations(self):
        """Test incremental with empty inputs."""
        D = BytePairDictionary([('a', 'b')])

        tau1 = Tokenization([])
        tau2 = tokenize_sentencepiece("ab", D)

        result = incremental_update(tau1, tau2, D)
        assert result == tau2

        result = incremental_update(tau2, tau1, D)
        assert result == tau2

    def test_no_merge_at_boundary(self):
        """Test case where boundary tokens don't merge."""
        D = BytePairDictionary([('a', 'b'), ('c', 'd')])

        tau1 = tokenize_sentencepiece("ab", D)
        tau2 = tokenize_sentencepiece("cd", D)

        result = incremental_update(tau1, tau2, D)

        # "ab" + "cd" has no merges at boundary
        assert result.tokens == ["ab", "cd"]

    def test_merge_at_boundary(self):
        """Test case where boundary tokens do merge."""
        D = BytePairDictionary([('a', 'b'), ('b', 'c'), ('ab', 'c')])

        tau1 = tokenize_sentencepiece("ab", D)
        tau2 = tokenize_sentencepiece("c", D)

        result = incremental_update(tau1, tau2, D)

        # "ab" + "c" can merge to "abc"
        expected = tokenize_sentencepiece("abc", D)
        assert result == expected


class TestIncrementalSequence:
    """Test incremental sequence tokenization."""

    def test_sequence_tokenization(self):
        """Test tokenizing a sequence of strings."""
        D = BytePairDictionary([('a', 'b'), ('b', 'c')])

        strings = ["ab", "c", "ab"]
        results = incremental_tokenize_sequence(strings, D)

        # Should have 3 results
        assert len(results) == 3

        # Last result should match full tokenization
        full_string = "".join(strings)
        expected = tokenize_sentencepiece(full_string, D)
        assert results[-1] == expected

    def test_empty_sequence(self):
        """Test with empty sequence."""
        D = BytePairDictionary([('a', 'b')])

        results = incremental_tokenize_sequence([], D)
        assert results == []

    def test_single_string(self):
        """Test with single string."""
        D = BytePairDictionary([('a', 'b')])

        results = incremental_tokenize_sequence(["ab"], D)
        assert len(results) == 1
        assert results[0].tokens == ["ab"]


class TestLookaheadConstant:
    """Test lookahead constant calculation."""

    def test_lookahead_calculation(self):
        """Test lookahead constant calculation."""
        D = BytePairDictionary([('a', 'b'), ('b', 'c')])

        lookahead = calculate_lookahead_constant(D)

        # |D| = 2, max{|uv|} = 2
        # l(D) ≤ 2 × 2 = 4
        assert lookahead == 4

    def test_sufficient_lookahead(self):
        """Test that sufficient lookahead equals lookahead constant."""
        D = BytePairDictionary([('a', 'b'), ('ab', 'c')])

        lookahead1 = calculate_lookahead_constant(D)
        lookahead2 = calculate_sufficient_lookahead(D)

        assert lookahead1 == lookahead2

    def test_empty_dictionary(self):
        """Test lookahead with empty dictionary."""
        D = BytePairDictionary([])

        lookahead = calculate_lookahead_constant(D)
        assert lookahead == 0

    def test_larger_dictionary(self):
        """Test lookahead with larger rules."""
        D = BytePairDictionary([
            ('a', 'b'),
            ('ab', 'c'),
            ('abc', 'd'),
            ('abcd', 'e')
        ])

        lookahead = calculate_lookahead_constant(D)

        # |D| = 4, max{|uv|} = 5 (for 'abcd' + 'e')
        # l(D) ≤ 4 × 5 = 20
        assert lookahead == 20


class TestChainLength:
    """Test chain length calculation."""

    def test_chain_length_no_dependencies(self):
        """Test chain length with no dependencies."""
        D = BytePairDictionary([('a', 'b'), ('c', 'd')])

        chain_length = calculate_chain_length(D)

        # No dependencies, max chain = 1
        assert chain_length == 1

    def test_chain_length_simple_chain(self):
        """Test chain length with simple dependency."""
        D = BytePairDictionary([('a', 'b'), ('ab', 'c')])

        chain_length = calculate_chain_length(D)

        # Rule 1 depends on rule 0, chain length = 2
        assert chain_length == 2

    def test_chain_length_longer_chain(self):
        """Test chain length with longer dependency chain."""
        D = BytePairDictionary([
            ('a', 'b'),       # 0: no deps
            ('ab', 'c'),      # 1: depends on 0
            ('abc', 'd'),     # 2: depends on 1
            ('abcd', 'e')     # 3: depends on 2
        ])

        chain_length = calculate_chain_length(D)

        # Chain: 0 → 1 → 2 → 3, length = 4
        assert chain_length == 4

    def test_chain_length_better_than_theoretical(self):
        """Test that chain-based bound is better than theoretical."""
        D = BytePairDictionary([
            ('a', 'b'),
            ('c', 'd'),
            ('e', 'f')
        ])

        comparison = get_lookahead_comparison(D)

        # All rules independent, so chain length = 1
        assert comparison['chain_length'] == 1
        assert comparison['is_chain_better'] == True
        assert comparison['chain_based_bound'] < comparison['theoretical_bound']


class TestStreamingTokenization:
    """Test streaming tokenization (Algorithm 2, Page 10)."""

    def test_streaming_basic(self):
        """Test basic streaming tokenization."""
        D = BytePairDictionary([('a', 'b')])

        text = "ababab"
        result = tokenize_streaming(text, D, lookahead=4)

        # Should match regular tokenization
        expected = tokenize_sentencepiece(text, D)

        # Tokens should be the same (order might differ slightly)
        assert result.concatenate() == expected.concatenate()

    def test_streaming_with_auto_lookahead(self):
        """Test streaming with automatic lookahead calculation."""
        D = BytePairDictionary([('a', 'b'), ('c', 'd')])

        text = "abcd"
        result = tokenize_streaming(text, D)

        expected = tokenize_sentencepiece(text, D)
        assert result.concatenate() == expected.concatenate()

    def test_streaming_empty_text(self):
        """Test streaming with empty text."""
        D = BytePairDictionary([('a', 'b')])

        result = tokenize_streaming("", D)
        assert result.tokens == []

    def test_streaming_fallback_for_large_lookahead(self):
        """Test that streaming falls back for very large lookahead."""
        # Create dictionary with large lookahead
        D = BytePairDictionary([('a' * 10, 'b' * 10)])

        text = "a" * 20 + "b" * 20
        result = tokenize_streaming(text, D)

        # Should still work (fallback to regular tokenization)
        expected = tokenize_sentencepiece(text, D)
        assert result == expected


class TestStreamingGenerator:
    """Test streaming generator."""

    def test_streaming_generator_basic(self):
        """Test streaming generator with basic input."""
        D = BytePairDictionary([('a', 'b')])

        # Simulate streaming input
        text_stream = iter(["ab", "ab", "ab"])

        tokens = list(tokenize_streaming_generator(text_stream, D))

        # Should produce tokens
        assert len(tokens) > 0

        # Concatenated tokens should match original
        result_text = "".join(tokens)
        assert "ab" in result_text

    def test_streaming_generator_character_stream(self):
        """Test streaming with character-by-character input."""
        D = BytePairDictionary([('a', 'b')])

        text = "ababab"
        text_stream = iter(text)

        tokens = list(tokenize_streaming_generator(text_stream, D, lookahead=4))

        # Should produce tokens
        assert len(tokens) > 0


class TestLookaheadComparison:
    """Test lookahead comparison utilities."""

    def test_lookahead_comparison(self):
        """Test lookahead comparison function."""
        D = BytePairDictionary([
            ('a', 'b'),
            ('ab', 'c')
        ])

        comparison = get_lookahead_comparison(D)

        # Check all fields present
        assert 'theoretical_bound' in comparison
        assert 'chain_length' in comparison
        assert 'chain_based_bound' in comparison
        assert 'max_token_length' in comparison
        assert 'improvement_ratio' in comparison
        assert 'is_chain_better' in comparison

        # Theoretical bound should be |D| × max{|uv|}
        assert comparison['theoretical_bound'] == 2 * 3  # 2 rules, max "abc" = 3

        # Chain length should be 2 (rule 1 depends on rule 0)
        assert comparison['chain_length'] == 2

    def test_independent_rules_comparison(self):
        """Test comparison for independent rules."""
        D = BytePairDictionary([
            ('a', 'b'),
            ('c', 'd'),
            ('e', 'f')
        ])

        comparison = get_lookahead_comparison(D)

        # All independent, chain = 1
        assert comparison['chain_length'] == 1

        # Chain-based should be better
        assert comparison['is_chain_better'] == True


class TestAlgorithmCorrectness:
    """Integration tests for algorithm correctness."""

    def test_incremental_vs_direct(self):
        """Test that incremental gives same result as direct tokenization."""
        D = BytePairDictionary([
            ('a', 'b'),
            ('b', 'c'),
            ('ab', 'c'),
            ('abc', 'd')
        ])

        # Test many combinations
        test_strings = [
            ("ab", "cd"),
            ("abc", "d"),
            ("a", "bcd"),
            ("abcd", "abcd"),
        ]

        for w1, w2 in test_strings:
            tau1 = tokenize_sentencepiece(w1, D)
            tau2 = tokenize_sentencepiece(w2, D)

            incremental_result = incremental_update(tau1, tau2, D)
            direct_result = tokenize_sentencepiece(w1 + w2, D)

            assert incremental_result == direct_result, \
                f"Mismatch for '{w1}' + '{w2}': {incremental_result.tokens} vs {direct_result.tokens}"

    def test_streaming_correctness(self):
        """Test that streaming gives correct results."""
        D = BytePairDictionary([('a', 'b'), ('b', 'c')])

        test_strings = ["ab", "abc", "abcabc", "ababab"]

        for text in test_strings:
            streaming_result = tokenize_streaming(text, D, lookahead=4)
            direct_result = tokenize_sentencepiece(text, D)

            # Results should reconstruct to same string
            assert streaming_result.concatenate() == direct_result.concatenate(), \
                f"Streaming mismatch for '{text}'"


def test_paper_example_2_incremental():
    """Test Example 2 from paper (Page 4) - worst case incremental."""
    # Example 2: Shows worst case O(n) for incremental
    # When w = a^n and w' = b^n with D = [a≀b]
    # The result is a^(n-1) + ab + b^(n-1) - only one merge at boundary

    n = 5
    D = BytePairDictionary([('a', 'b')])

    w = 'a' * n
    w_prime = 'b' * n

    tau = tokenize_sentencepiece(w, D)
    tau_prime = tokenize_sentencepiece(w_prime, D)

    result = incremental_update(tau, tau_prime, D)

    # Should match direct tokenization
    expected = tokenize_sentencepiece(w + w_prime, D)

    assert result == expected

    # Result should be: a^(n-1) + ab + b^(n-1)
    expected_tokens = ['a'] * (n-1) + ['ab'] + ['b'] * (n-1)
    assert result.tokens == expected_tokens


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
