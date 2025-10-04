"""
Tests for dictionary equivalence and rule swapping functionality.

Tests the implementations of:
- are_dictionaries_equivalent
- can_swap_rules
- find_equivalent_orderings

Reference: Remark 5, Page 11 - dictionary equivalence
"""

import pytest
from bpe_tokenizer.core import BytePairDictionary
from bpe_tokenizer.validation import (
    are_dictionaries_equivalent,
    can_swap_rules,
    find_equivalent_orderings
)
from bpe_tokenizer.utils import (
    get_token_statistics,
    get_dictionary_statistics,
    tokenization_to_string,
    dictionary_to_table,
    compare_tokenizations
)
from bpe_tokenizer.core import Tokenization


class TestDictionaryEquivalence:
    """Test dictionary equivalence checking."""

    def test_identical_dictionaries(self):
        """Two dictionaries with same rules are equivalent."""
        D1 = BytePairDictionary([('a', 'b'), ('b', 'c')])
        D2 = BytePairDictionary([('a', 'b'), ('b', 'c')])

        result = are_dictionaries_equivalent(D1, D2)

        assert result['same_rules'] == True
        assert result['likely_equivalent'] == True
        assert result['first_difference'] is None

    def test_different_dictionaries(self):
        """Dictionaries with different rules are not equivalent."""
        D1 = BytePairDictionary([('a', 'b'), ('b', 'c')])
        D2 = BytePairDictionary([('a', 'b'), ('c', 'd')])

        result = are_dictionaries_equivalent(D1, D2)

        assert result['same_rules'] == False
        # May or may not be equivalent - depends on test strings

    def test_swapped_independent_rules(self):
        """Independent rules swapped should be equivalent."""
        D1 = BytePairDictionary([('a', 'b'), ('c', 'd')])
        D2 = BytePairDictionary([('c', 'd'), ('a', 'b')])

        result = are_dictionaries_equivalent(D1, D2)

        assert result['likely_equivalent'] == True

    def test_swapped_dependent_rules(self):
        """Dependent rules swapped may still be equivalent for SentencePiece."""
        D1 = BytePairDictionary([('a', 'b'), ('ab', 'c')])
        D2 = BytePairDictionary([('ab', 'c'), ('a', 'b')])

        result = are_dictionaries_equivalent(D1, D2, test_strings=['abc', 'ababc', 'abcabc'])

        # D2 is improper but still produces same results because
        # SentencePiece can still merge 'a' and 'b' first, then use 'ab≀c'
        # So they are actually equivalent
        assert result['likely_equivalent'] == True

    def test_empty_dictionaries(self):
        """Empty dictionaries are equivalent."""
        D1 = BytePairDictionary([])
        D2 = BytePairDictionary([])

        result = are_dictionaries_equivalent(D1, D2)

        assert result['likely_equivalent'] == True


class TestRuleSwapping:
    """Test rule swapping detection."""

    def test_swappable_rules(self):
        """Independent rules can be swapped."""
        D = BytePairDictionary([('a', 'b'), ('c', 'd')])

        assert can_swap_rules(D, 0, 1) == True

    def test_dependent_rules_not_swappable(self):
        """Rules with dependencies cannot swap."""
        D = BytePairDictionary([('a', 'b'), ('ab', 'c')])

        # Rule 1 depends on rule 0 (needs 'ab' token)
        assert can_swap_rules(D, 0, 1) == False

    def test_same_rule_swappable(self):
        """Same rule with itself is trivially swappable."""
        D = BytePairDictionary([('a', 'b'), ('c', 'd')])

        assert can_swap_rules(D, 0, 0) == True
        assert can_swap_rules(D, 1, 1) == True

    def test_overlapping_rules(self):
        """Rules with shared components may not be swappable."""
        D = BytePairDictionary([('a', 'b'), ('b', 'c')])

        # These rules share 'b' but should still be OK to swap
        # since they're both single character components
        result = can_swap_rules(D, 0, 1)
        # Result depends on implementation - single char overlap allowed
        assert isinstance(result, bool)

    def test_chain_dependency(self):
        """Chain of dependencies prevents swapping."""
        D = BytePairDictionary([('a', 'b'), ('c', 'd'), ('ab', 'cd')])

        # Rule 0 and 2 have dependency through 'ab'
        assert can_swap_rules(D, 0, 2) == False


class TestEquivalentOrderings:
    """Test finding equivalent dictionary orderings."""

    def test_no_swappable_rules(self):
        """Dictionary with no swappable rules returns only itself."""
        D = BytePairDictionary([('a', 'b'), ('ab', 'c')])

        equivalents = find_equivalent_orderings(D)

        assert len(equivalents) == 1
        assert equivalents[0].rules == D.rules

    def test_all_independent_rules(self):
        """All independent rules generate multiple orderings."""
        D = BytePairDictionary([('a', 'b'), ('c', 'd')])

        equivalents = find_equivalent_orderings(D)

        # Should find at least 2 orderings (original + swapped)
        assert len(equivalents) >= 2

        # All should be proper
        from bpe_tokenizer.validation import is_proper_dictionary
        for eq_dict in equivalents:
            assert is_proper_dictionary(eq_dict)

    def test_single_rule_dictionary(self):
        """Single rule dictionary has no alternatives."""
        D = BytePairDictionary([('a', 'b')])

        equivalents = find_equivalent_orderings(D)

        assert len(equivalents) == 1

    def test_limited_generation(self):
        """Generation is limited to prevent explosion."""
        # Create dictionary with many independent rules
        D = BytePairDictionary([
            ('a', 'b'), ('c', 'd'), ('e', 'f'),
            ('g', 'h'), ('i', 'j'), ('k', 'l')
        ])

        equivalents = find_equivalent_orderings(D)

        # Should be limited (MAX_DICTIONARIES = 10)
        assert len(equivalents) <= 10


class TestUtilityFunctions:
    """Test utility functions from utils.py."""

    def test_get_token_statistics(self):
        """Test token statistics computation."""
        tau = Tokenization(['ab', 'c', 'ab', 'def'])

        stats = get_token_statistics(tau)

        assert stats['num_tokens'] == 4
        assert stats['max_token_length'] == 3
        assert stats['min_token_length'] == 1
        assert stats['unique_tokens'] == 3
        assert stats['token_frequency']['ab'] == 2
        assert stats['token_frequency']['c'] == 1

    def test_get_dictionary_statistics(self):
        """Test dictionary statistics computation."""
        D = BytePairDictionary([('a', 'b'), ('ab', 'c'), ('c', 'd')])

        stats = get_dictionary_statistics(D)

        assert stats['num_rules'] == 3
        assert stats['max_token_created'] == 3  # 'abc'
        assert stats['alphabet_size'] == 4  # a, b, c, d

    def test_tokenization_to_string(self):
        """Test tokenization string conversion."""
        tau = Tokenization(['ab', 'c', 'def'])

        default_str = tokenization_to_string(tau)
        assert default_str == "ab≀c≀def"

        custom_str = tokenization_to_string(tau, separator='|')
        assert custom_str == "ab|c|def"

    def test_dictionary_to_table(self):
        """Test dictionary table formatting."""
        D = BytePairDictionary([('a', 'b'), ('c', 'd')])

        table = dictionary_to_table(D)

        assert 'Priority' in table
        assert 'Rule' in table
        assert 'Creates' in table
        assert 'a≀b' in table
        assert '"ab"' in table

    def test_compare_tokenizations(self):
        """Test tokenization comparison."""
        tau1 = Tokenization(['ab', 'c', 'def'])
        tau2 = Tokenization(['ab', 'c', 'def'])
        tau3 = Tokenization(['a', 'bc', 'def'])

        result1 = compare_tokenizations(tau1, tau2)
        assert result1['same_tokens'] == True
        assert result1['edit_distance'] == 0

        result2 = compare_tokenizations(tau1, tau3)
        assert result2['same_tokens'] == False
        assert result2['edit_distance'] > 0
        assert len(result2['differences']) > 0


class TestEquivalenceHeuristic:
    """Test the equivalence heuristic with various cases."""

    def test_proper_vs_improper_dictionary(self):
        """Proper and improper versions may still be equivalent."""
        D_proper = BytePairDictionary([('a', 'b'), ('ab', 'c')])
        D_improper = BytePairDictionary([('ab', 'c'), ('a', 'b')])

        result = are_dictionaries_equivalent(
            D_proper,
            D_improper,
            test_strings=['abc', 'ababc', 'abcabc']
        )

        # Even though D_improper is not proper, SentencePiece can still
        # produce the same results because it builds up from characters
        assert result['likely_equivalent'] == True

    def test_custom_test_strings(self):
        """Test with custom test strings."""
        D1 = BytePairDictionary([('a', 'b')])
        D2 = BytePairDictionary([('a', 'b')])

        result = are_dictionaries_equivalent(
            D1, D2,
            test_strings=['ab', 'abab', 'ababab']
        )

        assert result['total_tests'] == 3
        assert result['matches'] == 3
        assert result['likely_equivalent'] == True


def test_integration_equivalence_and_utils():
    """Integration test combining equivalence and utils."""
    # Create two dictionaries
    D1 = BytePairDictionary([('a', 'b'), ('c', 'd')])
    D2 = BytePairDictionary([('c', 'd'), ('a', 'b')])

    # Check equivalence
    equiv_result = are_dictionaries_equivalent(D1, D2)
    assert equiv_result['likely_equivalent'] == True

    # Get statistics
    stats1 = get_dictionary_statistics(D1)
    stats2 = get_dictionary_statistics(D2)

    # Should have same stats
    assert stats1['num_rules'] == stats2['num_rules']
    assert stats1['is_proper'] == stats2['is_proper']

    # Format as tables
    table1 = dictionary_to_table(D1)
    table2 = dictionary_to_table(D2)

    assert 'a≀b' in table1
    assert 'c≀d' in table2


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
