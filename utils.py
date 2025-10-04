"""
Utility functions for BPE tokenization analysis and debugging.

Provides helper functions for:
- Token statistics and analysis
- Dictionary statistics and analysis
- Visualization and formatting
- Tokenization comparison
"""

from typing import Dict, List, Tuple, Any
from bpe_tokenizer.core import Tokenization, BytePairDictionary
from bpe_tokenizer.validation import is_proper_dictionary


def get_token_statistics(tokenization: Tokenization) -> Dict[str, Any]:
    """
    Compute statistics about a tokenization.

    Args:
        tokenization: Tokenization to analyze

    Returns:
        {
            'num_tokens': int,
            'avg_token_length': float,
            'max_token_length': int,
            'min_token_length': int,
            'unique_tokens': int,
            'token_frequency': Dict[str, int]
        }
    """
    tokens = tokenization.tokens

    if not tokens:
        return {
            'num_tokens': 0,
            'avg_token_length': 0.0,
            'max_token_length': 0,
            'min_token_length': 0,
            'unique_tokens': 0,
            'token_frequency': {}
        }

    num_tokens = len(tokens)

    token_lengths = [len(token) for token in tokens]
    avg_token_length = sum(token_lengths) / len(token_lengths)
    max_token_length = max(token_lengths)
    min_token_length = min(token_lengths)

    token_frequency = {}
    for token in tokens:
        token_frequency[token] = token_frequency.get(token, 0) + 1

    unique_tokens = len(token_frequency)

    return {
        'num_tokens': num_tokens,
        'avg_token_length': avg_token_length,
        'max_token_length': max_token_length,
        'min_token_length': min_token_length,
        'unique_tokens': unique_tokens,
        'token_frequency': token_frequency
    }


def get_dictionary_statistics(dictionary: BytePairDictionary) -> Dict[str, Any]:
    """
    Compute statistics about a dictionary.

    Args:
        dictionary: Dictionary to analyze

    Returns:
        {
            'num_rules': int,
            'avg_rule_length': float,  # avg(|u| + |v|)
            'max_token_created': int,  # max(|uv|)
            'alphabet_size': int,  # unique characters
            'is_proper': bool
        }
    """
    rules = dictionary.rules

    if not rules:
        return {
            'num_rules': 0,
            'avg_rule_length': 0.0,
            'max_token_created': 0,
            'alphabet_size': 0,
        }

    num_rules = len(rules)

    rule_lengths = []
    tokens_created = []
    alphabet = set()

    for u, v in rules:
        rule_lengths.append(len(u) + len(v))
        tokens_created.append(len(u + v))
        alphabet.update(u)
        alphabet.update(v)

    avg_rule_length = sum(rule_lengths) / len(rule_lengths)
    max_token_created = max(tokens_created)
    alphabet_size = len(alphabet)

    proper = is_proper_dictionary(dictionary)

    return {
        'num_rules': num_rules,
        'avg_rule_length': avg_rule_length,
        'max_token_created': max_token_created,
        'alphabet_size': alphabet_size,
        'is_proper': proper
    }


def tokenization_to_string(tokenization: Tokenization, separator: str = "≀") -> str:
    """
    Convert tokenization to readable string with custom separator.

    Args:
        tokenization: Tokenization to convert
        separator: Separator to use between tokens (default: ≀)

    Returns:
        String representation with custom separator
    """
    tokens = tokenization.tokens
    if not tokens:
        return "∅"
    return separator.join(tokens)


def dictionary_to_table(dictionary: BytePairDictionary) -> str:
    """
    Format dictionary as table for printing.

    Args:
        dictionary: Dictionary to format

    Returns:
        Formatted table string

    Output:
        Priority | Rule  | Creates
        ---------|-------|--------
        0        | a≀b   | "ab"
        1        | b≀c   | "bc"
    """
    rules = dictionary.rules

    if not rules:
        return "Empty Dictionary"

    rows = []
    header = ["Priority", "Rule", "Creates"]
    rows.append(header)

    max_priority_width = max(len(str(len(rules) - 1)), len("Priority"))
    max_rule_width = max(len(f"{u}≀{v}") for u, v in rules)
    max_rule_width = max(max_rule_width, len("Rule"))
    max_creates_width = max(len(f'"{u}{v}"') for u, v in rules)
    max_creates_width = max(max_creates_width, len("Creates"))

    separator = "-" * max_priority_width + " | " + "-" * max_rule_width + " | " + "-" * max_creates_width

    header_str = (
        "Priority".ljust(max_priority_width) + " | " +
        "Rule".ljust(max_rule_width) + " | " +
        "Creates".ljust(max_creates_width)
    )

    table_lines = [header_str, separator]

    for i, (u, v) in enumerate(rules):
        priority = str(i).ljust(max_priority_width)
        rule = f"{u}≀{v}".ljust(max_rule_width)
        creates = f'"{u}{v}"'.ljust(max_creates_width)
        table_lines.append(f"{priority} | {rule} | {creates}")

    return "\n".join(table_lines)


def compare_tokenizations(tau1: Tokenization, tau2: Tokenization) -> Dict[str, Any]:
    """
    Compare two tokenizations of (presumably) the same string.

    Args:
        tau1: First tokenization
        tau2: Second tokenization

    Returns:
        {
            'same_tokens': bool,
            'same_length': bool,
            'edit_distance': int,  # Levenshtein distance on token lists
            'differences': List[Tuple[int, str, str]]  # (position, token1, token2)
        }
    """
    tokens1 = tau1.tokens
    tokens2 = tau2.tokens

    same_tokens = tokens1 == tokens2
    same_length = len(tokens1) == len(tokens2)

    edit_distance = _levenshtein_distance(tokens1, tokens2)

    differences = []
    max_len = max(len(tokens1), len(tokens2))

    for i in range(max_len):
        token1 = tokens1[i] if i < len(tokens1) else None
        token2 = tokens2[i] if i < len(tokens2) else None

        if token1 != token2:
            differences.append((i, token1, token2))

    return {
        'same_tokens': same_tokens,
        'same_length': same_length,
        'edit_distance': edit_distance,
        'differences': differences
    }


def _levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
    """
    Calculate Levenshtein distance between two sequences.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Edit distance between sequences
    """
    if len(seq1) == 0:
        return len(seq2)
    if len(seq2) == 0:
        return len(seq1)

    matrix = [[0] * (len(seq2) + 1) for _ in range(len(seq1) + 1)]

    for i in range(len(seq1) + 1):
        matrix[i][0] = i
    for j in range(len(seq2) + 1):
        matrix[0][j] = j

    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            if seq1[i-1] == seq2[j-1]:
                cost = 0
            else:
                cost = 1

            matrix[i][j] = min(
                matrix[i-1][j] + 1,      # deletion
                matrix[i][j-1] + 1,      # insertion
                matrix[i-1][j-1] + cost  # substitution
            )

    return matrix[len(seq1)][len(seq2)]
