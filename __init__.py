"""
BPE Tokenizer Library

A Python implementation of Byte Pair Encoding (BPE) tokenization
following the formalization by Berglund & van der Merwe.

Implements both SentencePiece and HuggingFace tokenization algorithms.
"""

from .core import Tokenization, BytePairDictionary, T_empty
from .tokenizers import (
    apply_rule,
    find_applicable_rules,
    is_terminal,
    tokenize_base,
    find_highest_priority_rule,
    tokenize_sentencepiece,
    apply_rule_exhaustively,
    tokenize_huggingface,
    compare_tokenizers
)
from .validation import (
    is_proper_dictionary,
    get_rule_dependencies,
    validate_training_construction,
    find_unused_rules,
    analyze_dictionary_properties,
    are_dictionaries_equivalent,
    can_swap_rules,
    find_equivalent_orderings
)
from .algorithms import (
    incremental_update,
    incremental_tokenize_sequence,
    calculate_lookahead_constant,
    calculate_sufficient_lookahead,
    calculate_chain_length,
    tokenize_streaming,
    tokenize_streaming_generator,
    get_lookahead_comparison
)
from .utils import (
    get_token_statistics,
    get_dictionary_statistics,
    tokenization_to_string,
    dictionary_to_table,
    compare_tokenizations
)

__version__ = "1.0.0"
__all__ = [
    # Core
    "Tokenization",
    "BytePairDictionary",
    "T_empty",
    # Tokenizers
    "apply_rule",
    "find_applicable_rules",
    "is_terminal",
    "tokenize_base",
    "find_highest_priority_rule",
    "tokenize_sentencepiece",
    "apply_rule_exhaustively",
    "tokenize_huggingface",
    "compare_tokenizers",
    # Validation
    "is_proper_dictionary",
    "get_rule_dependencies",
    "validate_training_construction",
    "find_unused_rules",
    "analyze_dictionary_properties",
    "are_dictionaries_equivalent",
    "can_swap_rules",
    "find_equivalent_orderings",
    # Algorithms
    "incremental_update",
    "incremental_tokenize_sequence",
    "calculate_lookahead_constant",
    "calculate_sufficient_lookahead",
    "calculate_chain_length",
    "tokenize_streaming",
    "tokenize_streaming_generator",
    "get_lookahead_comparison",
    # Utils
    "get_token_statistics",
    "get_dictionary_statistics",
    "tokenization_to_string",
    "dictionary_to_table",
    "compare_tokenizations",
]
