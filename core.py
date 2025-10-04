"""
Core data structures for BPE tokenization.

Implements the formal definitions from "Formalizing BPE Tokenization":
- Tokenization: sequence of non-empty strings (u1≀⋯≀un)
- BytePairDictionary: ordered sequence of merge rules with priority
- T_empty: initial character-level tokenization
"""

import numpy as np
from typing import List, Tuple, Union


class Tokenization:
    """
    Represents a tokenization as a sequence of non-empty strings.
    Reference: Page 2, notation section - "u1≀⋯≀un"
    
    A tokenization is a sequence of tokens where each token is a non-empty string
    from Σ+. The concatenation π(τ) reconstructs the original string.
    """
    
    def __init__(self, tokens: List[str]):
        """
        Initialize a tokenization from a list of tokens.
        
        Args:
            tokens: List of non-empty strings representing the tokens
            
        Raises:
            ValueError: If any token is empty (violates Σ+ constraint)
        """
        if not tokens:
            self._tokens = []
        else:
            for i, token in enumerate(tokens):
                if not isinstance(token, str):
                    raise ValueError(f"Token at index {i} must be a string, got {type(token)}")
                if not token:
                    raise ValueError(f"Token at index {i} is empty. All tokens must be non-empty (Σ+)")
            
            self._tokens = list(tokens)
    
    def __len__(self) -> int:
        """
        Returns |τ| - the number of tokens in the tokenization.
        """
        return len(self._tokens)
    
    def concatenate(self) -> str:
        """
        π(τ) - concatenate tokens to reconstruct the original string.
        
        Returns:
            The concatenated string formed by joining all tokens
        """
        return ''.join(self._tokens)
    
    def __repr__(self) -> str:
        """
        Display as u1≀u2≀...≀un using the ≀ separator from the paper.
        """
        if not self._tokens:
            return "∅"
        return '≀'.join(self._tokens)
    
    def __str__(self) -> str:
        """String representation same as repr."""
        return self.__repr__()
    
    def __eq__(self, other) -> bool:
        """
        Check equality of tokenizations.
        """
        if not isinstance(other, Tokenization):
            return False
        return self._tokens == other._tokens
    
    def __getitem__(self, index) -> str:
        """
        Allow indexing into the tokenization to get individual tokens.
        """
        return self._tokens[index]
    
    def __iter__(self):
        """
        Allow iteration over tokens.
        """
        return iter(self._tokens)
    
    @property
    def tokens(self) -> List[str]:
        """
        Get the list of tokens (read-only access).
        """
        return self._tokens.copy()


class BytePairDictionary:
    """
    Dictionary D = [u1≀v1, ..., un≀vn] with priority ordering.
    Reference: Definition 1, Page 3
    
    A byte pair dictionary is a sequence of rules where each rule is a pair of strings.
    Rules have implicit priority based on their index (lower index = higher priority).
    """
    
    def __init__(self, rules: List[Tuple[str, str]]):
        """
        Initialize dictionary with ordered rules.
        
        Args:
            rules: List of (u, v) pairs representing merge rules
            
        Raises:
            ValueError: If rules are not valid pairs of strings
        """
        validated_rules = []
        for i, rule in enumerate(rules):
            if not isinstance(rule, (tuple, list)) or len(rule) != 2:
                raise ValueError(f"Rule at index {i} must be a pair (tuple/list of length 2), got {rule}")
            
            u, v = rule
            if not isinstance(u, str) or not isinstance(v, str):
                raise ValueError(f"Rule at index {i} must contain strings, got ({type(u)}, {type(v)})")
            
            validated_rules.append((u, v))
        
        self._rules = validated_rules
        
        self._rule_to_priority = {(u, v): i for i, (u, v) in enumerate(self._rules)}
    
    def __len__(self) -> int:
        """
        Returns |D| - the number of rules in the dictionary.
        """
        return len(self._rules)
    
    def get_priority(self, u: str, v: str) -> int:
        """
        Returns priority index of rule u≀v, or -1 if rule not in dictionary.
        
        Args:
            u: First part of the rule
            v: Second part of the rule
            
        Returns:
            Priority index (0-based, lower = higher priority) or -1 if not found
        """
        return self._rule_to_priority.get((u, v), -1)
    
    def can_apply(self, u: str, v: str) -> bool:
        """
        Check if rule u≀v exists in dictionary.
        
        Args:
            u: First part of the rule
            v: Second part of the rule
            
        Returns:
            True if the rule exists in the dictionary
        """
        return (u, v) in self._rule_to_priority
    
    def __getitem__(self, index: int) -> Tuple[str, str]:
        """
        Get rule at given index.
        
        Args:
            index: Priority index of the rule
            
        Returns:
            The (u, v) rule pair at the given index
        """
        return self._rules[index]
    
    def __iter__(self):
        """
        Iterate over rules in priority order.
        """
        return iter(self._rules)
    
    def __repr__(self) -> str:
        """
        Display dictionary as [u1≀v1, u2≀v2, ...].
        """
        if not self._rules:
            return "[]"
        
        rule_strs = [f"{u}≀{v}" for u, v in self._rules]
        return "[" + ", ".join(rule_strs) + "]"
    
    def __str__(self) -> str:
        """String representation same as repr."""
        return self.__repr__()
    
    @property
    def rules(self) -> List[Tuple[str, str]]:
        """
        Get the list of rules (read-only access).
        """
        return self._rules.copy()


def T_empty(w: str) -> Tokenization:
    """
    Creates initial tokenization where each character is its own token.
    Reference: Page 3, Definition 2 - "T∅(w) = α1≀⋯≀αn for w = α1...αn"
    
    This function implements the base tokenization T∅(w) that splits a string
    into individual characters, creating the starting point for BPE merging.
    
    Args:
        w: Input string to tokenize
        
    Returns:
        Tokenization where each character is a separate token
        
    Raises:
        ValueError: If input string is empty
    """
    if not isinstance(w, str):
        raise ValueError(f"Input must be a string, got {type(w)}")
    
    if not w:
        return Tokenization([])
    
    char_tokens = list(w)
    return Tokenization(char_tokens)
