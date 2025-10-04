"""
Base tokenizer implementation for BPE tokenization.

Implements the foundational tokenization operations from "Formalizing BPE Tokenization":
- Rule application (⇒ operator)
- Finding applicable rules
- Non-deterministic base tokenization
- Terminal state checking

Reference: Definition 2, Page 3
"""

from typing import List, Tuple, Set, Optional
from collections import deque
from .core import Tokenization, BytePairDictionary


def apply_rule(tokenization: Tokenization, rule: Tuple[str, str], position: int) -> Tokenization:
    """
    Apply a single rule u≀v at a specific position in the tokenization.
    
    Reference: Page 3 - "τ ⇒D τ' if τ = φ≀u≀v≀φ' and τ' = φ≀uv≀φ'"
    
    This implements the single-step derivation operation where we merge
    two adjacent tokens u and v into a single token uv.
    
    Args:
        tokenization: Current tokenization τ
        rule: Tuple (u, v) representing the merge rule u≀v
        position: Index where u≀v occurs (u is at position, v at position+1)
    
    Returns:
        New tokenization with rule applied: τ' = φ≀uv≀φ'
        
    Raises:
        IndexError: If position is invalid
        ValueError: If rule cannot be applied at position
        
    Complexity: O(n) where n = |τ| (need to copy tokens)
    """
    if not isinstance(tokenization, Tokenization):
        raise ValueError("tokenization must be a Tokenization instance")
    
    if not isinstance(rule, tuple) or len(rule) != 2:
        raise ValueError("rule must be a tuple of length 2")
    
    u, v = rule
    tokens = tokenization.tokens
    
    if position < 0 or position >= len(tokens) - 1:
        raise IndexError(f"Invalid position {position} for tokenization of length {len(tokens)}")
    
    if tokens[position] != u or tokens[position + 1] != v:
        raise ValueError(f"Rule ({u}, {v}) cannot be applied at position {position}. "
                        f"Found tokens: ({tokens[position]}, {tokens[position + 1]})")
    
    new_tokens = (
        tokens[:position] +           # φ (prefix)
        [u + v] +                    # uv (merged token)
        tokens[position + 2:]        # φ' (suffix)
    )
    
    return Tokenization(new_tokens)


def find_applicable_rules(tokenization: Tokenization, dictionary: BytePairDictionary) -> List[Tuple[Tuple[str, str], int]]:
    """
    Find all positions where dictionary rules can be applied.
    
    Scans through all adjacent token pairs in the tokenization and checks
    if any dictionary rules match those pairs.
    
    Args:
        tokenization: Current tokenization to analyze
        dictionary: Dictionary containing merge rules
    
    Returns:
        List of (rule, position) tuples where each rule can be applied
        
    Complexity: O(|τ| × |D|) where |τ| is tokenization length, |D| is dictionary size
    Note: Can be optimized to O(|τ|) using hash map lookup
    """
    if not isinstance(tokenization, Tokenization):
        raise ValueError("tokenization must be a Tokenization instance")
    
    if not isinstance(dictionary, BytePairDictionary):
        raise ValueError("dictionary must be a BytePairDictionary instance")
    
    applicable_rules = []
    tokens = tokenization.tokens
    
    for i in range(len(tokens) - 1):
        u = tokens[i]
        v = tokens[i + 1]
        
        if dictionary.can_apply(u, v):
            rule = (u, v)
            applicable_rules.append((rule, i))
    
    return applicable_rules


def is_terminal(tokenization: Tokenization, dictionary: BytePairDictionary) -> bool:
    """
    Check if tokenization is terminal (no rules can be applied).
    
    Reference: Page 3 - "there exists no τp+1 such that τp ⇒ τp+1"
    
    A tokenization is terminal when no dictionary rules can be applied
    to any adjacent token pairs.
    
    Args:
        tokenization: Tokenization to check
        dictionary: Dictionary containing merge rules
    
    Returns:
        True if no rules can be applied (terminal state)
        
    Complexity: O(|τ| × |D|) or O(|τ|) with hash map optimization
    """
    applicable_rules = find_applicable_rules(tokenization, dictionary)
    return len(applicable_rules) == 0


def tokenize_base(w: str, dictionary: BytePairDictionary) -> List[Tokenization]:
    """
    Compute all possible base tokenizations T^D_base(w).
    
    Reference: Definition 2, Page 3
    
    This implements the non-deterministic base tokenizer that explores
    all possible sequences of rule applications starting from T∅(w).
    
    Process:
    - Start with τ0 = T∅(w) 
    - Apply rules non-deterministically until no rules apply
    - Return all possible final tokenizations
    
    Args:
        w: Input string to tokenize
        dictionary: Byte pair dictionary containing merge rules
    
    Returns:
        List of all valid base tokenizations (T^D_base(w))
        Each tokenization is terminal (no further rules apply)
        
    Complexity: Exponential in worst case - O(|D|^|w|)
    Can have exponentially many derivation paths in degenerate cases.
    
    Note: This is primarily for validation/testing. 
    Production systems use deterministic versions (SentencePiece/HuggingFace).
    """
    from .core import T_empty
    
    if not isinstance(w, str):
        raise ValueError("w must be a string")
    
    if not isinstance(dictionary, BytePairDictionary):
        raise ValueError("dictionary must be a BytePairDictionary instance")
    
    initial = T_empty(w)
    
    queue = deque([initial])
    visited = set()  # Track visited tokenizations to avoid duplicates
    terminal_tokenizations = []
    
    while queue:
        current_tokenization = queue.popleft()
        
        current_tuple = tuple(current_tokenization.tokens)
        if current_tuple in visited:
            continue
        visited.add(current_tuple)
        
        applicable_rules = find_applicable_rules(current_tokenization, dictionary)
        
        if not applicable_rules:
            terminal_tokenizations.append(current_tokenization)
        else:
            for rule, position in applicable_rules:
                try:
                    new_tokenization = apply_rule(current_tokenization, rule, position)
                    queue.append(new_tokenization)
                except (ValueError, IndexError):
                    continue
    
    unique_terminals = []
    seen_terminals = set()
    
    for tokenization in terminal_tokenizations:
        token_tuple = tuple(tokenization.tokens)
        if token_tuple not in seen_terminals:
            seen_terminals.add(token_tuple)
            unique_terminals.append(tokenization)
    
    return unique_terminals


def find_highest_priority_rule(tokenization: Tokenization, 
                               dictionary: BytePairDictionary) -> Optional[Tuple[Tuple[str, str], int]]:
    """
    Find the highest priority applicable rule and its leftmost position.
    
    Reference: Definition 3, Page 4
    Two-step selection process:
    1. Find highest priority rule that can apply anywhere in tokenization
    2. Among all positions where that highest priority rule applies, return leftmost
    
    This implements the core logic of SentencePiece tokenization:
    - Priority: "u≀v is the highest priority rule in D for which such a decomposition exists"
    - Leftmost: "we pick the unique one which minimizes |φ|"
    
    Args:
        tokenization: Current tokenization to analyze
        dictionary: Dictionary containing merge rules with priority ordering
    
    Returns:
        (rule, position) tuple for highest priority rule at leftmost position,
        or None if no rules apply
        
    Complexity: O(|τ| × |D|) naive implementation
    Optimization: O(|τ| × log|D|) with sorted rule lookup
    """
    if not isinstance(tokenization, Tokenization):
        raise ValueError("tokenization must be a Tokenization instance")
    
    if not isinstance(dictionary, BytePairDictionary):
        raise ValueError("dictionary must be a BytePairDictionary instance")
    
    tokens = tokenization.tokens
    
    if len(tokens) < 2:
        return None
    
    highest_priority = float('inf')  # Lower number = higher priority
    best_rule = None
    
    for i in range(len(tokens) - 1):
        u = tokens[i]
        v = tokens[i + 1]
        
        if dictionary.can_apply(u, v):
            priority = dictionary.get_priority(u, v)
            if priority < highest_priority:
                highest_priority = priority
                best_rule = (u, v)
    
    if best_rule is None:
        return None
    
    for i in range(len(tokens) - 1):
        u = tokens[i]
        v = tokens[i + 1]
        
        if (u, v) == best_rule:
            return (best_rule, i)
    
    return None


def tokenize_sentencepiece(w: str, dictionary: BytePairDictionary) -> Tokenization:
    """
    Compute the SentencePiece tokenization T^D(w) - the "correct" tokenization.
    
    Reference: Definition 3, Page 4
    
    This is the main production tokenizer - deterministic, efficient, and matches 
    the SentencePiece library behavior.
    
    Algorithm:
    1. Start with τ0 = T∅(w) (character-level tokenization)
    2. Repeat until no rules apply:
       a. Find the HIGHEST PRIORITY rule that can be applied anywhere
       b. Among all positions where this rule applies, choose the LEFTMOST
       c. Apply the rule at that position
    3. Return final tokenization
    
    Args:
        w: Input string to tokenize
        dictionary: Byte pair dictionary with priority ordering
    
    Returns:
        The unique correct tokenization τn
        
    Complexity: O(|w|² × |D|) worst case
    - At most |w| merge operations (each reduces token count by 1)
    - Each iteration scans O(|w|) positions 
    - Each position checks against O(|D|) rules
    
    Optimization: O(|w|² × log|D|) with priority queue/hash map
    """
    from .core import T_empty
    
    if not isinstance(w, str):
        raise ValueError("w must be a string")
    
    if not isinstance(dictionary, BytePairDictionary):
        raise ValueError("dictionary must be a BytePairDictionary instance")
    
    current_tokenization = T_empty(w)
    
    while True:
        rule_and_position = find_highest_priority_rule(current_tokenization, dictionary)
        
        if rule_and_position is None:
            break
        
        rule, position = rule_and_position
        
        current_tokenization = apply_rule(current_tokenization, rule, position)
    
    return current_tokenization


def validate_sentencepiece(w: str, dictionary: BytePairDictionary) -> bool:
    """
    Verify that SentencePiece result is in T^D_base(w).
    
    Reference: Page 3-4, Definitions 2 & 3
    The deterministic tokenizer should select one result from all possible base tokenizations.
    
    This function validates that the SentencePiece algorithm correctly selects a valid
    tokenization from the set of all possible base tokenizations.
    
    Args:
        w: Input string to test
        dictionary: Dictionary to use for tokenization
    
    Returns:
        True if SentencePiece result is in the base tokenization set
        
    Complexity: Expensive - uses exponential base tokenizer
    Only use for testing/validation on small inputs (|w| <= 10 recommended)
    
    Warning: This function can be very slow for large inputs due to the exponential
    nature of the base tokenizer.
    """
    if not isinstance(w, str):
        raise ValueError("w must be a string")
    
    if not isinstance(dictionary, BytePairDictionary):
        raise ValueError("dictionary must be a BytePairDictionary instance")
    
    sentencepiece_result = tokenize_sentencepiece(w, dictionary)
    
    base_results = tokenize_base(w, dictionary)
    
    for base_result in base_results:
        if sentencepiece_result.tokens == base_result.tokens:
            return True
    
    return False


def apply_rule_exhaustively(tokenization: Tokenization, 
                           rule: Tuple[str, str]) -> Tokenization:
    """
    Apply a single rule from left to right until it no longer applies.
    
    Reference: Definition 4, Page 5 - "apply u≀v from left to right, 
    until it is no longer applicable"
    
    This function implements the core difference between HuggingFace and SentencePiece:
    instead of applying a rule once and re-evaluating, it exhaustively applies
    the same rule until no more applications are possible.
    
    Args:
        tokenization: Current tokenization
        rule: (u, v) pair to apply exhaustively
    
    Returns:
        Tokenization with rule applied as many times as possible
        
    Complexity: O(|τ|²) worst case
    - At most |τ| applications (each reduces length by 1)
    - Each application: O(|τ|) to find next occurrence and rebuild
    
    Optimization: O(|τ|) with in-place modification and position tracking
    """
    if not isinstance(tokenization, Tokenization):
        raise ValueError("tokenization must be a Tokenization instance")
    
    if not isinstance(rule, tuple) or len(rule) != 2:
        raise ValueError("rule must be a tuple of length 2")
    
    u, v = rule
    
    tokens = tokenization.tokens.copy()
    
    position = 0
    while position < len(tokens) - 1:
        if tokens[position] == u and tokens[position + 1] == v:
            tokens[position] = u + v
            del tokens[position + 1]
        else:
            position += 1
    
    return Tokenization(tokens)


def tokenize_huggingface(w: str, dictionary: BytePairDictionary) -> Tokenization:
    """
    Compute the HuggingFace tokenization T^D_hf(w).
    
    Reference: Definition 4, Page 4-5
    
    Algorithm Difference from SentencePiece:
    - SentencePiece: Apply highest priority rule once (leftmost), then re-evaluate
    - HuggingFace: Apply highest priority rule EXHAUSTIVELY (all occurrences left-to-right), 
                   then move to next priority rule
    
    This creates different behavior for improper dictionaries, but should match
    SentencePiece for proper dictionaries (Lemma 1, Page 5-6).
    
    Algorithm:
    1. Start with τ0 = T∅(w)
    2. For each rule in priority order (highest to lowest):
       a. Apply rule exhaustively from left to right until it no longer applies
       b. Move to next priority rule
    3. Return final tokenization
    
    Args:
        w: Input string
        dictionary: Byte pair dictionary
    
    Returns:
        HuggingFace tokenization τn
        
    Complexity: O(|w| × |D|²) worst case
    - Outer loop: |D| rules
    - Each rule: potentially |w| applications
    - Each application: O(|D|) to check if rule still applies (naive)
    
    Optimization: O(|w| × |D|) with careful position tracking
    """
    from .core import T_empty
    
    if not isinstance(w, str):
        raise ValueError("w must be a string")
    
    if not isinstance(dictionary, BytePairDictionary):
        raise ValueError("dictionary must be a BytePairDictionary instance")
    
    current_tokenization = T_empty(w)
    
    for rule in dictionary:
        current_tokenization = apply_rule_exhaustively(current_tokenization, rule)
    
    return current_tokenization


def compare_tokenizers(w: str, dictionary: BytePairDictionary) -> dict:
    """
    Compare SentencePiece and HuggingFace tokenization results.
    
    This function helps validate the theoretical result from Lemma 1 (Page 5-6):
    for proper dictionaries, both tokenizers should produce the same result.
    
    Args:
        w: Input string to tokenize
        dictionary: Dictionary to use for both tokenizers
    
    Returns:
        Dictionary with:
        - 'sentencepiece': SentencePiece result
        - 'huggingface': HuggingFace result  
        - 'match': Boolean indicating if they match
        - 'dictionary_proper': Whether dictionary is proper (requires validation.py)
        - 'input_string': The input string
        - 'dictionary_size': Number of rules in dictionary
    
    Reference: Lemma 1, Page 5-6 - They should match for proper dictionaries
    """
    if not isinstance(w, str):
        raise ValueError("w must be a string")
    
    if not isinstance(dictionary, BytePairDictionary):
        raise ValueError("dictionary must be a BytePairDictionary instance")
    
    sentencepiece_result = tokenize_sentencepiece(w, dictionary)
    huggingface_result = tokenize_huggingface(w, dictionary)
    
    match = (sentencepiece_result.tokens == huggingface_result.tokens)
    
    from .validation import is_proper_dictionary
    dictionary_proper = is_proper_dictionary(dictionary)
    
    return {
        'sentencepiece': sentencepiece_result,
        'huggingface': huggingface_result,
        'match': match,
        'dictionary_proper': dictionary_proper,
        'input_string': w,
        'dictionary_size': len(dictionary)
    }
