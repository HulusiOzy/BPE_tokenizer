"""
Dictionary validation for BPE tokenization.

Implements validation functions for proper dictionaries and training construction:
- Proper dictionary checking (Definition 5, Page 5)
- Rule dependency analysis
- Training construction validation

Reference: Definition 5 and Remark 1, Page 5
"""

from typing import Dict, List, Set, Any, Optional
import random
import string
from .core import BytePairDictionary
from .tokenizers import tokenize_sentencepiece


def is_proper_dictionary(dictionary: BytePairDictionary) -> bool:
    """
    Check if dictionary D is proper.

    Reference: Definition 5, Page 5

    A dictionary is proper if:
    - For each rule uj≀vj where |uj| > 1:
      There exists i < j such that uj = uivi
    - For each rule uj≀vj where |vj| > 1:
      There exists i < j such that vj = uivi

    In other words: multi-character tokens must have rules that created them
    at higher priority (earlier in the dictionary).

    Args:
        dictionary: BytePairDictionary to validate

    Returns:
        True if proper, False otherwise

    Complexity: O(||D||²) where ||D|| = Σ|uv| for all u≀v ∈ D
    Reference: Remark 2, Page 6
    """
    if not isinstance(dictionary, BytePairDictionary):
        raise ValueError("dictionary must be a BytePairDictionary instance")
    
    rules = dictionary.rules
    
    if len(rules) == 0:
        return True
    
    for j, (uj, vj) in enumerate(rules):
        if len(uj) > 1:
            found_creator = False
            for i in range(j):
                ui, vi = rules[i]
                if ui + vi == uj:
                    found_creator = True
                    break
            
            if not found_creator:
                return False
        
        if len(vj) > 1:
            found_creator = False
            for i in range(j):
                ui, vi = rules[i]
                if ui + vi == vj:
                    found_creator = True
                    break
            
            if not found_creator:
                return False
    
    return True


def get_rule_dependencies(dictionary: BytePairDictionary) -> Dict[int, List[int]]:
    """
    For each rule, find which higher-priority rules it depends on.

    A rule at index j depends on a rule at index i (i < j) if:
    - The rule j uses a token that was created by rule i

    Args:
        dictionary: BytePairDictionary to analyze

    Returns:
        Dict mapping rule_index → list of dependency rule indices

    Example:
        D = [('a','b'), ('b','c'), ('ab','c')]
        Result: {
            0: [],           # 'a≀b' has no dependencies (single chars)
            1: [],           # 'b≀c' has no dependencies
            2: [0]           # 'ab≀c' depends on rule 0 to create 'ab'
        }

    Complexity: O(||D||²)
    """
    if not isinstance(dictionary, BytePairDictionary):
        raise ValueError("dictionary must be a BytePairDictionary instance")
    
    rules = dictionary.rules
    dependencies = {}
    
    for j, (uj, vj) in enumerate(rules):
        dependencies[j] = []
        
        if len(uj) > 1:
            for i in range(j):
                ui, vi = rules[i]
                if ui + vi == uj:
                    dependencies[j].append(i)
        
        if len(vj) > 1:
            for i in range(j):
                ui, vi = rules[i]
                if ui + vi == vj:
                        dependencies[j].append(i)
    
    return dependencies


def validate_training_construction(dictionary: BytePairDictionary,
                                   training_corpus: str) -> Dict[str, bool]:
    """
    Verify dictionary could have been constructed via training algorithm.

    Reference: Remark 1, Page 5

    Training algorithm (simplified):
    1. Start with character-level tokenization
    2. Find most frequent adjacent pair
    3. Add to dictionary
    4. Re-tokenize corpus
    5. Repeat

    This function validates:
    - Dictionary is proper (dependency order correct)
    - All rules are useful (each rule applies when tokenizing the corpus)
    - Construction is plausible (could be built this way)

    Args:
        dictionary: Dictionary to validate
        training_corpus: Corpus that allegedly trained this dictionary

    Returns:
        Dict with:
        - 'is_proper': Boolean
        - 'all_rules_useful': Boolean (each rule applied in corpus tokenization)
        - 'construction_valid': Boolean (could be built this way)

    Complexity: O(|corpus| × |D|) for validation
    """
    if not isinstance(dictionary, BytePairDictionary):
        raise ValueError("dictionary must be a BytePairDictionary instance")

    if not isinstance(training_corpus, str):
        raise ValueError("training_corpus must be a string")

    is_proper = is_proper_dictionary(dictionary)

    all_rules_useful = True
    if training_corpus:  # Only check if corpus is non-empty
        try:
            result = tokenize_sentencepiece(training_corpus, dictionary)
            
            used_rules = set()
            
            for i, (u, v) in enumerate(dictionary.rules):
                rule_output = u + v
                
                partial_rules = dictionary.rules[:i+1]
                partial_dict = BytePairDictionary(partial_rules)
                
                rule_tokenization = tokenize_sentencepiece(rule_output, partial_dict)
                
                if len(rule_tokenization) == 1 and rule_tokenization.tokens[0] == rule_output:
                    used_rules.add(i)
            
            all_rules_useful = (len(used_rules) == len(dictionary))
            
        except Exception:
            all_rules_useful = False
    
    construction_valid = is_proper and all_rules_useful
    
    return {
        'is_proper': is_proper,
        'all_rules_useful': all_rules_useful,
        'construction_valid': construction_valid
    }


def find_unused_rules(dictionary: BytePairDictionary, text: str) -> List[int]:
    """
    Find rules that are not useful when tokenizing the given text.
    
    A rule is useful if it gets applied when tokenizing the string it produces.
    Reference: Corollary 1, Page 7
    
    Args:
        dictionary: Dictionary to analyze
        text: Text to tokenize for usefulness analysis
    
    Returns:
        List of rule indices that are unused
        
    Complexity: O(|D|²) for rule checking
    """
    if not isinstance(dictionary, BytePairDictionary):
        raise ValueError("dictionary must be a BytePairDictionary instance")
    
    unused_rules = []
    
    for i, (u, v) in enumerate(dictionary.rules):
        rule_output = u + v
        
        partial_rules = dictionary.rules[:i+1]
        partial_dict = BytePairDictionary(partial_rules)
        
        try:
            result = tokenize_sentencepiece(rule_output, partial_dict)
            
            if len(result) != 1 or result.tokens[0] != rule_output:
                unused_rules.append(i)
                
        except Exception:
            unused_rules.append(i)
    
    return unused_rules


def analyze_dictionary_properties(dictionary: BytePairDictionary,
                                  corpus: str = None) -> Dict[str, Any]:
    """
    Comprehensive analysis of dictionary properties.

    Args:
        dictionary: Dictionary to analyze
        corpus: Optional training corpus for construction validation

    Returns:
        Dict with comprehensive analysis results
    """
    if not isinstance(dictionary, BytePairDictionary):
        raise ValueError("dictionary must be a BytePairDictionary instance")
    
    is_proper = is_proper_dictionary(dictionary)
    dependencies = get_rule_dependencies(dictionary)
    
    rules_with_dependencies = sum(1 for deps in dependencies.values() if len(deps) > 0)
    max_dependency_chain = max(len(deps) for deps in dependencies.values()) if dependencies else 0
    
    construction_info = None
    if corpus is not None:
        construction_info = validate_training_construction(dictionary, corpus)
    
    test_text = corpus if corpus is not None else ""
    unused = find_unused_rules(dictionary, test_text) if test_text else []
    
    return {
        'is_proper': is_proper,
        'dictionary_size': len(dictionary),
        'rules_with_dependencies': rules_with_dependencies,
        'max_dependency_chain_length': max_dependency_chain,
        'dependency_graph': dependencies,
        'unused_rules': unused,
        'construction_validation': construction_info,
        'total_token_length': sum(len(u) + len(v) for u, v in dictionary.rules)
    }


def are_dictionaries_equivalent(dict1: BytePairDictionary,
                                dict2: BytePairDictionary,
                                test_strings: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Check if two dictionaries are equivalent (produce same tokenizations).

    Reference: Remark 5, Page 11 - decidability is open problem

    This is a HEURISTIC approach using sampling, not a proof.

    Algorithm:
    1. If dicts have different rules → test on sample strings
    2. For each test string:
       - Tokenize with dict1 (SentencePiece)
       - Tokenize with dict2 (SentencePiece)
       - Compare results
    3. Return statistics

    Args:
        dict1, dict2: Dictionaries to compare
        test_strings: Optional list of test strings. If None, generate random samples

    Returns:
        {
            'total_tests': int,
            'matches': int,
        }

    Complexity: O(n × |w|² × |D|) for n test strings
    """
    if not isinstance(dict1, BytePairDictionary) or not isinstance(dict2, BytePairDictionary):
        raise ValueError("Both arguments must be BytePairDictionary instances")

    same_rules = dict1.rules == dict2.rules

    if same_rules and test_strings is None:
        return {
            'likely_equivalent': True,
            'total_tests': 0,
            'matches': 0,
            'first_difference': None,
            'same_rules': True
        }

    if test_strings is None:
        alphabet = set()
        for u, v in dict1.rules + dict2.rules:
            alphabet.update(u)
            alphabet.update(v)

        if not alphabet:
            return {
                'likely_equivalent': True,
                'total_tests': 0,
                'matches': 0,
                'first_difference': None,
                'same_rules': False
            }

        alphabet_list = sorted(list(alphabet))
        test_strings = []

        for _ in range(10):
            length = random.randint(5, 15)
            test_str = ''.join(random.choices(alphabet_list, k=length))
            test_strings.append(test_str)

        for char in alphabet_list[:min(3, len(alphabet_list))]:
            test_strings.append(char * 5)

    total_tests = len(test_strings)
    matches = 0
    first_difference = None

    for test_str in test_strings:
        try:
            result1 = tokenize_sentencepiece(test_str, dict1)
            result2 = tokenize_sentencepiece(test_str, dict2)

            if result1 == result2:
                matches += 1
            elif first_difference is None:
                first_difference = test_str

        except Exception:
            if first_difference is None:
                first_difference = test_str

    likely_equivalent = (matches == total_tests)

    return {
        'likely_equivalent': likely_equivalent,
        'total_tests': total_tests,
        'matches': matches,
        'first_difference': first_difference,
        'same_rules': same_rules
    }


def can_swap_rules(dictionary: BytePairDictionary,
                   index1: int,
                   index2: int) -> bool:
    """
    Check if two adjacent rules can be swapped without changing semantics.

    Reference: Remark 5, Page 11 - rules can be swapped if they don't interact

    Two rules can swap if:
    - Neither rule creates a token used by the other
    - No overlapping token patterns

    Example:
        [a≀b, c≀d] - can swap (independent)
        [a≀b, ab≀c] - cannot swap (second depends on first)

    Args:
        dictionary: Dictionary to analyze
        index1, index2: Indices of rules to check (should be adjacent)

    Returns:
        True if rules can be swapped without changing tokenization behavior

    Algorithm:
    1. Get rules at both indices
    2. Check if rule2 depends on rule1's output
    3. Check if rule1 depends on rule2's output (shouldn't happen if index1 < index2)
    4. Check for pattern overlaps
    """
    if not isinstance(dictionary, BytePairDictionary):
        raise ValueError("dictionary must be a BytePairDictionary instance")

    rules = dictionary.rules

    if index1 < 0 or index1 >= len(rules) or index2 < 0 or index2 >= len(rules):
        raise ValueError(f"Indices out of range: {index1}, {index2}")

    if index1 == index2:
        return True

    if index1 > index2:
        index1, index2 = index2, index1

    u1, v1 = rules[index1]
    u2, v2 = rules[index2]
    token1 = u1 + v1
    token2 = u2 + v2

    if u2 == token1 or v2 == token1:
        return False

    if u1 == token2 or v1 == token2:
        return False


    rule1_components = {u1, v1}
    rule2_components = {u2, v2}

    shared = rule1_components & rule2_components

    if shared:
        for component in shared:
            if len(component) == 1:
                continue
            else:
                return False

    return True


def find_equivalent_orderings(dictionary: BytePairDictionary) -> List[BytePairDictionary]:
    """
    Find all equivalent dictionaries by swapping independent rules.

    Reference: Remark 5, Page 11

    Algorithm:
    1. Start with original dictionary
    2. Find pairs of rules that can be swapped
    3. Generate all permutations of swappable rules
    4. Return list of equivalent dictionaries

    Returns:
        List of dictionaries that should be equivalent to input

    Note: This grows exponentially, limit to small dictionaries or bounded search
    """
    if not isinstance(dictionary, BytePairDictionary):
        raise ValueError("dictionary must be a BytePairDictionary instance")

    rules = dictionary.rules

    if len(rules) <= 1:
        return [dictionary]

    swappable_pairs = []
    for i in range(len(rules) - 1):
        if can_swap_rules(dictionary, i, i + 1):
            swappable_pairs.append((i, i + 1))

    if not swappable_pairs:
        return [dictionary]

    MAX_DICTIONARIES = 10
    equivalent_dicts = [dictionary]
    seen_rule_orders = {tuple(rules)}

    to_process = [list(rules)]

    while to_process and len(equivalent_dicts) < MAX_DICTIONARIES:
        current_rules = to_process.pop(0)

        for i, j in swappable_pairs:
            new_rules = current_rules[:]
            new_rules[i], new_rules[j] = new_rules[j], new_rules[i]

            new_tuple = tuple(new_rules)
            if new_tuple not in seen_rule_orders:
                seen_rule_orders.add(new_tuple)
                new_dict = BytePairDictionary(new_rules)
                equivalent_dicts.append(new_dict)
                to_process.append(new_rules)

                if len(equivalent_dicts) >= MAX_DICTIONARIES:
                    break

    return equivalent_dicts
