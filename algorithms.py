"""
Advanced BPE tokenization algorithms.

Implements the main algorithmic contributions from the paper:
- Incremental tokenization updates (Algorithm 1, Page 7)
- Streaming tokenization with lookahead (Algorithm 2, Page 10)
- Lookahead optimization and chain length calculation

Reference: "Formalizing BPE Tokenization" by Berglund & van der Merwe
"""

from typing import List, Dict, Iterator, Optional, Any
import itertools
from bpe_tokenizer.core import Tokenization, BytePairDictionary
from bpe_tokenizer.tokenizers import tokenize_sentencepiece
from bpe_tokenizer.validation import get_rule_dependencies


"""
Given T^D(w) = τ and T^D(w') = τ', compute T^D(ww') efficiently.

Reference: Algorithm 1, Page 7

Algorithm (from paper):
1. Let τ = u1≀⋯≀un and τ' = u'1≀⋯≀u'm
2. Initialize i = n, j = 1
3. Compute T^D(ui⋯unu'1⋯u'j) = v1≀⋯≀vk
4. If (v1 = ui OR i = 1) AND (vk = u'j OR j = m):
   - Output: u1≀⋯≀ui-1≀v1≀⋯≀vk≀u'j+1≀⋯≀u'm
   - HALT
5. If ui ≠ v1 and i > 1: i ← i-1
6. If u'j ≠ vk and j < m: j ← j+1
7. Go to step 3

Args:
    tau: Tokenization of w (already computed)
    tau_prime: Tokenization of w' (already computed)
    dictionary: BPE dictionary

Returns:
    T^D(ww') without re-tokenizing from scratch

Complexity: Best case O(1), worst case O(|w|)
            (see Example 2, Page 4 for worst case)
"""
def incremental_update(tau: Tokenization,
                      tau_prime: Tokenization,
                      dictionary: BytePairDictionary) -> Tokenization:
    tokens_w = tau.tokens
    tokens_w_prime = tau_prime.tokens

    if not tokens_w:
        return tau_prime
    if not tokens_w_prime:
        return tau

    n = len(tokens_w)
    m = len(tokens_w_prime)

    i = n - 1
    j = 0

    max_iterations = n + m
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        middle_string = ''.join(tokens_w[i:]) + ''.join(tokens_w_prime[:j+1])
        middle_tokenization = tokenize_sentencepiece(middle_string, dictionary)
        middle_tokens = middle_tokenization.tokens

        if not middle_tokens:
            break

        v1 = middle_tokens[0]
        vk = middle_tokens[-1]

        condition1 = (v1 == tokens_w[i]) or (i == 0)
        condition2 = (vk == tokens_w_prime[j]) or (j == m - 1)

        if condition1 and condition2:
            result_tokens = []

            if i > 0:
                result_tokens.extend(tokens_w[:i])

            result_tokens.extend(middle_tokens)

            if j < m - 1:
                result_tokens.extend(tokens_w_prime[j+1:])

            return Tokenization(result_tokens)

        if tokens_w[i] != v1 and i > 0:
            i -= 1

        if tokens_w_prime[j] != vk and j < m - 1:
            j += 1

    full_string = tau.concatenate() + tau_prime.concatenate()
    return tokenize_sentencepiece(full_string, dictionary)


"""
Tokenize a sequence of strings using incremental updates.

Use case: Streaming text arrives in chunks, tokenize efficiently.

Algorithm:
1. Tokenize first string normally
2. For each subsequent string:
   - Tokenize it normally
   - Use incremental_update to merge with previous result

Args:
    strings: List of strings to tokenize and concatenate
    dictionary: BPE dictionary

Returns:
    List of intermediate tokenizations
    Final element is tokenization of concatenated string
"""
def incremental_tokenize_sequence(strings: List[str],
                                 dictionary: BytePairDictionary) -> List[Tokenization]:
    if not strings:
        return []

    results = []

    current_tokenization = tokenize_sentencepiece(strings[0], dictionary)
    results.append(current_tokenization)

    for string in strings[1:]:
        new_tokenization = tokenize_sentencepiece(string, dictionary)

        current_tokenization = incremental_update(
            current_tokenization,
            new_tokenization,
            dictionary
        )

        results.append(current_tokenization)

    return results


"""
Calculate l(D) - the lookahead constant for dictionary D.

Reference: Theorem 2, Page 9
Bound: l(D) ≤ |D| × max{|uv| | u≀v ∈ D}

This is the minimum number of tokens we need to look ahead
to guarantee correct tokenization.

Args:
    dictionary: Dictionary to analyze

Returns:
    Lookahead constant l(D)

Note: Paper's bound is loose. We implement the theoretical bound.
      Better bounds discussed in Remark 5 (Page 11).
"""
def calculate_lookahead_constant(dictionary: BytePairDictionary) -> int:
    rules = dictionary.rules

    if not rules:
        return 0

    dict_size = len(rules)

    max_rule_length = max(len(u) + len(v) for u, v in rules)

    return dict_size * max_rule_length


"""
Calculate sufficient lookahead (Definition 7, Page 9).

Same as lookahead constant but explicit about what it means.
"""
def calculate_sufficient_lookahead(dictionary: BytePairDictionary) -> int:
    return calculate_lookahead_constant(dictionary)


"""
Calculate c(D) - chain length of dictionary.

Reference: Remark 5, Page 11

c(D) = maximum length of rule sequence in decreasing priority

This gives a tighter bound: l(D) ≤ c(D) × max{|uv|}

Algorithm:
1. For each rule, find longest dependency chain
2. Return maximum chain length found
"""
def calculate_chain_length(dictionary: BytePairDictionary) -> int:
    rules = dictionary.rules

    if not rules:
        return 0

    dependencies = get_rule_dependencies(dictionary)

    def get_chain_depth(rule_index: int, visited: set) -> int:
        if rule_index in visited:
            return 0

        rule_deps = dependencies.get(rule_index, [])

        if not rule_deps:
            return 1

        visited.add(rule_index)
        max_dep_depth = 0

        for dep_index in rule_deps:
            dep_depth = get_chain_depth(dep_index, visited.copy())
            max_dep_depth = max(max_dep_depth, dep_depth)

        return max_dep_depth + 1

    max_chain_length = 0
    for rule_index in range(len(rules)):
        chain_depth = get_chain_depth(rule_index, set())
        max_chain_length = max(max_chain_length, chain_depth)

    return max_chain_length


"""
Tokenize left-to-right with finite lookahead (Algorithm 2, Page 10).

Reference: Algorithm 2, Page 10

Key idea: We can output tokens as we go, only keeping
'lookahead' characters in memory at any time.

Algorithm:
1. Precompute f: (Σ∪{z})^k → Σ≀ where k = lookahead
   f(w) = first token of T^D(w)
2. Read k symbols at a time
3. Lookup f(buffer) to get next token
4. Output token, advance buffer
5. Repeat until string ends (padded with special symbol z)

Args:
    text: String to tokenize
    dictionary: BPE dictionary
    lookahead: Number of characters to buffer (None = calculate automatically)

Returns:
    Tokenization (same as tokenize_sentencepiece but using streaming)

Complexity:
    Precompute: O(|Σ|^k × complexity of tokenization)
    Per char: O(1) lookup
    Space: O(1) in string length (O(|Σ|^k) for lookup table)

Note: Precomputation is expensive! Only practical for small alphabets.
"""
def tokenize_streaming(text: str,
                      dictionary: BytePairDictionary,
                      lookahead: Optional[int] = None) -> Tokenization:
    if lookahead is None:
        lookahead = calculate_lookahead_constant(dictionary)

    if lookahead > 20 or not text:
        return tokenize_sentencepiece(text, dictionary)

    alphabet = set()
    for u, v in dictionary.rules:
        alphabet.update(u)
        alphabet.update(v)

    alphabet.update(text)

    PADDING = '\x00'
    alphabet.add(PADDING)

    alphabet_list = sorted(list(alphabet))

    lookup_table: Dict[str, str] = {}

    if len(alphabet_list) ** lookahead > 10000:
        return tokenize_sentencepiece(text, dictionary)

    for combo in itertools.product(alphabet_list, repeat=lookahead):
        key = ''.join(combo)
        key_clean = key.replace(PADDING, '')

        if key_clean:
            try:
                tokenization = tokenize_sentencepiece(key_clean, dictionary)
                if tokenization.tokens:
                    lookup_table[key] = tokenization.tokens[0]
                else:
                    lookup_table[key] = key_clean
            except:
                lookup_table[key] = key_clean if key_clean else ''
        else:
            lookup_table[key] = ''

    padded_text = text + (PADDING * lookahead)

    result_tokens = []
    pos = 0

    while pos < len(text):
        window = padded_text[pos:pos + lookahead]

        if window in lookup_table:
            token = lookup_table[window]

            if token and token != PADDING:
                result_tokens.append(token)
                pos += len(token)
            else:
                if pos < len(text):
                    result_tokens.append(text[pos])
                    pos += 1
                else:
                    break
        else:
            if pos < len(text):
                result_tokens.append(text[pos])
                pos += 1
            else:
                break

    result_tokens = [t for t in result_tokens if PADDING not in t]

    return Tokenization(result_tokens) if result_tokens else Tokenization([])


"""
Generator version for truly streaming input.

Yields tokens as they become available without storing full text.

Args:
    text_stream: Iterator yielding characters/chunks
    dictionary: BPE dictionary
    lookahead: Buffer size (None = calculate automatically)

Yields:
    Tokens as they are determined

Note: This is a practical approximation. Full correctness requires
      lookahead buffer, but for most texts we can emit tokens early.
"""
def tokenize_streaming_generator(text_stream: Iterator[str],
                                dictionary: BytePairDictionary,
                                lookahead: Optional[int] = None) -> Iterator[str]:
    if lookahead is None:
        lookahead = calculate_lookahead_constant(dictionary)
        lookahead = min(lookahead, 50)

    buffer = ""

    for chunk in text_stream:
        buffer += chunk

        while len(buffer) >= lookahead:
            tokenization = tokenize_sentencepiece(buffer[:lookahead], dictionary)

            if tokenization.tokens:
                first_token = tokenization.tokens[0]
                yield first_token

                buffer = buffer[len(first_token):]
            else:
                if buffer:
                    yield buffer[0]
                    buffer = buffer[1:]
                else:
                    break

    if buffer:
        tokenization = tokenize_sentencepiece(buffer, dictionary)
        for token in tokenization.tokens:
            yield token


"""
Compare different lookahead bounds for analysis.

Returns:
    Dictionary with theoretical bound, chain-based bound, and comparison
"""
def get_lookahead_comparison(dictionary: BytePairDictionary) -> Dict[str, Any]:
    theoretical = calculate_lookahead_constant(dictionary)
    chain_based = calculate_chain_length(dictionary)

    rules = dictionary.rules
    max_token_length = max(len(u) + len(v) for u, v in rules) if rules else 0

    chain_bound = chain_based * max_token_length

    return {
        'theoretical_bound': theoretical,
        'chain_length': chain_based,
        'chain_based_bound': chain_bound,
        'max_token_length': max_token_length,
        'improvement_ratio': theoretical / chain_bound if chain_bound > 0 else 1,
        'is_chain_better': chain_bound < theoretical
    }
