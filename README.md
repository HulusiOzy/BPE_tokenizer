# BPE Tokenizer

Implementation of the formal BPE tokenization algorithms from Berglund & van der Merwe (2023), "Formalizing BPE Tokenization" (NCMA 2023). Paper available at https://arxiv.org/abs/2309.08715.

This implements the theoretical foundations and advanced algorithms described in the paper, including three tokenization variants, dictionary validation, incremental updates, and streaming tokenization with provable lookahead bounds.

## What's Implemented

**Core Tokenizers**
The paper formally defines three tokenization methods. All are implemented:
- Base tokenizer: Non-deterministic, explores all valid tokenizations (Definition 2, Page 3)
- SentencePiece: Deterministic, highest-priority leftmost selection (Definition 3, Page 4)  
- HuggingFace: Deterministic, exhaustive rule application (Definition 4, Page 5)

**Dictionary Analysis**
- Proper dictionary detection (Definition 5, Page 5)
- Rule dependency analysis
- Dictionary equivalence checking (heuristic approach to the open problem in Remark 5, Page 11)

**Advanced Algorithms**
- Incremental tokenization updates when concatenating strings (Algorithm 1, Page 7)
- Streaming tokenization with finite lookahead (Algorithm 2, Page 10)
- Lookahead constant calculation (Theorem 2, Page 9)
- Chain length optimization (Remark 5, Page 11)

## Usage

```python
from bpe_tokenizer.core import BytePairDictionary
from bpe_tokenizer.tokenizers import tokenize_sentencepiece

# Create dictionary (priority order matters)
D = BytePairDictionary([
    ('a', 'b'),
    ('b', 'c'),
    ('ab', 'c')
])

# Tokenize
result = tokenize_sentencepiece("abc", D)
print(result.tokens)  # ['abc']
```

The paper's Example 1 (Page 4):

```python
D = BytePairDictionary([('a', 'b'), ('a', 'bc'), ('b', 'c'), ('ab', 'c')])
result = tokenize_sentencepiece("abcbcab", D)
# Result: ['abc', 'bc', 'ab']
```

Example 3 (Page 5) demonstrates when SentencePiece and HuggingFace differ:

```python
D = BytePairDictionary([('ab', 'a'), ('a', 'b')])  # Improper dictionary
w = "abababab"

sp = tokenize_sentencepiece(w, D)  # ['aba', 'b', 'aba', 'b']
hf = tokenize_huggingface(w, D)     # ['ab', 'ab', 'ab', 'ab']
```

Incremental updates for efficient concatenation:

```python
from bpe_tokenizer.algorithms import incremental_update

tau1 = tokenize_sentencepiece("hello", D)
tau2 = tokenize_sentencepiece("world", D)
result = incremental_update(tau1, tau2, D)  # Faster than re-tokenizing "helloworld"
```

## Testing

The implementation has 107 tests covering all paper examples, edge cases, and theoretical properties.

```bash
pytest tests/ -v
```

Key test coverage:
- All examples from the paper (Example 1, Example 3, etc.)
- Lemma 1 validation: proper dictionaries produce identical results across tokenizers
- Theorem 2 validation: lookahead bounds hold
- Property-based testing with randomly generated dictionaries
- Cross-validation between all three tokenizers

## Implementation Notes

**Complexity**
The base tokenizer has exponential worst-case complexity O(|D|^|w|) and is only used for validation. Production tokenizers (SentencePiece: O(|w|² × |D|), HuggingFace: O(|w| × |D|)) are efficient for real use.

**Dictionary Training**
The paper describes dictionary training in Remark 1 (Page 5) but training from corpus is not yet implemented. Current focus is on the tokenization algorithms given a pre-built dictionary.

**Open Problems**
Dictionary equivalence (Remark 5, Page 11) is noted as an open research problem. This implementation includes heuristic checking via sampling but not a complete solution.

## Citation

```bibtex
@inproceedings{berglund2023formalizing,
  title={Formalizing BPE Tokenization},
  author={Berglund, Martin and van der Merwe, Brink},
  booktitle={NCMA 2023},
  year={2023}
}
```
