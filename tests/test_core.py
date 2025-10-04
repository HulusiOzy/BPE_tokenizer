"""
Unit tests for core BPE tokenization data structures.

Tests the implementation of:
- Tokenization class
- BytePairDictionary class  
- T_empty function

Following the specifications from Tasks file and the formal paper.
"""

import pytest
from bpe_tokenizer.core import Tokenization, BytePairDictionary, T_empty


class TestTokenization:
    """Test cases for Tokenization class."""
    
    def test_valid_tokenization(self):
        """Test creating valid tokenizations."""
        # Single token
        tau = Tokenization(["hello"])
        assert len(tau) == 1
        assert tau.concatenate() == "hello"
        assert str(tau) == "hello"
        
        # Multiple tokens
        tau = Tokenization(["a", "b", "c"])
        assert len(tau) == 3
        assert tau.concatenate() == "abc"
        assert str(tau) == "a≀b≀c"
        
        # Empty tokenization
        tau = Tokenization([])
        assert len(tau) == 0
        assert tau.concatenate() == ""
        assert str(tau) == "∅"
    
    def test_invalid_tokenization(self):
        """Test validation of tokenization constraints."""
        # Empty token should raise ValueError (violates Σ+ constraint)
        with pytest.raises(ValueError, match="Token at index 1 is empty"):
            Tokenization(["a", "", "c"])
        
        # Non-string token should raise ValueError
        with pytest.raises(ValueError, match="Token at index 0 must be a string"):
            Tokenization([123])
    
    def test_tokenization_methods(self):
        """Test tokenization methods and properties."""
        tokens = ["ab", "c", "def"]
        tau = Tokenization(tokens)
        
        # Test indexing
        assert tau[0] == "ab"
        assert tau[1] == "c" 
        assert tau[2] == "def"
        
        # Test iteration
        result = list(tau)
        assert result == tokens
        
        # Test tokens property (should be a copy)
        token_copy = tau.tokens
        assert token_copy == tokens
        token_copy.append("modified")
        assert tau.tokens == tokens  # Original unchanged
        
        # Test equality
        tau2 = Tokenization(["ab", "c", "def"])
        tau3 = Tokenization(["ab", "cd", "ef"])
        assert tau == tau2
        assert tau != tau3
        assert tau != "not a tokenization"


class TestBytePairDictionary:
    """Test cases for BytePairDictionary class."""
    
    def test_valid_dictionary(self):
        """Test creating valid dictionaries."""
        # Empty dictionary
        D = BytePairDictionary([])
        assert len(D) == 0
        assert str(D) == "[]"
        
        # Single rule
        D = BytePairDictionary([("a", "b")])
        assert len(D) == 1
        assert D.can_apply("a", "b")
        assert D.get_priority("a", "b") == 0
        assert str(D) == "[a≀b]"
        
        # Multiple rules - priority ordering
        rules = [("a", "b"), ("c", "d"), ("ab", "c")]
        D = BytePairDictionary(rules)
        assert len(D) == 3
        assert D.get_priority("a", "b") == 0  # Highest priority
        assert D.get_priority("c", "d") == 1
        assert D.get_priority("ab", "c") == 2  # Lowest priority
        assert str(D) == "[a≀b, c≀d, ab≀c]"
    
    def test_dictionary_validation(self):
        """Test dictionary validation."""
        # Invalid rule format
        with pytest.raises(ValueError, match="Rule at index 0 must be a pair"):
            BytePairDictionary(["not a pair"])
        
        with pytest.raises(ValueError, match="Rule at index 0 must be a pair"):
            BytePairDictionary([("a", "b", "c")])  # Too many elements
        
        # Non-string elements
        with pytest.raises(ValueError, match="Rule at index 0 must contain strings"):
            BytePairDictionary([(1, 2)])
    
    def test_dictionary_methods(self):
        """Test dictionary methods and properties."""
        rules = [("a", "b"), ("c", "d"), ("ab", "c")]
        D = BytePairDictionary(rules)
        
        # Test can_apply
        assert D.can_apply("a", "b") == True
        assert D.can_apply("c", "d") == True
        assert D.can_apply("x", "y") == False
        
        # Test get_priority
        assert D.get_priority("a", "b") == 0
        assert D.get_priority("c", "d") == 1
        assert D.get_priority("ab", "c") == 2
        assert D.get_priority("x", "y") == -1  # Not found
        
        # Test indexing
        assert D[0] == ("a", "b")
        assert D[1] == ("c", "d")
        assert D[2] == ("ab", "c")
        
        # Test iteration
        result = list(D)
        assert result == rules
        
        # Test rules property (should be a copy)
        rules_copy = D.rules
        assert rules_copy == rules
        rules_copy.append(("x", "y"))
        assert D.rules == rules  # Original unchanged


class TestTEmpty:
    """Test cases for T_empty function."""
    
    def test_valid_strings(self):
        """Test T_empty with valid strings."""
        # Single character
        tau = T_empty("a")
        assert len(tau) == 1
        assert tau.concatenate() == "a"
        assert str(tau) == "a"
        
        # Multiple characters
        tau = T_empty("abc")
        assert len(tau) == 3
        assert tau.concatenate() == "abc"
        assert str(tau) == "a≀b≀c"
        
        # Complex string
        tau = T_empty("hello")
        assert len(tau) == 5
        assert tau.concatenate() == "hello"
        assert str(tau) == "h≀e≀l≀l≀o"
    
    def test_empty_string(self):
        """Test T_empty with empty string."""
        tau = T_empty("")
        assert len(tau) == 0
        assert tau.concatenate() == ""
        assert str(tau) == "∅"
    
    def test_invalid_input(self):
        """Test T_empty validation."""
        with pytest.raises(ValueError, match="Input must be a string"):
            T_empty(123)
    
    def test_character_level_property(self):
        """Test that T_empty produces character-level tokenization."""
        w = "abcdef"
        tau = T_empty(w)
        
        # Each character should be its own token
        for i, char in enumerate(w):
            assert tau[i] == char
        
        # Concatenation should reconstruct original
        assert tau.concatenate() == w


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_basic_workflow(self):
        """Test basic workflow from Tasks file."""
        # Create dictionary from Example 1 (page 4)
        D = BytePairDictionary([
            ('a', 'b'),
            ('a', 'bc'), 
            ('b', 'c'),
            ('ab', 'c')
        ])
        
        # Create initial tokenization
        w = "abcbcab"
        initial = T_empty(w)
        
        # Verify basic properties
        assert initial.concatenate() == w
        assert len(initial) == 7  # 7 characters
        assert str(initial) == "a≀b≀c≀b≀c≀a≀b"
        
        # Verify dictionary priorities
        assert D.get_priority('a', 'b') == 0      # Highest priority
        assert D.get_priority('a', 'bc') == 1
        assert D.get_priority('b', 'c') == 2  
        assert D.get_priority('ab', 'c') == 3     # Lowest priority