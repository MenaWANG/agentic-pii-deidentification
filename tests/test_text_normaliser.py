"""
Comprehensive tests for text normalisation layer.

Tests cover:
1. Spelled-out letter combination
2. Number word to digit conversion
3. Email address reconstruction
4. Edge cases and integration scenarios
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
src_path = str(Path(__file__).parent.parent / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from utils.text_normaliser import (
    TextNormaliser,
    normalize_text,
    normalize_spelled_letters,
    normalize_number_words,
    normalize_email_addresses
)


class TestTextNormaliser:
    """Test class for TextNormaliser functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.normaliser = TextNormaliser()
    
    def test_init(self):
        """Test TextNormaliser initialization."""
        assert self.normaliser.number_words['zero'] == '0'
        assert self.normaliser.number_words['nine'] == '9'
        assert 'gmail' in self.normaliser.email_providers
        assert self.normaliser.email_providers['gmail'] == 'gmail.com'
    
    def test_get_supported_email_providers(self):
        """Test getting supported email providers."""
        providers = self.normaliser.get_supported_email_providers()
        assert isinstance(providers, list)
        assert 'gmail' in providers
        assert 'hotmail' in providers
        assert 'outlook' in providers
        assert len(providers) > 10  # Should have many providers


class TestSpelledLettersNormalization:
    """Test spelled-out letter combination functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.normaliser = TextNormaliser()
    
    def test_simple_spelled_letters(self):
        """Test basic spelled-out letter combinations."""
        # Basic cases
        assert self.normaliser.normalize_spelled_letters("d o e") == "doe"
        assert self.normaliser.normalize_spelled_letters("j o h n") == "john"
        assert self.normaliser.normalize_spelled_letters("a b c") == "abc"
    
    def test_spelled_letters_with_context(self):
        """Test spelled-out letters within sentences."""
        text = "My name is j o h n and I live in s y d n e y"
        expected = "My name is john and I live in sydney"
        assert self.normaliser.normalize_spelled_letters(text) == expected
    
    def test_spelled_letters_various_lengths(self):
        """Test different lengths of spelled-out sequences."""
        # Two letters
        assert self.normaliser.normalize_spelled_letters("a b") == "ab"
        
        # Three letters
        assert self.normaliser.normalize_spelled_letters("c a t") == "cat"
        
        # Four letters
        assert self.normaliser.normalize_spelled_letters("j o h n") == "john"
        
        # Five letters
        assert self.normaliser.normalize_spelled_letters("s m i t h") == "smith"
        
        # Six letters
        assert self.normaliser.normalize_spelled_letters("w i l s o n") == "wilson"
    
    def test_spelled_letters_case_insensitive(self):
        """Test case insensitive spelled-out letters."""
        assert self.normaliser.normalize_spelled_letters("J O H N") == "JOHN"
        assert self.normaliser.normalize_spelled_letters("j O h N") == "jOhN"
        assert self.normaliser.normalize_spelled_letters("D o E") == "DoE"
    
    def test_spelled_letters_with_punctuation(self):
        """Test spelled-out letters with punctuation."""
        text = "My surname is s m i t h, and my first name is j o h n."
        expected = "My surname is smith, and my first name is john."
        assert self.normaliser.normalize_spelled_letters(text) == expected
    
    def test_spelled_letters_multiple_sequences(self):
        """Test multiple spelled-out sequences in one text."""
        text = "Hi, I'm j o h n d o e from s y d n e y"
        expected = "Hi, I'm john doe from sydney"
        assert self.normaliser.normalize_spelled_letters(text) == expected
    
    def test_spelled_letters_no_change(self):
        """Test text that shouldn't be changed."""
        # Normal words should not be affected
        text = "This is a normal sentence with regular words."
        assert self.normaliser.normalize_spelled_letters(text) == text
        
        # Single letters should not be affected
        text = "I have a grade of A in mathematics."
        assert self.normaliser.normalize_spelled_letters(text) == text
    
    def test_spelled_letters_edge_cases(self):
        """Test edge cases for spelled-out letters."""
        # Empty string
        assert self.normaliser.normalize_spelled_letters("") == ""
        
        # Only spaces
        assert self.normaliser.normalize_spelled_letters("   ") == "   "
        
        # Single letter
        assert self.normaliser.normalize_spelled_letters("a") == "a"
        
        # Non-alphabetic characters
        text = "My number is 1 2 3 4"
        assert self.normaliser.normalize_spelled_letters(text) == text


class TestNumberWordsNormalization:
    """Test number word to digit conversion functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.normaliser = TextNormaliser()
    
    def test_basic_number_conversion(self):
        """Test basic number word to digit conversion."""
        # Single digits
        assert self.normaliser.normalize_number_words("zero") == "0"
        assert self.normaliser.normalize_number_words("one") == "1"
        assert self.normaliser.normalize_number_words("nine") == "9"
    
    def test_number_sequences(self):
        """Test conversion of number sequences."""
        # Phone number example from README
        text = "zero four one two three four five six seven"
        expected = "041234567"
        assert self.normaliser.normalize_number_words(text) == expected
        
        # Another sequence
        text = "nine eight seven six five"
        expected = "98765"
        assert self.normaliser.normalize_number_words(text) == expected
    
    def test_number_words_in_context(self):
        """Test number words within sentences."""
        text = "My phone number is zero four one two three four five six seven"
        expected = "My phone number is 041234567"
        assert self.normaliser.normalize_number_words(text) == expected
        
        text = "Call me at zero four one two and ask for extension five six seven"
        expected = "Call me at 0412 and ask for extension 567"
        assert self.normaliser.normalize_number_words(text) == expected
    
    def test_number_words_case_insensitive(self):
        """Test case insensitive number word conversion."""
        assert self.normaliser.normalize_number_words("ZERO ONE TWO") == "012"
        assert self.normaliser.normalize_number_words("Zero One Two") == "012"
        assert self.normaliser.normalize_number_words("zero ONE two") == "012"
    
    def test_number_words_mixed_content(self):
        """Test number words mixed with other content."""
        text = "I have zero dollars and five cents in my account number one two three"
        expected = "I have 0 dollars and 5 cents in my account number 123"
        assert self.normaliser.normalize_number_words(text) == expected
    
    def test_number_words_no_change(self):
        """Test text that shouldn't be changed."""
        # Regular text
        text = "This is a normal sentence without number words."
        assert self.normaliser.normalize_number_words(text) == text
        
        # Already numeric
        text = "My number is 123456789"
        assert self.normaliser.normalize_number_words(text) == text
    
    def test_number_words_edge_cases(self):
        """Test edge cases for number words."""
        # Empty string
        assert self.normaliser.normalize_number_words("") == ""
        
        # Only spaces
        assert self.normaliser.normalize_number_words("   ") == "   "
        
        # Partial matches (words that contain number words but aren't exact)
        text = "I have nothing and everything"
        assert self.normaliser.normalize_number_words(text) == text


class TestEmailAddressNormalization:
    """Test email address reconstruction functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.normaliser = TextNormaliser()
    
    def test_basic_email_conversion(self):
        """Test basic email address conversion."""
        # Gmail example from README
        text = "john doe at gmail dot com"
        expected = "john.doe@gmail.com"
        assert self.normaliser.normalize_email_addresses(text) == expected
    
    def test_various_email_providers(self):
        """Test different email providers."""
        # Gmail
        assert self.normaliser.normalize_email_addresses("user at gmail dot com") == "user@gmail.com"
        
        # Hotmail
        assert self.normaliser.normalize_email_addresses("user at hotmail dot com") == "user@hotmail.com"
        
        # Outlook
        assert self.normaliser.normalize_email_addresses("user at outlook dot com") == "user@outlook.com"
        
        # Yahoo
        assert self.normaliser.normalize_email_addresses("user at yahoo dot com") == "user@yahoo.com"
        
        # Custom domain
        assert self.normaliser.normalize_email_addresses("user at example dot org") == "user@example.org"
    
    def test_email_with_multiple_names(self):
        """Test email addresses with multiple name parts."""
        text = "john michael smith at gmail dot com"
        expected = "john.michael.smith@gmail.com"
        assert self.normaliser.normalize_email_addresses(text) == expected
        
        text = "mary jane watson at hotmail dot com"
        expected = "mary.jane.watson@hotmail.com"
        assert self.normaliser.normalize_email_addresses(text) == expected
    
    def test_email_case_insensitive(self):
        """Test case insensitive email conversion."""
        text = "JOHN DOE AT GMAIL DOT COM"
        expected = "JOHN.DOE@gmail.com"
        assert self.normaliser.normalize_email_addresses(text) == expected
        
        text = "John Doe At Gmail Dot Com"
        expected = "John.Doe@gmail.com"
        assert self.normaliser.normalize_email_addresses(text) == expected
    
    def test_email_in_context(self):
        """Test email addresses within sentences."""
        text = "You can reach me at john doe at gmail dot com or call me"
        expected = "You can reach me at john.doe@gmail.com or call me"
        assert self.normaliser.normalize_email_addresses(text) == expected
        
        text = "My email is mary smith at hotmail dot com and my phone is 123456789"
        expected = "My email is mary.smith@hotmail.com and my phone is 123456789"
        assert self.normaliser.normalize_email_addresses(text) == expected
    
    def test_email_provider_aliases(self):
        """Test email provider aliases."""
        # Google should map to gmail.com
        text = "user at google dot com"
        expected = "user@gmail.com"
        assert self.normaliser.normalize_email_addresses(text) == expected
        
        # Apple should map to icloud.com
        text = "user at apple dot com"
        expected = "user@icloud.com"
        assert self.normaliser.normalize_email_addresses(text) == expected
        
        # Proton should map to protonmail.com
        text = "user at proton dot com"
        expected = "user@protonmail.com"
        assert self.normaliser.normalize_email_addresses(text) == expected
    
    def test_email_multiple_addresses(self):
        """Test multiple email addresses in one text."""
        text = "Contact john doe at gmail dot com or mary smith at hotmail dot com"
        expected = "Contact john.doe@gmail.com or mary.smith@hotmail.com"
        assert self.normaliser.normalize_email_addresses(text) == expected
    
    def test_email_no_change(self):
        """Test text that shouldn't be changed."""
        # Normal text
        text = "This is a normal sentence without email addresses."
        assert self.normaliser.normalize_email_addresses(text) == text
        
        # Already formatted email
        text = "My email is john.doe@gmail.com"
        assert self.normaliser.normalize_email_addresses(text) == text
    
    def test_email_edge_cases(self):
        """Test edge cases for email addresses."""
        # Empty string
        assert self.normaliser.normalize_email_addresses("") == ""
        
        # Only spaces
        assert self.normaliser.normalize_email_addresses("   ") == "   "
        
        # Incomplete email patterns
        text = "john at gmail"  # Missing "dot com"
        assert self.normaliser.normalize_email_addresses(text) == text
        
        text = "john dot com"  # Missing "at gmail"
        assert self.normaliser.normalize_email_addresses(text) == text


class TestIntegratedNormalization:
    """Test integrated normalization functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.normaliser = TextNormaliser()
    
    def test_normalize_text_all_features(self):
        """Test full normalization with all features."""
        text = "Hi, I'm j o h n d o e, my phone is zero four one two three four five six seven and email is john doe at gmail dot com"
        expected = "Hi, I'm john doe, my phone is 041234567 and email is john.doe@gmail.com"
        assert self.normaliser.normalize_text(text) == expected
    
    def test_normalize_text_partial_features(self):
        """Test normalization with only some features present."""
        # Only spelled letters
        text = "My name is j o h n"
        expected = "My name is john"
        assert self.normaliser.normalize_text(text) == expected
        
        # Only numbers
        text = "Call me at zero four one two"
        expected = "Call me at 0412"
        assert self.normaliser.normalize_text(text) == expected
        
        # Only email
        text = "Email me at john at gmail dot com"
        expected = "Email me at john@gmail.com"
        assert self.normaliser.normalize_text(text) == expected
    
    def test_normalize_text_complex_scenario(self):
        """Test complex real-world scenario."""
        text = """
        Hello, this is j o h n s m i t h from customer service.
        Your membership number is nine five nine two four six one seven.
        We can be reached at support team at company dot com.
        For urgent matters, call us at zero four one six four eight nine nine six three seven four.
        """
        
        result = self.normaliser.normalize_text(text)
        
        # Check that all transformations were applied
        assert "john smith" in result.lower()
        assert "95924617" in result
        assert "support.team@company.com" in result
        assert "041648996374" in result
    
    def test_normalize_text_edge_cases(self):
        """Test edge cases for full normalization."""
        # None input
        assert self.normaliser.normalize_text(None) is None
        
        # Empty string
        assert self.normaliser.normalize_text("") == ""
        
        # Non-string input
        assert self.normaliser.normalize_text(123) == 123
        
        # Only spaces
        assert self.normaliser.normalize_text("   ") == "   "
    
    def test_normalize_text_no_changes_needed(self):
        """Test text that doesn't need normalization."""
        text = "This is a normal sentence with no special patterns."
        assert self.normaliser.normalize_text(text) == text


class TestConvenienceFunctions:
    """Test standalone convenience functions."""
    
    def test_normalize_text_function(self):
        """Test the standalone normalize_text function."""
        text = "My name is j o h n and my number is zero one two three"
        result = normalize_text(text)
        assert "john" in result
        assert "0123" in result
    
    def test_normalize_spelled_letters_function(self):
        """Test the standalone normalize_spelled_letters function."""
        text = "My name is j o h n d o e"
        result = normalize_spelled_letters(text)
        assert result == "My name is john doe"
    
    def test_normalize_number_words_function(self):
        """Test the standalone normalize_number_words function."""
        text = "My number is zero one two three four"
        result = normalize_number_words(text)
        assert result == "My number is 01234"
    
    def test_normalize_email_addresses_function(self):
        """Test the standalone normalize_email_addresses function."""
        text = "Contact me at john doe at gmail dot com"
        result = normalize_email_addresses(text)
        assert result == "Contact me at john.doe@gmail.com"


class TestRealWorldScenarios:
    """Test real-world scenarios and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.normaliser = TextNormaliser()
    
    def test_call_transcript_scenario(self):
        """Test a realistic call transcript scenario."""
        transcript = """
        Agent: Hello, could you please confirm your name?
        Customer: Yes, it's j o h n d o e.
        Agent: And your phone number?
        Customer: It's zero four one two three four five six seven eight.
        Agent: What's your email address?
        Customer: It's john doe at gmail dot com.
        Agent: Thank you for confirming your details.
        """
        
        result = self.normaliser.normalize_text(transcript)
        
        # Verify all normalizations were applied
        assert "john doe" in result.lower()
        assert "04123456789" in result
        assert "john.doe@gmail.com" in result
    
    def test_mixed_content_scenario(self):
        """Test mixed content with various patterns."""
        text = "Customer ID: a b c one two three, Email: support at company dot net, Phone: zero four one two"
        result = self.normaliser.normalize_text(text)
        
        assert "abc123" in result
        assert "support@company.net" in result
        assert "0412" in result
    
    def test_preserves_existing_formatting(self):
        """Test that existing proper formatting is preserved."""
        text = "My email is john.doe@gmail.com and my phone is 0412345678"
        result = self.normaliser.normalize_text(text)
        
        # Should remain unchanged
        assert result == text
    
    def test_boundary_conditions(self):
        """Test boundary conditions and special cases."""
        # Very long spelled-out sequence
        text = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
        result = self.normaliser.normalize_spelled_letters(text)
        assert result == "abcdefghijklmnopqrstuvwxyz"
        
        # Very long number sequence
        text = "zero one two three four five six seven eight nine"
        result = self.normaliser.normalize_number_words(text)
        assert result == "0123456789"
    
    def test_special_characters_handling(self):
        """Test handling of special characters."""
        text = "My name is j-o-h-n and email is john.doe at gmail dot com"
        result = self.normaliser.normalize_text(text)
        
        # Should not affect hyphenated names incorrectly
        assert "j-o-h-n" in result  # Should not be normalized
        assert "john.doe@gmail.com" in result  # Should be normalized


@pytest.mark.unit
class TestPerformance:
    """Test performance characteristics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.normaliser = TextNormaliser()
    
    def test_large_text_processing(self):
        """Test processing of large text blocks."""
        # Create a large text block
        large_text = "My name is j o h n d o e and my number is zero one two three. " * 1000
        
        # Should complete without errors
        result = self.normaliser.normalize_text(large_text)
        assert len(result) > 0
        assert "john doe" in result
        assert "0123" in result
    
    def test_pattern_compilation(self):
        """Test that patterns are compiled once during initialization."""
        # Create multiple instances and verify they work
        normaliser1 = TextNormaliser()
        normaliser2 = TextNormaliser()
        
        text = "j o h n at gmail dot com"
        result1 = normaliser1.normalize_text(text)
        result2 = normaliser2.normalize_text(text)
        
        assert result1 == result2
        assert "john@gmail.com" in result1 