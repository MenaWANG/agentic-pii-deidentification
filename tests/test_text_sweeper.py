"""
Tests for text_sweeper.py functionality.

This module tests the TextSweeper class and related functions for pattern-based replacements
in the PII deidentification pipeline.
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, List

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.text_sweeper import (
    TextSweeper, 
    sweep_text, 
    sweep_months,
    sweep_ordinals
)

class TestTextSweeper:
    """Test cases for the TextSweeper class."""

    @pytest.fixture
    def sweeper(self):
        """Create a TextSweeper instance for testing."""
        return TextSweeper()

    def test_initialization(self, sweeper):
        """Test that sweeper initializes correctly with default patterns."""
        assert sweeper is not None
        assert len(sweeper.month_names) > 0
        assert len(sweeper.ordinal_word_patterns) > 0
        assert isinstance(sweeper.replacement_dict, dict)

    def test_compile_patterns(self, sweeper):
        """Test that patterns are properly compiled."""
        assert sweeper.month_pattern is not None
        assert sweeper.ordinal_number_pattern is not None
        assert sweeper.ordinal_word_pattern is not None

    def test_sweep_months_full_names(self, sweeper):
        """Test that full month names are replaced."""
        test_text = "I was born in January and moved in December."
        result = sweeper.sweep_months(test_text)
        assert "January" not in result
        assert "December" not in result
        assert "<MONTH>" in result
        assert result == "I was born in <MONTH> and moved in <MONTH>."

    def test_sweep_months_abbreviations(self, sweeper):
        """Test that abbreviated month names are replaced."""
        test_text = "Meeting on Jan 15th and another on Dec 20th."
        result = sweeper.sweep_months(test_text)
        assert "Jan" not in result
        assert "Dec" not in result
        assert "<MONTH>" in result
        assert result == "Meeting on <MONTH> 15th and another on <MONTH> 20th."

    def test_sweep_months_mixed_case(self, sweeper):
        """Test that month names with mixed case are replaced."""
        test_text = "Events in january, FEBRUARY, and March."
        result = sweeper.sweep_months(test_text)
        assert "january" not in result
        assert "FEBRUARY" not in result
        assert "March" not in result
        assert result == "Events in <MONTH>, <MONTH>, and <MONTH>."

    def test_sweep_months_custom_replacement(self, sweeper):
        """Test month replacement with custom placeholder."""
        test_text = "Born in April and died in October."
        result = sweeper.sweep_months(test_text, replacement="[MONTH]")
        assert "April" not in result
        assert "October" not in result
        assert "[MONTH]" in result
        assert result == "Born in [MONTH] and died in [MONTH]."

    def test_sweep_ordinal_numbers_digits(self, sweeper):
        """Test that ordinal numbers in digit form are replaced."""
        test_text = "The 1st prize, 2nd place, 3rd spot, and 4th position."
        result = sweeper.sweep_ordinal_numbers(test_text)
        assert "1st" not in result
        assert "2nd" not in result
        assert "3rd" not in result
        assert "4th" not in result
        assert "<GENERIC_NUMBER>" in result
        assert result == "The <GENERIC_NUMBER> prize, <GENERIC_NUMBER> place, <GENERIC_NUMBER> spot, and <GENERIC_NUMBER> position."

    def test_sweep_ordinal_numbers_words(self, sweeper):
        """Test that ordinal numbers in word form are replaced."""
        test_text = "The first prize, second place, third spot, and fourth position."
        result = sweeper.sweep_ordinal_numbers(test_text)
        # assert "first" not in result # decide to keep "first" as it may carry other meanings in a setence
        assert "second" not in result
        assert "third" not in result
        assert "fourth" not in result
        assert "<GENERIC_NUMBER>" in result
        assert result == "The first prize, <GENERIC_NUMBER> place, <GENERIC_NUMBER> spot, and <GENERIC_NUMBER> position."

    def test_sweep_ordinal_numbers_compound_words(self, sweeper):
        """Test that compound ordinal words are replaced."""
        test_text = "The twenty first century and the thirty-first day."
        result = sweeper.sweep_ordinal_numbers(test_text)
        assert "twenty first" not in result
        assert "thirty-first" not in result
        assert "<GENERIC_NUMBER>" in result
        assert result == "The <GENERIC_NUMBER> century and the <GENERIC_NUMBER> day."

    def test_sweep_ordinal_numbers_custom_replacement(self, sweeper):
        """Test ordinal replacement with custom placeholder."""
        test_text = "The 5th element and the twentieth century."
        result = sweeper.sweep_ordinal_numbers(test_text, replacement="[NUMBER]")
        assert "5th" not in result
        assert "twentieth" not in result
        assert "[NUMBER]" in result
        assert result == "The [NUMBER] element and the [NUMBER] century."

    def test_sweep_custom_entities_basic(self, sweeper):
        """Test basic custom entity replacement."""
        test_text = "John Smith lives in Sydney and works at Google."
        entity_dict = {
            '<PERSON>': ['John Smith', 'Jane Doe'],
            '<LOCATION>': ['Sydney', 'Melbourne'],
            '<ORGANIZATION>': ['Google', 'Microsoft']
        }
        result = sweeper.sweep_custom_entities(test_text, entity_dict)
        assert "John Smith" not in result
        assert "Sydney" not in result
        assert "Google" not in result
        assert "<PERSON>" in result
        assert "<LOCATION>" in result
        assert "<ORGANIZATION>" in result
        assert result == "<PERSON> lives in <LOCATION> and works at <ORGANIZATION>."

    def test_sweep_custom_entities_case_insensitive(self, sweeper):
        """Test that custom entity replacement is case-insensitive."""
        test_text = "john smith lives in SYDNEY."
        entity_dict = {
            '<PERSON>': ['John Smith'],
            '<LOCATION>': ['Sydney']
        }
        result = sweeper.sweep_custom_entities(test_text, entity_dict)
        assert "john smith" not in result.lower()
        assert "sydney" not in result.lower()
        assert "<PERSON>" in result
        assert "<LOCATION>" in result
        assert result == "<PERSON> lives in <LOCATION>."

    def test_sweep_custom_entities_partial_match(self, sweeper):
        """Test that only complete words/phrases are replaced."""
        test_text = "Joe is not John, and Swiss is not Smith."
        entity_dict = {
            '<PERSON>': ['John', 'Smith']
        }
        result = sweeper.sweep_custom_entities(test_text, entity_dict)
        assert "Joe" in result  # Should not be replaced
        assert "Swiss" in result  # Should not be replaced
        assert "John" not in result
        assert "Smith" not in result
        assert result == "Joe is not <PERSON>, and Swiss is not <PERSON>."

    def test_set_replacement_dict(self, sweeper):
        """Test setting the replacement dictionary."""
        entity_dict = {
            '<PERSON>': ['Alice', 'Bob'],
            '<LOCATION>': ['London', 'Paris']
        }
        sweeper.set_replacement_dict(entity_dict)
        assert sweeper.replacement_dict == entity_dict
        
        # Test that it's used in sweep_custom_entities when no dict is provided
        test_text = "Alice lives in Paris."
        result = sweeper.sweep_custom_entities(test_text)
        assert "Alice" not in result
        assert "Paris" not in result
        assert result == "<PERSON> lives in <LOCATION>."

    def test_sweep_text_all_features(self, sweeper):
        """Test the combined sweeping with all features enabled."""
        test_text = "John Smith was born on January 1st, 1990 in Sydney."
        entity_dict = {
            '<PERSON>': ['John Smith'],
            '<LOCATION>': ['Sydney']
        }
        result = sweeper.sweep_text(
            text=test_text,
            sweep_months=True,
            sweep_ordinals=True,
            custom_dict=entity_dict
        )
        assert "John Smith" not in result
        assert "January" not in result
        assert "1st" not in result
        assert "Sydney" not in result
        assert "<PERSON>" in result
        assert "<MONTH>" in result
        assert "<GENERIC_NUMBER>" in result
        assert "<LOCATION>" in result
        assert result == "<PERSON> was born on <MONTH> <GENERIC_NUMBER>, 1990 in <LOCATION>."

    def test_sweep_text_selective_features(self, sweeper):
        """Test selective enabling of sweeping features."""
        test_text = "John Smith was born on January 1st, 1990 in Sydney."
        entity_dict = {
            '<PERSON>': ['John Smith'],
            '<LOCATION>': ['Sydney']
        }
        
        # Only sweep months
        result_months_only = sweeper.sweep_text(
            text=test_text,
            sweep_months=True,
            sweep_ordinals=False,
            custom_dict=None
        )
        assert "John Smith" in result_months_only
        assert "January" not in result_months_only
        assert "1st" in result_months_only
        assert "Sydney" in result_months_only
        
        # Only sweep ordinals
        result_ordinals_only = sweeper.sweep_text(
            text=test_text,
            sweep_months=False,
            sweep_ordinals=True,
            custom_dict=None
        )
        assert "John Smith" in result_ordinals_only
        assert "January" in result_ordinals_only
        assert "1st" not in result_ordinals_only
        assert "Sydney" in result_ordinals_only
        
        # Only custom entities
        result_custom_only = sweeper.sweep_text(
            text=test_text,
            sweep_months=False,
            sweep_ordinals=False,
            custom_dict=entity_dict
        )
        assert "John Smith" not in result_custom_only
        assert "January" in result_custom_only
        assert "1st" in result_custom_only
        assert "Sydney" not in result_custom_only

    def test_sweep_text_empty_input(self, sweeper):
        """Test sweeping with empty input."""
        assert sweeper.sweep_text("") == ""
        assert sweeper.sweep_text(None) is None

    def test_sweep_text_no_matches(self, sweeper):
        """Test sweeping with no pattern matches."""
        test_text = "This contains no months, ordinals, or custom entities."
        result = sweeper.sweep_text(test_text)
        assert result == test_text  # Should be unchanged


class TestConvenienceFunctions:
    """Test convenience functions for direct usage."""

    def test_sweep_text_function(self):
        """Test the sweep_text convenience function."""
        test_text = "Meeting on march 1st with john."
        entity_dict = {'<PERSON>': ['John']}
        result = sweep_text(
            text=test_text,
            sweep_months=True,
            sweep_ordinals=True,
            custom_dict=entity_dict
        )
        assert "march" not in result #case insensitive
        assert "1st" not in result
        assert "john" not in result #case insensitive
        assert "<MONTH>" in result
        assert "<GENERIC_NUMBER>" in result
        assert "<PERSON>" in result

    def test_sweep_months_function(self):
        """Test the sweep_months convenience function."""
        test_text = "Meeting in February and March."
        result = sweep_months(test_text)
        assert "February" not in result
        assert "March" not in result
        assert "<MONTH>" in result
        assert result == "Meeting in <MONTH> and <MONTH>."

    def test_sweep_ordinals_function(self):
        """Test the sweep_ordinals convenience function."""
        test_text = "The 5th and sixth items."
        result = sweep_ordinals(test_text)
        assert "5th" not in result
        assert "sixth" not in result
        assert "<GENERIC_NUMBER>" in result
        assert result == "The <GENERIC_NUMBER> and <GENERIC_NUMBER> items."


@pytest.mark.integration
class TestTextSweeperIntegration:
    """Integration tests for TextSweeper with complex scenarios."""

    def test_complex_mixed_scenario(self):
        """Test a complex scenario with mixed content types."""
        test_text = """
        John Smith has an appointment on january 21st, 2023 at 3:00 PM.
        He will meet with Jane Doe to discuss the first quarter results.
        The meeting will take place in Sydney office on the 3rd floor.
        John's email is john.smith@example.com and he lives in Melbourne.
        The second meeting is scheduled for Feb 15th.
        """
        
        entity_dict = {
            '<PERSON>': ['John Smith', 'Jane Doe', 'John', 'Jane'],
            '<LOCATION>': ['Sydney', 'Melbourne']
        }
        
        sweeper = TextSweeper()
        result = sweeper.sweep_text(
            text=test_text,
            sweep_months=True,
            sweep_ordinals=True,
            custom_dict=entity_dict
        )
        
        # Check that all entities are properly replaced
        assert "John Smith" not in result
        assert "Jane Doe" not in result
        assert "John's" not in result  # Should catch possessive forms too
        assert "january" not in result
        assert "Feb" not in result
        assert "21st" not in result
        # assert "first" not in result
        assert "3rd" not in result
        assert "15th" not in result
        assert "Sydney" not in result
        assert "Melbourne" not in result
        assert "john.smith@example.com" not in result
        
        # Check that replacements are present
        assert "<PERSON>" in result
        assert "<LOCATION>" in result
        assert "<MONTH>" in result
        assert "<GENERIC_NUMBER>" in result
        
        # Check that non-target text is preserved
        assert "has an appointment on" in result
        assert "2023 at 3:00 PM" in result
        assert "to discuss the" in result
        assert "quarter results" in result
        assert "meeting will take place in" in result
        assert "floor" in result
        assert "email is" in result
        assert "lives in" in result
        assert "meeting is scheduled for" in result
    
    def test_multi_word_phrases_and_boundaries(self):
        """Test proper handling of multi-word phrases and word boundaries."""
        test_text = """
        John Smith and John Smithson are different people.
        JohnSmith (no space) shouldn't match. But John should match on its own.
        Sydney, Australia is a location. Sydneyville is not in our list.
        """
        
        entity_dict = {
            '<PERSON>': ['John Smith', 'John'],
            '<LOCATION>': ['Sydney']
        }
        
        sweeper = TextSweeper()
        result = sweeper.sweep_text(
            text=test_text,
            sweep_months=False,
            sweep_ordinals=False,
            custom_dict=entity_dict
        )
        
        # These should be replaced
        assert "John Smith" not in result
        assert "John " not in result  # John as a standalone word should be replaced
        
        # These should NOT be replaced
        assert "Smithson" in result
        assert "JohnSmith" in result
        assert "Sydneyville" in result
        
        # Check replacements
        assert "<PERSON> and <PERSON> Smithson" in result
        assert "But <PERSON> should match" in result
        assert "<LOCATION>, Australia" in result


@pytest.mark.integration
class TestTranscriptFieldSweeping:
    """Test the transcript field-based sweeping functionality."""
    
    @pytest.fixture
    def sweeper(self):
        """Create a TextSweeper instance for testing."""
        return TextSweeper()
    
    def test_sweep_transcript_fields_basic(self, sweeper):
        """Test basic transcript field sweeping."""
        transcript_row = {
            'member_first_name': 'John',
            'member_last_name': 'Smith',
            'member_mobile': '0412345678'
        }
        
        field_mapping = {
            '<FIRST_NAME>': ['member_first_name'],
            '<LAST_NAME>': ['member_last_name'],
            '<PHONE_NUMBER>': ['member_mobile']
        }
        
        test_text = "Hi, this is John Smith calling from 0412345678"
        result = sweeper.sweep_transcript_fields(test_text, transcript_row, field_mapping)
        
        assert "John" not in result
        assert "Smith" not in result
        assert "0412345678" not in result
        assert "<FIRST_NAME>" in result
        assert "<LAST_NAME>" in result
        assert "<PHONE_NUMBER>" in result
        assert result == "Hi, this is <FIRST_NAME> <LAST_NAME> calling from <PHONE_NUMBER>"
    
    def test_sweep_transcript_fields_multiple_fields_per_type(self, sweeper):
        """Test sweeping with multiple fields mapping to the same placeholder."""
        transcript_row = {
            'member_first_name': 'John',
            'member_last_name': 'Smith',
            'agent_first_name': 'Jane',
            'agent_last_name': 'Doe'
        }
        
        field_mapping = {
            '<FIRST_NAME>': ['member_first_name', 'agent_first_name'],
            '<LAST_NAME>': ['member_last_name', 'agent_last_name']
        }
        
        test_text = "Agent Jane Doe speaking with John Smith"
        result = sweeper.sweep_transcript_fields(test_text, transcript_row, field_mapping)
        
        assert "John" not in result
        assert "Smith" not in result
        assert "Jane" not in result
        assert "Doe" not in result
        assert result == "Agent <FIRST_NAME> <LAST_NAME> speaking with <FIRST_NAME> <LAST_NAME>"
    
    def test_sweep_transcript_fields_missing_fields(self, sweeper):
        """Test handling of missing or empty fields in transcript data."""
        transcript_row = {
            'member_first_name': 'John',
            'member_last_name': '',  # Empty field
            'member_mobile': None    # None value
        }
        
        field_mapping = {
            '<PERSON>': ['member_first_name', 'member_last_name'],
            '<PHONE_NUMBER>': ['member_mobile']
        }
        
        test_text = "Hi, this is John calling"
        result = sweeper.sweep_transcript_fields(test_text, transcript_row, field_mapping)
        
        assert "John" not in result
        assert result == "Hi, this is <PERSON> calling"
    
    def test_sweep_transcript_fields_case_insensitive(self, sweeper):
        """Test case-insensitive field value matching."""
        transcript_row = {
            'member_first_name': 'John',
            'member_last_name': 'Smith'
        }
        
        field_mapping = {
            '<PERSON>': ['member_first_name', 'member_last_name']
        }
        
        test_text = "This is JOHN SMITH and john smith"
        result = sweeper.sweep_transcript_fields(test_text, transcript_row, field_mapping)
        
        assert "JOHN" not in result
        assert "SMITH" not in result
        assert "john" not in result
        assert "smith" not in result
        assert result == "This is <PERSON> <PERSON> and <PERSON> <PERSON>"
    
    def test_sweep_transcript_fields_empty_inputs(self, sweeper):
        """Test handling of empty or invalid inputs."""
        transcript_row = {
            'member_first_name': 'John',
            'member_last_name': 'Smith'
        }
        
        field_mapping = {
            '<PERSON>': ['member_first_name', 'member_last_name']
        }
        
        # Empty text
        assert sweeper.sweep_transcript_fields("", transcript_row, field_mapping) == ""
        
        # None text
        assert sweeper.sweep_transcript_fields(None, transcript_row, field_mapping) is None
        
        # Empty transcript row
        assert sweeper.sweep_transcript_fields("Hi John", {}, field_mapping) == "Hi John"
        
        # Empty field mapping
        assert sweeper.sweep_transcript_fields("Hi John", transcript_row, {}) == "Hi John"
        
        # None inputs
        assert sweeper.sweep_transcript_fields("Hi John", None, field_mapping) == "Hi John"
        assert sweeper.sweep_transcript_fields("Hi John", transcript_row, None) == "Hi John"
    
    def test_sweep_text_with_transcript_fields(self, sweeper):
        """Test integration of transcript field sweeping with other sweeping features."""
        transcript_row = {
            'member_first_name': 'John',
            'member_last_name': 'Smith',
            'member_mobile': '0412345678'
        }
        
        field_mapping = {
            '<FIRST_NAME>': ['member_first_name'],
            '<LAST_NAME>': ['member_last_name'],
            '<PHONE_NUMBER>': ['member_mobile']
        }
        
        test_text = "John Smith called on January 21st from 0412345678"
        
        # Test with all features enabled
        result = sweeper.sweep_text(
            text=test_text,
            sweep_months=True,
            sweep_ordinals=True,
            transcript_row=transcript_row,
            field_mapping=field_mapping
        )
        
        assert "John" not in result
        assert "Smith" not in result
        assert "January" not in result
        assert "21st" not in result
        assert "0412345678" not in result
        assert "<FIRST_NAME>" in result
        assert "<LAST_NAME>" in result
        assert "<MONTH>" in result
        assert "<GENERIC_NUMBER>" in result
        assert "<PHONE_NUMBER>" in result
        assert result == "<FIRST_NAME> <LAST_NAME> called on <MONTH> <GENERIC_NUMBER> from <PHONE_NUMBER>"
        
        # Test with selective features
        result_partial = sweeper.sweep_text(
            text=test_text,
            sweep_months=False,
            sweep_ordinals=False,
            transcript_row=transcript_row,
            field_mapping=field_mapping
        )
        
        assert "John" not in result_partial
        assert "Smith" not in result_partial
        assert "January" in result_partial  # Month not swept
        assert "21st" in result_partial     # Ordinal not swept
        assert "0412345678" not in result_partial
        assert result_partial == "<FIRST_NAME> <LAST_NAME> called on January 21st from <PHONE_NUMBER>"
    
    def test_sweep_text_convenience_function(self):
        """Test the convenience function with transcript field sweeping."""
        transcript_row = {
            'member_first_name': 'John',
            'member_last_name': 'Smith'
        }
        
        field_mapping = {
            '<FIRST_NAME>': ['member_first_name'],
            '<LAST_NAME>': ['member_last_name']
        }
        
        test_text = "Call from John Smith on January 1st"
        
        result = sweep_text(
            text=test_text,
            sweep_months=True,
            sweep_ordinals=True,
            transcript_row=transcript_row,
            field_mapping=field_mapping
        )
        
        assert "John" not in result
        assert "Smith" not in result
        assert "January" not in result
        assert "1st" not in result
        assert "<FIRST_NAME>" in result
        assert "<LAST_NAME>" in result
        assert "<MONTH>" in result
        assert "<GENERIC_NUMBER>" in result
        assert result == "Call from <FIRST_NAME> <LAST_NAME> on <MONTH> <GENERIC_NUMBER>"
