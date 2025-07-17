"""
Text Sweeping Layer for PII Deidentification

A complementary processing layer that applies targeted pattern-based replacements
for specific types of information that should be consistently masked.

This layer focuses on:
1. Calendar information (months, dates, ordinal numbers)
2. Custom dictionary-based entity replacement
3. Transcript field-based PII sweeping
"""

import re
from typing import Dict, List, Tuple, Optional, Union
import logging


logger = logging.getLogger(__name__)


class TextSweeper:
    """
    Rule-based text sweeper for targeted pattern-based replacements.
    
    This class implements sweeping capabilities to replace specific patterns
    that should be consistently masked before PII detection or after normalization.
    """
    
    def __init__(self):
        """Initialize the text sweeper with predefined patterns."""
        # Month names for replacement
        self.month_names = [
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december',
            'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec'
        ]
        
        # Ordinal suffixes and patterns
        self.ordinal_suffixes = ['st', 'nd', 'rd', 'th']
        self.ordinal_word_patterns = [
            'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
            'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth', 
            'eighteenth', 'nineteenth', 'twentieth', 'twenty-first', 'twenty-second', 'twenty-third', 
            'twenty-fourth', 'twenty-fifth', 'twenty-sixth', 'twenty-seventh', 'twenty-eighth', 
            'twenty-ninth', 'thirtieth', 'thirty-first', 'twenty first', 'twenty second', 'twenty third', 
            'twenty fourth', 'twenty fifth', 'twenty sixth', 'twenty seventh', 'twenty eighth', 
            'twenty ninth', 'thirty first'
        ] # in voice to text, the hyphen is not always preserved
        
        # Custom entity replacements (can be updated at runtime)
        self.replacement_dict = {}
        
        # Compile regex patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        # Pattern for month names (case insensitive)
        month_pattern = '|'.join(self.month_names)
        self.month_pattern = re.compile(
            rf'\b({month_pattern})\b',
            re.IGNORECASE
        )
        
        # Pattern for ordinal numbers (1st, 2nd, 3rd, 4th, ..., 31st)
        self.ordinal_number_pattern = re.compile(
            r'\b([1-9]|[12][0-9]|3[01])(st|nd|rd|th)\b',
            re.IGNORECASE
        )
        
        # Pattern for ordinal words (first, second, third, ..., thirty-first)
        ordinal_words = '|'.join(self.ordinal_word_patterns)
        self.ordinal_word_pattern = re.compile(
            rf'\b({ordinal_words})\b',
            re.IGNORECASE
        )
    
    def sweep_months(self, text: str, replacement: str = '<MONTH>') -> str:
        """
        Replace month names with a placeholder.
        
        Examples:
        - "I was born in January" → "I was born in <MONTH>"
        - "The meeting is on Dec 5th" → "The meeting is on <MONTH> 5th"
        
        Args:
            text: Input text containing month names
            replacement: Placeholder to replace month names with
            
        Returns:
            Text with month names replaced
        """
        if not text or not isinstance(text, str):
            return text
        
        result = self.month_pattern.sub(replacement, text)
        return result
    
    def sweep_ordinal_numbers(self, text: str, replacement: str = '<GENERIC_NUMBER>') -> str:
        """
        Replace ordinal numbers and words with a placeholder.
        
        Examples:
        - "The meeting is on the 5th" → "The meeting is on the <GENERIC_NUMBER>"
        - "I'll be there on the twenty-first" → "I'll be there on the <GENERIC_NUMBER>"
        
        Args:
            text: Input text containing ordinal numbers
            replacement: Placeholder to replace ordinal numbers with
            
        Returns:
            Text with ordinal numbers replaced
        """
        if not text or not isinstance(text, str):
            return text
        
        # Replace numeric ordinals (1st, 2nd, etc.)
        result = self.ordinal_number_pattern.sub(replacement, text)
        
        # Replace word ordinals (first, second, etc.)
        result = self.ordinal_word_pattern.sub(replacement, result)
        
        return result
    
    def sweep_custom_entities(self, text: str, entity_dict: Optional[Dict[str, List[str]]] = None) -> str:
        """
        Replace custom entities based on a dictionary.
        
        The dictionary maps placeholders to lists of terms to be replaced.
        For example: {'<PERSON>': ['John', 'Smith'], '<LOCATION>': ['Sydney']}
        
        Args:
            text: Input text to sweep
            entity_dict: Dictionary mapping placeholders to terms to replace
                         If None, uses self.replacement_dict
            
        Returns:
            Text with custom entities replaced
        """
        if not text or not isinstance(text, str):
            return text
            
        if entity_dict is None:
            entity_dict = self.replacement_dict
            
        if not entity_dict:
            return text
            
        result = text
        
        # Apply each replacement set
        for placeholder, terms in entity_dict.items():
            if not terms:
                continue
                
            # Create pattern for the terms
            term_pattern = '|'.join(map(re.escape, terms))
            pattern = re.compile(rf'\b({term_pattern})\b', re.IGNORECASE)
            
            # Apply replacement
            result = pattern.sub(placeholder, result)
            
        return result
    
    def set_replacement_dict(self, entity_dict: Dict[str, List[str]]):
        """
        Set or update the custom replacement dictionary.
        
        Args:
            entity_dict: Dictionary mapping placeholders to terms to replace
        """
        self.replacement_dict = entity_dict
    
    def sweep_transcript_fields(self, text: str, transcript_row: dict, field_mapping: Dict[str, List[str]]) -> str:
        """
        Replace PII in text based on transcript field values.
        
        This method uses values from specific fields in the transcript data to sweep PII.
        For example, if field_mapping is {'<PERSON>': ['member_first_name', 'member_last_name']},
        it will use the values from those fields to replace matching text with '<PERSON>'.
        
        Args:
            text: Input text to sweep
            transcript_row: Dictionary containing the transcript data fields
            field_mapping: Dictionary mapping PII placeholders to lists of field names
                         e.g. {'<PERSON>': ['member_first_name', 'member_last_name'],
                              '<PHONE_NUMBER>': ['member_mobile', 'member_landline']}
            
        Returns:
            Text with PII from transcript fields replaced
        """
        if not text or not isinstance(text, str):
            return text
            
        if not transcript_row or not field_mapping:
            return text
            
        result = text
        
        # Create a temporary entity dictionary for the current transcript
        temp_entity_dict = {}
        
        # Build replacement lists from transcript fields
        for placeholder, field_list in field_mapping.items():
            values = []
            for field in field_list:
                if field in transcript_row and transcript_row[field]:
                    # Add the field value if it exists and is not empty
                    field_value = str(transcript_row[field]).strip()
                    if field_value:
                        values.append(field_value)
            
            if values:
                temp_entity_dict[placeholder] = values
        
        # Use the existing sweep_custom_entities method with our temporary dictionary
        if temp_entity_dict:
            result = self.sweep_custom_entities(result, temp_entity_dict)
            
        return result
    
    def sweep_text(self, text: str, 
                  sweep_months: bool = True, 
                  sweep_ordinals: bool = True,
                  custom_dict: Optional[Dict[str, List[str]]] = None,
                  transcript_row: Optional[dict] = None,
                  field_mapping: Optional[Dict[str, List[str]]] = None) -> str:
        """
        Apply all sweeping rules to the input text.
        
        Args:
            text: Input text to sweep
            sweep_months: Whether to replace month names
            sweep_ordinals: Whether to replace ordinal numbers
            custom_dict: Custom entity replacement dictionary
            transcript_row: Optional transcript data for field-based sweeping
            field_mapping: Optional mapping of PII placeholders to transcript field names
            
        Returns:
            Swept text with all selected rules applied
        """
        if not text or not isinstance(text, str):
            return text
        
        result = text
        
        # Step 1: Replace month names if enabled
        if sweep_months:
            result = self.sweep_months(result)
        
        # Step 2: Replace ordinal numbers if enabled
        if sweep_ordinals:
            result = self.sweep_ordinal_numbers(result)
        
        # Step 3: Apply transcript field-based sweeping if provided
        if transcript_row is not None and field_mapping is not None:
            result = self.sweep_transcript_fields(result, transcript_row, field_mapping)
        
        # Step 4: Apply custom entity replacements if provided
        if custom_dict is not None:
            # Temporarily set the replacement dictionary
            original_dict = self.replacement_dict
            self.set_replacement_dict(custom_dict)
            result = self.sweep_custom_entities(result)
            # Restore the original dictionary
            self.replacement_dict = original_dict
        elif self.replacement_dict:
            # Use the instance's replacement dictionary if available
            result = self.sweep_custom_entities(result)
        
        logger.info(f"Text sweeping: '{text[:50]}...' → '{result[:50]}...'")
        return result


# Convenience functions for direct usage
def sweep_text(text: str,
              sweep_months: bool = True,
              sweep_ordinals: bool = True,
              custom_dict: Optional[Dict[str, List[str]]] = None,
              transcript_row: Optional[dict] = None,
              field_mapping: Optional[Dict[str, List[str]]] = None) -> str:
    """
    Convenience function to sweep text using default sweeper.
    
    Args:
        text: Input text to sweep
        sweep_months: Whether to replace month names
        sweep_ordinals: Whether to replace ordinal numbers
        custom_dict: Custom entity replacement dictionary
        transcript_row: Optional transcript data for field-based sweeping
        field_mapping: Optional mapping of PII placeholders to transcript field names
        
    Returns:
        Swept text
    """
    sweeper = TextSweeper()
    return sweeper.sweep_text(text, sweep_months, sweep_ordinals, custom_dict, 
                            transcript_row, field_mapping)


def sweep_months(text: str, replacement: str = '<MONTH>') -> str:
    """
    Convenience function to replace only month names.
    
    Args:
        text: Input text containing month names
        replacement: Placeholder to replace month names with
        
    Returns:
        Text with month names replaced
    """
    sweeper = TextSweeper()
    return sweeper.sweep_months(text, replacement)


def sweep_ordinals(text: str, replacement: str = '<GENERIC_NUMBER>') -> str:
    """
    Convenience function to replace only ordinal numbers.
    
    Args:
        text: Input text containing ordinal numbers
        replacement: Placeholder to replace ordinal numbers with
        
    Returns:
        Text with ordinal numbers replaced
    """
    sweeper = TextSweeper()
    return sweeper.sweep_ordinal_numbers(text, replacement)
