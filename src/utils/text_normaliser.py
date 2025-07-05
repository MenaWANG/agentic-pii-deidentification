"""
Text Normalisation Layer for PII Deidentification

A lightweight, rule-based pre-processor that converts spoken forms into canonical
digits and symbols before Presidio runs to improve recall while keeping costs low.

This module provides:
1. Spelled-out letter combination: " d o e " → " doe "
2. Number conversion: "zero four one two three four five six seven" → "041234567"
3. Email conversion: "john doe at gmail dot com" → "john.doe@gmail.com"
"""

import re
from typing import Dict, List, Tuple
import logging


logger = logging.getLogger(__name__)


class TextNormaliser:
    """
    Rule-based text normaliser for converting spoken forms to canonical text.
    
    This class implements three main normalization capabilities:
    - Spelled-out letter combination
    - Number word to digit conversion
    - Email address reconstruction from spoken form
    """
    
    def __init__(self):
        """Initialize the text normaliser with predefined mappings."""
        self.number_words = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
        }
        
        # Common email providers and their variations
        self.email_providers = {
            'gmail': 'gmail.com',
            'google': 'gmail.com',
            'hotmail': 'hotmail.com',
            'outlook': 'outlook.com',
            'yahoo': 'yahoo.com',
            'icloud': 'icloud.com',
            'apple': 'icloud.com',
            'aol': 'aol.com',
            'protonmail': 'protonmail.com',
            'proton': 'protonmail.com',
            'yandex': 'yandex.com',
            'zoho': 'zoho.com',
            'live': 'live.com',
            'msn': 'msn.com',
            'mail': 'mail.com',
            'gmx': 'gmx.com',
            'fastmail': 'fastmail.com',
            'me': 'me.com',
            'mac': 'mac.com'
        }
        
        # Patterns for different normalizations
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        # Pattern for spelled-out letters (e.g., " d o e " or "d o e")
        self.spelled_letters_pattern = re.compile(
            r'\b([a-zA-Z])\s+([a-zA-Z])\s+([a-zA-Z])\b|\b([a-zA-Z])\s+([a-zA-Z])\b|\b([a-zA-Z])\s+([a-zA-Z])\s+([a-zA-Z])\s+([a-zA-Z])\b|\b([a-zA-Z])\s+([a-zA-Z])\s+([a-zA-Z])\s+([a-zA-Z])\s+([a-zA-Z])\b|\b([a-zA-Z])\s+([a-zA-Z])\s+([a-zA-Z])\s+([a-zA-Z])\s+([a-zA-Z])\s+([a-zA-Z])\b',
            re.IGNORECASE
        )
        
        # More flexible pattern for any sequence of single letters separated by spaces
        self.flexible_letters_pattern = re.compile(
            r'\b([a-zA-Z])(?:\s+([a-zA-Z]))+\b',
            re.IGNORECASE
        )
        
        # Pattern for number words
        number_words_pattern = '|'.join(self.number_words.keys())
        self.number_words_pattern = re.compile(
            rf'\b(?:{number_words_pattern})(?:\s+(?:{number_words_pattern}))*\b',
            re.IGNORECASE
        )
        
        # Pattern for email addresses in spoken form
        # Matches patterns like "john doe at gmail dot com" or "user at domain dot extension"
        self.email_pattern = re.compile(
            r'\b([a-zA-Z0-9._-]+(?:\s+[a-zA-Z0-9._-]+)*)\s+at\s+([a-zA-Z0-9.-]+(?:\s+[a-zA-Z0-9.-]+)*)\s+dot\s+([a-zA-Z]{2,})\b',
            re.IGNORECASE
        )
    
    def normalize_spelled_letters(self, text: str) -> str:
        """
        Combine spelled-out letters into words.
        
        Example: " d o e " → " doe "
        
        Args:
            text: Input text containing spelled-out letters
            
        Returns:
            Text with spelled-out letters combined into words
        """
        def replace_letters(match):
            # Extract all non-None groups (letters)
            letters = [g for g in match.groups() if g is not None]
            return ''.join(letters)
        
        # Use the flexible pattern to match any sequence of single letters
        def replace_flexible(match):
            full_match = match.group(0)
            # Split by spaces and join without spaces
            letters = full_match.split()
            return ''.join(letters)
        
        # Apply the flexible pattern
        normalized = self.flexible_letters_pattern.sub(replace_flexible, text)
        
        logger.debug(f"Spelled letters normalization: '{text}' → '{normalized}'")
        return normalized
    
    def normalize_number_words(self, text: str) -> str:
        """
        Convert number words to digits.
        
        Example: "zero four one two three four five six seven" → "041234567"
        
        Args:
            text: Input text containing number words
            
        Returns:
            Text with number words converted to digits
        """
        def replace_numbers(match):
            number_sequence = match.group(0)
            words = number_sequence.lower().split()
            digits = ''.join(self.number_words.get(word, word) for word in words)
            return digits
        
        normalized = self.number_words_pattern.sub(replace_numbers, text)
        
        logger.debug(f"Number words normalization: '{text}' → '{normalized}'")
        return normalized
    
    def normalize_email_addresses(self, text: str) -> str:
        """
        Convert spoken email addresses to proper format.
        
        Example: "john doe at gmail dot com" → "john.doe@gmail.com"
        
        Args:
            text: Input text containing spoken email addresses
            
        Returns:
            Text with email addresses converted to proper format
        """
        def replace_email(match):
            username_part = match.group(1).strip()
            domain_part = match.group(2).strip()
            extension = match.group(3).strip()
            
            # Normalize username: replace spaces with dots, remove extra spaces
            username = re.sub(r'\s+', '.', username_part.strip())
            
            # Normalize domain: remove spaces, check if it's a known provider
            domain = re.sub(r'\s+', '', domain_part.strip().lower())
            
            # Check if domain is a known provider (without extension)
            if domain in self.email_providers:
                # Use the full domain from our mapping
                full_domain = self.email_providers[domain]
            else:
                # Construct domain with provided extension
                full_domain = f"{domain}.{extension.lower()}"
            
            return f"{username}@{full_domain}"
        
        normalized = self.email_pattern.sub(replace_email, text)
        
        logger.debug(f"Email normalization: '{text}' → '{normalized}'")
        return normalized
    
    def normalize_text(self, text: str) -> str:
        """
        Apply all normalization rules to the input text.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text with all rules applied
        """
        if not text or not isinstance(text, str):
            return text
        
        # Apply normalizations in order
        # 1. First normalize spelled letters
        normalized = self.normalize_spelled_letters(text)
        
        # 2. Then normalize number words
        normalized = self.normalize_number_words(normalized)
        
        # 3. Finally normalize email addresses
        normalized = self.normalize_email_addresses(normalized)
        
        logger.info(f"Full normalization: '{text}' → '{normalized}'")
        return normalized
    
    def get_supported_email_providers(self) -> List[str]:
        """
        Get list of supported email providers.
        
        Returns:
            List of supported email provider names
        """
        return list(self.email_providers.keys())


# Convenience functions for direct usage
def normalize_text(text: str) -> str:
    """
    Convenience function to normalize text using default normaliser.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    normaliser = TextNormaliser()
    return normaliser.normalize_text(text)


def normalize_spelled_letters(text: str) -> str:
    """
    Convenience function to normalize only spelled-out letters.
    
    Args:
        text: Input text containing spelled-out letters
        
    Returns:
        Text with spelled-out letters combined
    """
    normaliser = TextNormaliser()
    return normaliser.normalize_spelled_letters(text)


def normalize_number_words(text: str) -> str:
    """
    Convenience function to normalize only number words.
    
    Args:
        text: Input text containing number words
        
    Returns:
        Text with number words converted to digits
    """
    normaliser = TextNormaliser()
    return normaliser.normalize_number_words(text)


def normalize_email_addresses(text: str) -> str:
    """
    Convenience function to normalize only email addresses.
    
    Args:
        text: Input text containing spoken email addresses
        
    Returns:
        Text with email addresses converted to proper format
    """
    normaliser = TextNormaliser()
    return normaliser.normalize_email_addresses(text) 