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
    
    This class implements normalization capabilities using a two-step approach:
    1. Convert spelled-out numbers to digits (keeping spaces)
    2. Combine separated characters/numbers (letters or digits)
    3. Reconstruct email addresses from spoken form
    4. Remove conversational filler words (uh, huh, mm, hmm, etc.)
    
    Examples:
    - "my number is zero four one two" → "my number is 0412"
    - "my member id is a b c one two three" → "my member id is abc123"
    - "my email address is john doe at gmail dot com" → "my email address is john.doe@gmail.com"
    - "I uh need to um check my hmm account" → "I need to check my account"
    """
    
    def __init__(self, email_username_words=2, remove_fillers=True):
        """
        Initialize the text normaliser with predefined mappings.
        
        Args:
            email_username_words: Number of words before 'at' to consider as email username (default: 2)
            remove_fillers: Whether to automatically remove filler words during normalization (default: True)
        """
        self.email_username_words = email_username_words
        self.remove_fillers = remove_fillers
        # Number word mappings - comprehensive set of spoken numbers to digits
        # The main goal is to convert spoken numbers to digit format so they can be 
        # properly detected as GENERIC_NUMBER, PHONE_NUMBER, or other numeric PII by Presidio
        self.number_words = {
            # Single digits
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            # Teens
            'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14',
            'fifteen': '15', 'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19',
            # Tens
            'twenty': '20', 'thirty': '30', 'forty': '40', 'fifty': '50',
            'sixty': '60', 'seventy': '70', 'eighty': '80', 'ninety': '90'
        }
        
        # Filler words configuration
        self.filler_patterns = [
            r'\buh\b', r'\bhuh\b', r'\bmm+\b', r'\bhmm+\b', r'\bum+\b'
        ]        
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
            'mac': 'mac.com',
            'qq': 'qq.com'
        }
        
        # Patterns for different normalizations
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        # Pattern for sequences of single characters (letters or digits) separated by spaces
        # Simple pattern that matches sequences like "j o h n" or "1 2 3"
        self.separated_chars_pattern = re.compile(
            r'\b([a-zA-Z0-9](?:\s+[a-zA-Z0-9])+)\b',
            re.IGNORECASE
        )
        
        # Pattern for number words
        number_words_pattern = '|'.join(self.number_words.keys())
        self.number_words_pattern = re.compile(
            rf'\b(?:{number_words_pattern})(?:\s+(?:{number_words_pattern}))*\b',
            re.IGNORECASE
        )
        
        # Pattern for email addresses in spoken form
        # Matches username (1-4 words) + "at" + provider + "dot" + extension
        self.email_pattern = re.compile(
            r'([a-zA-Z][a-zA-Z0-9._-]*(?:\s+[a-zA-Z0-9._-]+){0,3})\s+at\s+([a-zA-Z0-9.-]+(?:\s+[a-zA-Z0-9.-]+)*)\s+dot\s+([a-zA-Z]{2,})\b',
            re.IGNORECASE
        )
        
        # Compile filler word patterns
        self.filler_word_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.filler_patterns]
    
    def normalize_separated_chars(self, text: str) -> str:
        """
        Combine separated characters (letters or digits) into continuous sequences.
        
        Example: "j o h n d o e" → "johndoe", "1 2 3" → "123"
        
        Args:
            text: Input text containing separated characters
            
        Returns:
            Text with separated characters combined
        """
        def replace_separated(match):
            full_match = match.group(1)  # Get the captured group
            # Split by spaces and join without spaces
            chars = full_match.split()
            combined = ''.join(chars)
            return combined
        
        # Apply the pattern to combine separated characters
        result = self.separated_chars_pattern.sub(replace_separated, text)
        
        logger.debug(f"Separated chars normalization: '{text}' → '{result}'")
        return result
    
    def normalize_number_words(self, text: str) -> str:
        """
        Convert number words to digits for improved PII detection.
        
        The primary purpose is to convert spoken number representations into digit format
        so that Presidio can properly identify them as PII (GENERIC_NUMBER, PHONE_NUMBER, etc.).
        We prioritize detection over numerical precision, simply mapping each number word 
        directly to its corresponding digit representation.
        
        Examples: 
        - "zero four one two three four five six seven" → "041234567"
        
        Args:
            text: Input text containing number words
            
        Returns:
            Text with number words converted to digits
        """
        def replace_numbers(match):
            number_sequence = match.group(0)
            words = number_sequence.lower().split()
            
            # Simply map each number word to its digit representation
            digits = ''.join(self.number_words.get(word, word) for word in words)
            return digits
        
        normalized = self.number_words_pattern.sub(replace_numbers, text)
        
        logger.debug(f"Number words normalization: '{text}' → '{normalized}'")
        return normalized
    
    def normalize_email_addresses(self, text: str) -> str:
        """
        Convert spoken email addresses to proper format.
        
        Takes the last N words before "at" as the username (configurable, default=2).
        
        Examples:
        - "my email is john doe at gmail dot com" → "my email is john.doe@gmail.com"
        - "contact mary smith at hotmail dot com" → "contact mary.smith@hotmail.com"
        
        Args:
            text: Input text containing spoken email addresses
            
        Returns:
            Text with email addresses converted to proper format
        """
        # Find email patterns: (username words) at (provider) dot (extension)
        # Use a restrictive pattern that limits both username and domain parts
        # to avoid matching across multiple emails
        max_username_words = max(self.email_username_words + 1, 4)  # Allow some flexibility
        email_pattern = re.compile(
            rf'(?<!\w)(\w+(?:\s+\w+){{0,{max_username_words-1}}})\s+at\s+([a-zA-Z0-9.-]+(?:\s+[a-zA-Z0-9.-]+){{0,2}})\s+dot\s+([a-zA-Z]{{2,}})\b(?!\s+\w+\s+at)',
            re.IGNORECASE
        )
        
        result = text
        
        # Process all matches from end to beginning to avoid offset issues
        matches = list(email_pattern.finditer(text))
        
        for match in reversed(matches):
            before_at_captured = match.group(1).strip()
            domain_part = match.group(2).strip()
            extension = match.group(3).strip()
            
            # Split the captured text before "at" into words
            words = before_at_captured.split()
            
            # Take the last N words as the username
            if len(words) >= self.email_username_words:
                username_words = words[-self.email_username_words:]
                remaining_words = words[:-self.email_username_words]
            else:
                username_words = words
                remaining_words = []
            
            # Create username by joining with dots
            username = '.'.join(username_words)
            
            # Normalize domain: remove spaces, check if it's a known provider
            domain = re.sub(r'\s+', '', domain_part.lower())
            
            # Check if domain is a known provider (without extension)
            if domain in self.email_providers:
                # Use the full domain from our mapping
                full_domain = self.email_providers[domain]
            else:
                # Construct domain with provided extension
                full_domain = f"{domain}.{extension.lower()}"
            
            # Construct the replacement
            email = f"{username}@{full_domain}"
            if remaining_words:
                replacement = f"{' '.join(remaining_words)} {email}"
            else:
                replacement = email
            
            # Replace in the result string
            result = result[:match.start()] + replacement + result[match.end():]
        
        logger.debug(f"Email normalization: '{text}' → '{result}'")
        return result
    
    def remove_filler_words(self, text: str, replacement=' ', keep_placeholder=False) -> str:
        """
        Remove or replace conversational filler words from transcript text.
        
        Examples:
        - "I uh need to mm mm check my hmm account" → "I need to check my account"
        - "You know, it's uh really important" → "it's really important"
        
        Args:
            text: Input text containing filler words
            replacement: String to replace fillers with (default: empty string)
            keep_placeholder: If True, replace with <FILLER> token instead of removing
            
        Returns:
            Text with filler words removed or replaced
        """
        if not text or not isinstance(text, str):
            return text
            
        result = text
        
        # Process each filler pattern
        for pattern in self.filler_word_patterns:
            if keep_placeholder:
                placeholder = ' <FILLER> '
                result = pattern.sub(placeholder, result)
            else:
                result = pattern.sub(replacement, result)
        
        # Clean up multiple spaces and whitespace at ends
        result = re.sub(r'\s{2,}', ' ', result).strip()
        
        logger.debug(f"Filler word removal: '{text}' → '{result}'")
        return result       

    
    def normalize_text(self, text: str) -> str:
        """
        Apply all normalization rules to the input text.
        
        Uses a multi-step approach:
        1. First convert spelled-out numbers to digits (keeping spaces)
        2. Then combine separated characters/numbers
        3. Normalize email addresses
        4. Remove filler words (if enabled)
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text with all rules applied
        """
        if not text or not isinstance(text, str):
            return text
        
        # Step 1: Convert spelled-out numbers to digits (keeping spaces)
        normalized = self.normalize_number_words(text)
        
        # Step 2: Combine separated characters/numbers
        normalized = self.normalize_separated_chars(normalized)
        
        # Step 3: Normalize email addresses
        normalized = self.normalize_email_addresses(normalized)
        
        # Step 4: Remove filler words (if enabled)
        if self.remove_fillers:
            normalized = self.remove_filler_words(normalized)
        
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
def normalize_text(text: str, email_username_words: int = 2) -> str:
    """
    Convenience function to normalize text using default normaliser.
    
    Args:
        text: Input text to normalize
        email_username_words: Number of words before 'at' to consider as email username
        
    Returns:
        Normalized text
    """
    normaliser = TextNormaliser(email_username_words=email_username_words)
    return normaliser.normalize_text(text)


def normalize_separated_chars(text: str) -> str:
    """
    Convenience function to normalize only separated characters.
    
    Args:
        text: Input text containing separated characters
        
    Returns:
        Text with separated characters combined
    """
    normaliser = TextNormaliser()
    return normaliser.normalize_separated_chars(text)


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


def normalize_email_addresses(text: str, email_username_words: int = 2) -> str:
    """
    Convenience function to normalize only email addresses.
    
    Args:
        text: Input text containing spoken email addresses
        email_username_words: Number of words before 'at' to consider as email username
        
    Returns:
        Text with email addresses converted to proper format
    """
    normaliser = TextNormaliser(email_username_words=email_username_words)
    return normaliser.normalize_email_addresses(text)


# Legacy function for backward compatibility
def normalize_spelled_letters(text: str) -> str:
    """
    Legacy function for backward compatibility.
    Now uses normalize_separated_chars internally.
    
    Args:
        text: Input text containing separated characters
        
    Returns:
        Text with separated characters combined
    """
    return normalize_separated_chars(text)
