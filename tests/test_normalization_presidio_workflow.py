"""
Tests for baseline Presidio framework functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from baseline.presidio_framework import PurePresidioFramework


@pytest.mark.unit
@pytest.mark.baseline
class TestPurePresidioFramework:
    """Test cases for the Pure Presidio Framework."""

    @pytest.fixture(autouse=True)
    def setup_method(self, mock_presidio_analyzer, mock_presidio_anonymizer):
        """Set up test fixtures."""
        with patch('baseline.presidio_framework.AnalyzerEngine', return_value=mock_presidio_analyzer), \
             patch('baseline.presidio_framework.AnonymizerEngine', return_value=mock_presidio_anonymizer):
            self.framework = PurePresidioFramework(enable_mlflow=False)

    def test_framework_initialization(self):
        """Test that framework initializes correctly."""
        assert self.framework is not None
        assert self.framework.analyzer is not None
        assert self.framework.anonymizer is not None
        assert not self.framework.enable_mlflow  # Should be disabled for testing
        assert self.framework.stats['total_transcripts'] == 0

    def test_process_transcript_basic(self):
        """Test basic transcript processing functionality."""
        # Setup mock analyzer results
        mock_result = Mock()
        mock_result.entity_type = 'PERSON'
        mock_result.start = 0
        mock_result.end = 4
        mock_result.score = 0.95
        
        self.framework.analyzer.analyze.return_value = [mock_result]
        
        # Setup mock anonymizer result
        mock_anonymized = Mock()
        mock_anonymized.text = "[PERSON] called customer service"
        self.framework.anonymizer.anonymize.return_value = mock_anonymized
        
        # Test processing
        test_text = "John called customer service"
        result = self.framework.process_transcript(test_text)
        
        # Verify results
        assert result['original_text'] == test_text
        assert result['anonymized_text'] == "[PERSON] called customer service"
        assert len(result['pii_detections']) == 1
        assert result['pii_detections'][0]['entity_type'] == 'PERSON'
        assert result['pii_detections'][0]['text'] == 'John'
        assert result['processing_time'] >= 0  # Allow for zero time in mocked tests
        assert result['workflow'] == 'pure_presidio'  # Verify workflow field
        assert 'custom_detections' in result  # Verify custom_detections field

    def test_process_transcript_no_pii(self):
        """Test processing transcript with no PII detected."""
        # Setup mocks for no PII
        self.framework.analyzer.analyze.return_value = []
        
        mock_anonymized = Mock()
        mock_anonymized.text = "Generic customer inquiry"
        self.framework.anonymizer.anonymize.return_value = mock_anonymized
        
        # Test processing
        test_text = "Generic customer inquiry"
        result = self.framework.process_transcript(test_text)
        
        # Verify results
        assert result['pii_count'] == 0
        assert len(result['pii_detections']) == 0
        assert result['anonymized_text'] == "Generic customer inquiry"
        assert result['workflow'] == 'pure_presidio'  # Verify workflow field
        assert 'custom_detections' in result  # Verify custom_detections field

    def test_process_transcript_multiple_pii(self):
        """Test processing transcript with multiple PII types."""
        # Setup multiple mock results
        mock_person = Mock()
        mock_person.entity_type = 'PERSON'
        mock_person.start = 0
        mock_person.end = 4
        mock_person.score = 0.95
        
        mock_email = Mock()
        mock_email.entity_type = 'EMAIL_ADDRESS'
        mock_email.start = 20
        mock_email.end = 35
        mock_email.score = 0.99
        
        self.framework.analyzer.analyze.return_value = [mock_person, mock_email]
        
        mock_anonymized = Mock()
        mock_anonymized.text = "[PERSON] email is [EMAIL]"
        self.framework.anonymizer.anonymize.return_value = mock_anonymized
        
        # Test processing
        test_text = "John email is john@test.com"
        result = self.framework.process_transcript(test_text)
        
        # Verify results
        assert result['pii_count'] == 2
        assert len(result['pii_detections']) == 2
        assert any(d['entity_type'] == 'PERSON' for d in result['pii_detections'])
        assert any(d['entity_type'] == 'EMAIL_ADDRESS' for d in result['pii_detections'])
        assert result['workflow'] == 'pure_presidio'  # Verify workflow field
        assert 'custom_detections' in result  # Verify custom_detections field

    def test_stats_update(self):
        """Test that statistics are properly updated."""
        # Setup mock for single PII detection
        mock_result = Mock()
        mock_result.entity_type = 'PERSON'
        mock_result.start = 0
        mock_result.end = 4
        mock_result.score = 0.95
        
        self.framework.analyzer.analyze.return_value = [mock_result]
        
        mock_anonymized = Mock()
        mock_anonymized.text = "[PERSON] called"
        self.framework.anonymizer.anonymize.return_value = mock_anonymized
        
        # Process transcript
        result = self.framework.process_transcript("John called")
        
        # Manually call stats update since it's called in process_dataset
        self.framework._update_stats(result)
        
        # Verify stats
        assert self.framework.stats['total_pii_detected'] == 1
        assert 'PERSON' in self.framework.stats['pii_types']

    def test_workflow_field_validation(self):
        """Test that workflow field is correctly set in process_transcript output."""
        # Setup minimal mock
        self.framework.analyzer.analyze.return_value = []
        
        mock_anonymized = Mock()
        mock_anonymized.text = "Test text"
        self.framework.anonymizer.anonymize.return_value = mock_anonymized
        
        # Test processing
        result = self.framework.process_transcript("Test text")
        
        # Verify workflow field specifically
        assert 'workflow' in result
        assert result['workflow'] == 'pure_presidio'
        assert isinstance(result['workflow'], str)

    @pytest.mark.parametrize("entity_type,expected_text", [
        ('PERSON', 'name'),
        ('EMAIL_ADDRESS', 'email'),
        ('PHONE_NUMBER', 'phone'),
        ('MEMBER_NUMBER', '12345678'),  # Custom recognizer
        ('AU_PHONE_NUMBER', '0412345678'),  # Custom recognizer
        ('GENERIC_NUMBER', '123456'),  # Custom recognizer
        ('AU_ADDRESS', '123 Main St'),  # Custom recognizer
    ])
    def test_different_entity_types(self, entity_type, expected_text):
        """Test processing different types of PII entities."""
        mock_result = Mock()
        mock_result.entity_type = entity_type
        mock_result.start = 0
        mock_result.end = len(expected_text)
        mock_result.score = 0.95
        
        self.framework.analyzer.analyze.return_value = [mock_result]
        
        mock_anonymized = Mock()
        mock_anonymized.text = f"[{entity_type}] detected"
        self.framework.anonymizer.anonymize.return_value = mock_anonymized
        
        # Test processing
        result = self.framework.process_transcript(expected_text)
        
        # Verify the entity type is correctly captured
        assert result['pii_detections'][0]['entity_type'] == entity_type
        assert result['workflow'] == 'pure_presidio'  # Verify workflow field
        assert 'custom_detections' in result  # Verify custom_detections field

    def test_custom_detections_tracking(self):
        """Test that custom PII detections are properly tracked."""
        # Setup mock results with custom entities
        mock_member_number = Mock()
        mock_member_number.entity_type = 'MEMBER_NUMBER'
        mock_member_number.start = 0
        mock_member_number.end = 8
        mock_member_number.score = 0.95
        
        mock_address = Mock()
        mock_address.entity_type = 'AU_ADDRESS'
        mock_address.start = 20
        mock_address.end = 35
        mock_address.score = 0.90
        
        self.framework.analyzer.analyze.return_value = [mock_member_number, mock_address]
        
        mock_anonymized = Mock()
        mock_anonymized.text = "[MEMBER_NUMBER] at [AU_ADDRESS]"
        self.framework.anonymizer.anonymize.return_value = mock_anonymized
        
        # Test processing
        result = self.framework.process_transcript("12345678 at 123 Main St")
        
        # Verify custom detections tracking
        assert 'custom_detections' in result
        assert result['custom_detections']['member_numbers'] == 1
        assert result['custom_detections']['addresses'] == 1
        assert result['pii_count'] == 2

@pytest.mark.integration
@pytest.mark.baseline
class TestPurePresidioFrameworkIntegration:
    """Integration tests for Pure Presidio Framework."""

    def test_process_dataset_integration(self, temp_csv_file):
        """Test dataset processing with mocked dependencies."""
        with patch('baseline.presidio_framework.AnalyzerEngine') as mock_analyzer_class, \
             patch('baseline.presidio_framework.AnonymizerEngine') as mock_anonymizer_class:
            
            # Setup mocks
            mock_analyzer = Mock()
            mock_anonymizer = Mock()
            mock_analyzer_class.return_value = mock_analyzer
            mock_anonymizer_class.return_value = mock_anonymizer
            
            # Mock analyzer to return simple PII detection
            mock_result = Mock()
            mock_result.entity_type = 'PERSON'
            mock_result.start = 17
            mock_result.end = 27
            mock_result.score = 0.95
            mock_analyzer.analyze.return_value = [mock_result]
            
            # Mock anonymizer
            mock_anonymized = Mock()
            mock_anonymized.text = "Hello, my name is [PERSON] and my email is [EMAIL]"
            mock_anonymizer.anonymize.return_value = mock_anonymized
            
            # Create framework and process dataset
            framework = PurePresidioFramework(enable_mlflow=False)
            results_df = framework.process_dataset(temp_csv_file)
            
            # Verify results
            assert len(results_df) == 2  # Should have 2 processed transcripts
            assert 'anonymized_text' in results_df.columns
            assert 'pii_detections' in results_df.columns
            assert 'processing_time' in results_df.columns
            assert 'workflow' in results_df.columns  # Verify workflow field in DataFrame
            assert 'custom_detections' in results_df.columns  # Verify custom_detections field
            
            # Verify all rows were processed
            assert all(results_df['pii_count'] >= 0)  # Should have non-negative PII counts
            assert all(results_df['workflow'] == 'pure_presidio')  # Verify workflow values

@pytest.mark.performance
@pytest.mark.baseline
class TestPurePresidioFrameworkPerformance:
    """Performance tests using real Presidio framework with actual PII data."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up real framework for performance testing."""
        self.framework = PurePresidioFramework(enable_mlflow=False)

    def test_person_name_detection(self):
        """Test that obvious person names are detected and anonymized."""
        test_text = "Hello, my name is David Smith and I need help with my account."
        
        result = self.framework.process_transcript(test_text)
        
        # Verify person name was detected
        person_detections = [d for d in result['pii_detections'] if d['entity_type'] == 'PERSON']
        assert len(person_detections) > 0, "Should detect at least one person name"
        
        # Verify David Smith was detected
        detected_names = [d['text'] for d in person_detections]
        assert any('David' in name or 'Smith' in name for name in detected_names), \
            f"Should detect David or Smith in names: {detected_names}"
        
        # Verify anonymization occurred
        assert '[PERSON]' in result['anonymized_text'] or '<PERSON>' in result['anonymized_text'], \
            f"Anonymized text should contain person placeholder: {result['anonymized_text']}"
        
        # Verify workflow and timing
        assert result['workflow'] == 'pure_presidio'
        # assert result['processing_time'] > 0 # this fails cause it runs that fast :P

    def test_email_detection(self):
        """Test that email addresses are detected and anonymized."""
        test_text = "Please contact me at david.smith@example.com for further assistance."
        
        result = self.framework.process_transcript(test_text)
        
        # Verify email was detected
        email_detections = [d for d in result['pii_detections'] if d['entity_type'] == 'EMAIL_ADDRESS']
        assert len(email_detections) > 0, "Should detect email address"
        
        # Verify specific email was detected
        detected_emails = [d['text'] for d in email_detections]
        assert 'david.smith@example.com' in detected_emails, \
            f"Should detect the specific email: {detected_emails}"
        
        # Verify anonymization occurred
        assert 'david.smith@example.com' not in result['anonymized_text'], \
            "Original email should not appear in anonymized text"
        assert '[EMAIL' in result['anonymized_text'] or '<EMAIL' in result['anonymized_text'], \
            f"Anonymized text should contain email placeholder: {result['anonymized_text']}"

    def test_generic_number_detection(self):
        """Test that generic numbers are detected and anonymized."""
        test_text = "My reference number is 98765432 and my account ID is 1234567890."
        
        result = self.framework.process_transcript(test_text)
        
        # Verify generic numbers were detected
        generic_detections = [d for d in result['pii_detections'] 
                            if d['entity_type'] in ['GENERIC_NUMBER', 'MEMBER_NUMBER']]
        assert len(generic_detections) > 0, "Should detect at least one generic number"
        
        # Verify specific numbers were detected
        detected_numbers = [d['text'] for d in generic_detections]
        assert any('98765432' in num or '1234567890' in num for num in detected_numbers), \
            f"Should detect one of the reference numbers: {detected_numbers}"
        
        # Verify anonymization occurred
        anonymized = result['anonymized_text']
        assert ('98765432' not in anonymized or '1234567890' not in anonymized), \
            "At least one original number should be anonymized"
        assert ('GENERIC_NUMBER' in anonymized or 'MEMBER_NUMBER' in anonymized or 
                '[' in anonymized), \
            f"Anonymized text should contain number placeholder: {anonymized}"
        
        # Verify custom detections tracking
        assert 'custom_detections' in result
        print(f"Generic number test - Original: {test_text}")
        print(f"Anonymized: {anonymized}")
        print(f"Detected numbers: {detected_numbers}")

    def test_australian_phone_variations(self):
        """Test various Australian phone number formats."""
        test_cases = [
            "Call me on 0412 345 678",  # Mobile with spaces
            "Phone: 0412345678",       # Mobile without spaces
            "Contact: (03) 9876 5432", # Landline with area code
            "Fax: 03 9876 5432",      # Landline variation
            "Mobile: +61 412 345 678", # International format
        ]
        
        for i, test_text in enumerate(test_cases):
            result = self.framework.process_transcript(test_text)
            
            # Verify phone detection
            phone_detections = [d for d in result['pii_detections'] 
                              if d['entity_type'] in ['PHONE_NUMBER', 'AU_PHONE_NUMBER']]
            
            print(f"Phone test {i+1}: {test_text}")
            print(f"  Detections: {len(phone_detections)}")
            if phone_detections:
                print(f"  Found: {[d['text'] for d in phone_detections]}")
            print(f"  Anonymized: {result['anonymized_text']}")
            
            # At least some formats should be detected
            if i < 3:  # First 3 should definitely be detected
                assert len(phone_detections) > 0, \
                    f"Should detect phone in format: {test_text}"

    def test_mixed_case_sensitivity(self):
        """Test PII detection with mixed case variations."""
        test_text = "Hi, I'm david SMITH and my email is DAVID.smith@EXAMPLE.com"
        
        result = self.framework.process_transcript(test_text)
        
        # Verify case-insensitive detection
        person_detections = [d for d in result['pii_detections'] if d['entity_type'] == 'PERSON']
        email_detections = [d for d in result['pii_detections'] if d['entity_type'] == 'EMAIL_ADDRESS']
        
        # Should detect names regardless of case
        assert len(person_detections) > 0, "Should detect person names regardless of case"
        assert len(email_detections) > 0, "Should detect email regardless of case"
        
        print(f"Case sensitivity test:")
        print(f"  Original: {test_text}")
        print(f"  Anonymized: {result['anonymized_text']}")
        print(f"  Person detections: {[d['text'] for d in person_detections]}")
        print(f"  Email detections: {[d['text'] for d in email_detections]}")

    def test_overlapping_pii_detection(self):
        """Test detection when PII elements overlap or are adjacent."""
        test_text = "Contact John.Smith@company.com or john.smith@personal.com immediately."
        
        result = self.framework.process_transcript(test_text)
        
        # Should detect both person name and emails
        person_detections = [d for d in result['pii_detections'] if d['entity_type'] == 'PERSON']
        email_detections = [d for d in result['pii_detections'] if d['entity_type'] == 'EMAIL_ADDRESS']
        
        assert len(email_detections) >= 1, "Should detect at least one email"
        # Person name might be tricky since it's part of email
        
        print(f"Overlapping PII test:")
        print(f"  Original: {test_text}")
        print(f"  Anonymized: {result['anonymized_text']}")
        print(f"  Person detections: {[d['text'] for d in person_detections]}")
        print(f"  Email detections: {[d['text'] for d in email_detections]}")
        print(f"  Total detections: {result['pii_count']}")

    def test_multiple_same_type_pii(self):
        """Test detection of multiple instances of the same PII type."""
        test_text = "John called, then Mary called, then David Smith also called about the same issue."
        
        result = self.framework.process_transcript(test_text)
        
        # Should detect multiple person names
        person_detections = [d for d in result['pii_detections'] if d['entity_type'] == 'PERSON']
        
        # Should detect at least 2 person names
        assert len(person_detections) >= 2, \
            f"Should detect multiple person names, found: {len(person_detections)}"
        
        detected_names = [d['text'] for d in person_detections]
        
        print(f"Multiple PII test:")
        print(f"  Original: {test_text}")
        print(f"  Anonymized: {result['anonymized_text']}")
        print(f"  Detected names: {detected_names}")
        print(f"  Count: {len(person_detections)}")

    def test_partial_pii_in_context(self):
        """Test PII detection in realistic conversational context."""
        test_text = """
        Hello, this is Sarah from customer service. 
        I'm calling about member 12345678, Mr. Johnson.
        We tried to reach you at 0412-345-678 but the line was busy.
        Could you please confirm your email address? Is it still johnson@email.com?
        """
        
        result = self.framework.process_transcript(test_text)
        
        # Should detect various PII types in context
        entity_types = set(d['entity_type'] for d in result['pii_detections'])
        
        print(f"Contextual PII test:")
        print(f"  Original: {test_text.strip()}")
        print(f"  Anonymized: {result['anonymized_text']}")
        print(f"  Entity types detected: {entity_types}")
        print(f"  Total detections: {result['pii_count']}")
        print(f"  Custom detections: {result['custom_detections']}")
        
        # Should have reasonable detection rate
        assert result['pii_count'] >= 2, "Should detect multiple PII items in context"
        assert result['pii_count'] <= 8, "Should not over-detect in context"

    def test_credit_card_detection(self):
        """Test credit card number detection and anonymization."""
        test_text = "My credit card number is 4532-1234-5678-9012 and the CVV is 123."
        
        result = self.framework.process_transcript(test_text)
        
        # Look for credit card detection
        cc_detections = [d for d in result['pii_detections'] if d['entity_type'] == 'CREDIT_CARD']
        
        if len(cc_detections) > 0:
            # If detected, verify anonymization
            assert '4532-1234-5678-9012' not in result['anonymized_text'], \
                "Credit card number should be anonymized"
            assert 'CREDIT_CARD' in result['anonymized_text'], \
                "Should contain credit card placeholder"
        
        print(f"Credit card test:")
        print(f"  Original: {test_text}")
        print(f"  Anonymized: {result['anonymized_text']}")
        print(f"  CC detections: {len(cc_detections)}")
        if cc_detections:
            print(f"  Detected: {[d['text'] for d in cc_detections]}")

    def test_performance_timing_validation(self):
        """Test that performance timing is reasonable across different text lengths."""
        test_cases = [
            "Short text with John Smith",
            "Medium length text with multiple PII items: John Smith, john@example.com, 0412345678, member 12345678",
            "Long text with extensive context: " + "This is a customer service call transcript. " * 50 + 
            "The customer John Smith called about his account. His email is john@example.com and phone is 0412345678. " +
            "Reference number is 98765432. " * 10
        ]
        
        for i, test_text in enumerate(test_cases):
            result = self.framework.process_transcript(test_text)
            
            # Verify timing is reasonable
            processing_time = result['processing_time']
            text_length = len(test_text)
            
            print(f"Timing test {i+1} (length: {text_length}):")
            print(f"  Processing time: {processing_time:.4f} seconds")
            print(f"  PII count: {result['pii_count']}")

            
            # Reasonable timing constraints
            # assert processing_time > 0, "Processing time should be positive"
            assert processing_time < 10, f"Processing should be under 10 seconds, got {processing_time}"
            
            # Performance should scale reasonably with text length
            chars_per_second = text_length / (processing_time + 0.00000000001)
            assert chars_per_second > 100, f"Should process at least 100 chars/sec, got {chars_per_second}"
