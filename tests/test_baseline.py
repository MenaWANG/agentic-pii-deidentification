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

    @pytest.mark.parametrize("entity_type,expected_text", [
        ('PERSON', 'name'),
        ('EMAIL_ADDRESS', 'email'),
        ('PHONE_NUMBER', 'phone'),
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
            
            # Verify all rows were processed
            assert all(results_df['pii_count'] >= 0)  # Should have non-negative PII counts 