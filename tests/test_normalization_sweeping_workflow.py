"""
Tests for the integration of normalization, sweeping and Presidio workflow.
"""

import pytest
import sys
from pathlib import Path
import pandas as pd
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.text_normaliser import TextNormaliser
from src.utils.text_sweeper import TextSweeper
from src.workflows.normalization_presidio_workflow import process_transcript_with_normalization
from src.baseline.presidio_framework import PurePresidioFramework


@pytest.mark.integration
class TestNormalizationSweepingWorkflow:
    """Test the integrated workflow with sweeping."""

    @pytest.fixture
    def mock_presidio_framework(self):
        """Create a mock presidio framework."""
        mock_framework = Mock(spec=PurePresidioFramework)
        # Mock the process_transcript method to return a simple result
        mock_framework.process_transcript.return_value = {
            'anonymized_text': '[PROCESSED TEXT]',
            'pii_detections': [{'entity_type': 'TEST', 'text': 'test'}],
            'pii_count': 1,
            'custom_detections': {}
        }
        return mock_framework

    def test_workflow_without_sweeping(self, mock_presidio_framework):
        """Test workflow without sweeping layer."""
        test_text = "My name is John Smith, born on January 21st."
        
        result = process_transcript_with_normalization(
            text=test_text,
            normalizer=TextNormaliser(),
            presidio_framework=mock_presidio_framework,
            apply_sweeping=False
        )
        
        # Verify workflow stages
        assert 'original_text' in result
        assert 'normalized_text' in result
        assert 'anonymized_text' in result
        assert 'swept_text' not in result  # Should not exist when sweeping is disabled
        assert result['workflow'] == 'normalization_presidio'
        assert result['sweeping_applied'] is False
        
        # Verify mock interaction
        mock_presidio_framework.process_transcript.assert_called_once()
        # The input to Presidio should be the normalized text
        args, _ = mock_presidio_framework.process_transcript.call_args
        normalized_text = args[0]
        # The normalized text should still contain "January" and "21st"
        assert "January" in normalized_text
        assert "21st" in normalized_text

    def test_workflow_with_sweeping(self, mock_presidio_framework):
        """Test workflow with sweeping layer enabled."""
        test_text = "My name is John Smith, born on January 21st."
        
        presidio_framework = PurePresidioFramework(enable_mlflow=False)
        
        result = process_transcript_with_normalization(
            text=test_text,
            normalizer=TextNormaliser(),
            presidio_framework=presidio_framework,
            sweeper=TextSweeper(),
            apply_sweeping=True,
            sweep_months=True,
            sweep_ordinals=True,
            custom_sweeping_dict={'<PERSON>': ['John Smith', 'John', 'Smith']}
        )
        
        # Verify workflow stages
        assert 'original_text' in result
        assert 'normalized_text' in result
        assert 'anonymized_text' in result
        assert 'swept_text' in result  # Should exist when sweeping is enabled
        assert result['workflow'] == 'normalization_presidio_sweeping'
        assert result['sweeping_applied'] is True
        assert 'sweeping_time' in result
        
        # Verify sweeping was applied correctly
        swept_text = result['swept_text']
        assert "January" not in swept_text
        assert "21st" not in swept_text
        assert "John Smith" not in swept_text
        assert "<MONTH>" in swept_text
        assert "<GENERIC_NUMBER>" in swept_text
        assert "<PERSON>" in swept_text
        


    def test_workflow_with_partial_sweeping(self, mock_presidio_framework):
        """Test workflow with only some sweeping features enabled."""
        test_text = "My name is John Smith, born on January 21st."
        
        presidio_framework = PurePresidioFramework(enable_mlflow=False)
        
        # Only sweep months, not ordinals or custom entities
        result = process_transcript_with_normalization(
            text=test_text,
            normalizer=TextNormaliser(),
            presidio_framework=presidio_framework,
            sweeper=TextSweeper(),
            apply_sweeping=True,
            sweep_months=True,
            sweep_ordinals=False,
            custom_sweeping_dict=None
        )
        
        # Verify workflow stages
        assert result['workflow'] == 'normalization_presidio_sweeping'
        assert result['sweeping_applied'] is True
        
        # Verify selective sweeping was applied correctly
        swept_text = result['swept_text']
        anonymized_text = result['anonymized_text']
        assert "January" not in swept_text
        assert "<MONTH>" in swept_text
        assert "21st" in swept_text  # Ordinals not swept
        assert "John Smith" not in swept_text  
        assert "John Smith" not in anonymized_text

    def test_performance_metrics(self, mock_presidio_framework):
        """Test that performance metrics are correctly tracked."""
        test_text = "My name is John Smith, born on January 21st."
        
        result = process_transcript_with_normalization(
            text=test_text,
            normalizer=TextNormaliser(),
            presidio_framework=mock_presidio_framework,
            sweeper=TextSweeper(),
            apply_sweeping=True
        )
        
        # Verify timing metrics
        assert 'processing_time' in result
        assert 'normalization_time' in result
        assert 'sweeping_time' in result
        assert 'presidio_time' in result
        
        # Verify timing values make sense
        assert result['processing_time'] >= 0
        assert result['normalization_time'] >= 0
        assert result['sweeping_time'] >= 0
        assert result['presidio_time'] >= 0
        
        # Total time should be greater than or equal to the sum of individual times
        # (allowing for small floating point differences)
        component_time_sum = result['normalization_time'] + result['sweeping_time'] + result['presidio_time']
        assert result['processing_time'] >= component_time_sum - 0.001  # Small epsilon for floating point comparison


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test the end-to-end workflow with real data."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        data = {
            'call_id': [1, 2],
            'call_transcript': [
                "My name is John Smith and I was born on January 21st, 1990. My phone is 0412345678.",
                "Jane Doe called on February 3rd about her account number 12345678. Her email is jane.doe@email.com"
            ]
        }
        return pd.DataFrame(data)

    def test_process_dataset_with_sweeping(self, sample_dataframe):
        """Test processing a dataset with sweeping enabled."""
        from src.workflows.normalization_presidio_workflow import process_dataset_with_normalization
        result_df = process_dataset_with_normalization(
            raw_df=sample_dataframe,
            apply_sweeping=True,
            sweep_months=True,
            sweep_ordinals=True
        )
        
        # Verify the results structure
        assert len(result_df) == 2
        assert 'original_transcript' in result_df.columns
        assert 'normalized_transcript' in result_df.columns
        assert 'presidio_transcript' in result_df.columns
        assert 'swept_transcript' in result_df.columns
        assert 'anonymized_transcript' in result_df.columns
        
        # Check first row for proper PII handling
        first_presidio = result_df.iloc[0]['presidio_transcript']
        assert "John Smith" not in first_presidio
        assert "<PERSON>" in first_presidio
        assert "0412345678" not in first_presidio

        
        first_swept = result_df.iloc[0]['swept_transcript']
        assert "<MONTH>" in first_swept
        assert "<GENERIC_NUMBER>" in first_swept
        assert "January" not in first_swept
        assert "21st" not in first_swept
        assert "<PERSON>" in first_swept  # Presidio replacements preserved

        
        # Check second row for proper PII handling
        second_presidio = result_df.iloc[1]['presidio_transcript']
        assert "Jane Doe" not in second_presidio
        assert "<PERSON>" in second_presidio
        assert "jane.doe@email.com" not in second_presidio
        assert "<EMAIL_ADDRESS>" in second_presidio
        assert "12345678" not in second_presidio
        
        second_swept = result_df.iloc[1]['swept_transcript']
        assert "<MONTH>" in second_swept
        assert "<GENERIC_NUMBER>" in second_swept
        assert "February" not in second_swept
        assert "3rd" not in second_swept
        assert "<PERSON>" in second_swept  # Presidio replacements preserved
        assert "<EMAIL_ADDRESS>" in second_swept
