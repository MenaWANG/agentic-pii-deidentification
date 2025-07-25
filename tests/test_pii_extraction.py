"""
Tests for PII extraction functionality.
"""
import pytest
from evaluation.metrics import PIIEvaluator, PIIMatch


@pytest.mark.unit
@pytest.mark.evaluation
class TestPIIExtraction:
    
    @pytest.fixture(autouse=True)
    def setup_method(self, sample_transcript, expected_results, sample_ground_truth):
        """Set up test fixtures."""
        self.evaluator = PIIEvaluator(matching_mode='business')
        self.transcript = sample_transcript
        self.expected = expected_results
        self.ground_truth = sample_ground_truth
    
    def test_pii_position_extraction(self):
        """Test that PII positions are correctly extracted from transcript."""
        # Test member first name (should find 2 occurrences)
        positions = self.evaluator._find_pii_positions('Ella', self.transcript, 'member_first_name')
        assert len(positions) == 2, "Should find 2 occurrences of 'Ella'"
        
        # Test member full name (should find 1 occurrence)
        positions = self.evaluator._find_pii_positions('Ella Michael Wilson', self.transcript, 'member_full_name')
        assert len(positions) == 1, "Should find 1 occurrence of full name"
        
        # Test email (should find 1 occurrence)
        positions = self.evaluator._find_pii_positions('ella.wilson@example.com', self.transcript, 'member_email')
        assert len(positions) == 1, "Should find 1 occurrence of email"
    
    def test_word_boundary_exclusion(self):
        """Test that names within emails are properly excluded."""
        # Test that 'ella' and 'wilson' within email are NOT found as separate names
        positions = self.evaluator._find_pii_positions('ella', self.transcript, 'member_first_name')
        
        # Should find standalone 'Ella' occurrences but not the one in email
        for start, end in positions:
            text_around = self.transcript[max(0, start-10):end+10]
            assert '@' not in text_around, f"Found 'ella' too close to email context: {text_around}"
    
    def test_total_pii_count(self):
        """Test that total PII count matches expected."""
        # Create a proper mock series that supports both attribute and dict access
        class MockSeries:
            def __init__(self, data):
                self._data = data
                for key, value in data.items():
                    setattr(self, key, value)
            
            def __getitem__(self, key):
                return self._data[key]
            
            def __contains__(self, key):
                return key in self._data
            
            def get(self, key, default=None):
                return self._data.get(key, default)
        
        gt_series = MockSeries(self.ground_truth)
        
        pii_items = self.evaluator._extract_ground_truth_pii(gt_series, self.transcript)
        
        assert len(pii_items) == self.expected['test_001']['expected_pii_count'], \
            f"Expected {self.expected['test_001']['expected_pii_count']} PII items, got {len(pii_items)}"
    
    def test_smart_matching_logic(self):
        """Test that smart matching allows one detection to match multiple overlapping GT items."""
        # Simulate a detection that covers full name
        detected_pii = [{
            'text': 'Ella Michael Wilson',
            'entity_type': 'PERSON',
            'start': 427,
            'end': 446,
            'score': 0.95
        }]
        
        # Ground truth has overlapping items: first name + full name
        ground_truth_pii = [
            {'value': 'Ella', 'type': 'member_first_name', 'start': 427, 'end': 431},
            {'value': 'Ella Michael Wilson', 'type': 'member_full_name', 'start': 427, 'end': 446}
        ]
        
        matches = self.evaluator._match_detections_to_ground_truth(
            detected_pii, ground_truth_pii, self.transcript
        )
        
        # Should have 2 matches (one detection matching both GT items)
        successful_matches = [m for m in matches if m.match_type in ['exact', 'partial']]
        assert len(successful_matches) == 2, \
            "Smart matching should allow one detection to match multiple overlapping GT items"
    
    def test_pii_protection_rate_calculation(self):
        """Test character-based PII protection rate calculation."""
        # Create detected PII entries
        detected_pii = [
            {
                'text': 'Ella',
                'entity_type': 'PERSON',
                'start': 427,
                'end': 431,
                'score': 0.95
            },
            {
                'text': 'Wilson',
                'entity_type': 'PERSON',
                'start': 500,
                'end': 505,
                'score': 0.90
            }
        ]
        
        # Ground truth PII covering 10 total characters
        ground_truth_pii = [
            {'value': 'Ella', 'type': 'member_first_name', 'start': 427, 'end': 431},  # 4 chars
            {'value': 'Wilson', 'type': 'member_last_name', 'start': 500, 'end': 506}   # 6 chars
        ]
        
        protection_rate = self.evaluator._calculate_pii_protection_rate(detected_pii, ground_truth_pii)
        
        # Should protect 9 out of 10 characters = 0.9
        expected_rate = 9 / 10  # 4 + 5 protected out of 4 + 6 total
        assert abs(protection_rate - expected_rate) < 0.01, \
            "PII protection rate should be calculated correctly"
    
    def test_recall_capping(self):
        """Test that recall is capped at 100% even with over-detection."""
        # Simulate over-detection scenario
        detected_pii = [
            {'text': 'Ella', 'entity_type': 'PERSON', 'start': 427, 'end': 431, 'score': 0.9},
            {'text': 'Michael', 'entity_type': 'PERSON', 'start': 432, 'end': 439, 'score': 0.9},
            {'text': 'Wilson', 'entity_type': 'PERSON', 'start': 440, 'end': 446, 'score': 0.9}
        ]
        
        # Only one ground truth item
        ground_truth_pii = [
            {'value': 'Ella Michael Wilson', 'type': 'member_full_name', 'start': 427, 'end': 446}
        ]
        
        matches = self.evaluator._match_detections_to_ground_truth(
            detected_pii, ground_truth_pii, self.transcript
        )
        
        metrics = self.evaluator._calculate_transcript_metrics(
            matches, ground_truth_pii, detected_pii
        )
        
        # Recall should be capped at 1.0 even if raw recall > 1.0
        assert metrics['recall'] <= 1.0, "Recall should be capped at 100%"
        assert 'raw_recall' in metrics, "Should include uncapped raw recall for debugging" 
    
    def test_recall_calculations_partial_match(self):
        """Test character-based PII protection rate calculation."""
        # Only first name detected
        detected_pii = [
            {'text': 'Amelia', 'entity_type': 'PERSON', 'start': 0, 'end': 6, 'score': 0.9}
        ]
        
        # Both first name and full name are demanded as ground truth
        ground_truth_pii = [
            {'value': 'Amelia Dai', 'type': 'member_full_name', 'start': 0, 'end': 10},
            {'value': 'Amelia', 'type': 'member_first_name', 'start': 0, 'end': 6},
        ]
        
        transcript ="""Amelia Dai loves playing the violin."""
        
        matches = self.evaluator._match_detections_to_ground_truth(
            detected_pii, ground_truth_pii, transcript
        )
        
        metrics = self.evaluator._calculate_transcript_metrics(
            matches, ground_truth_pii, detected_pii
        )
        
        # one match + partial_match
        expected_recall = (1 + 0.6) / 2
        assert abs(metrics['recall'] - expected_recall) < 0.01
        assert abs(metrics['raw_recall'] -  expected_recall) < 0.01

    
    def test_recall_calculations_full_match(self):
        """Test character-based PII protection rate calculation."""
        # Full name detected
        detected_pii = [
            {'text': 'Amelia Dai', 'entity_type': 'PERSON', 'start': 0, 'end': 10, 'score': 0.99}
        ]        
        # Both first name and full name are extracted as ground truth
        ground_truth_pii = [
            {'value': 'Amelia Dai', 'type': 'member_full_name', 'start': 0, 'end': 10},
            {'value': 'Amelia', 'type': 'member_first_name', 'start': 0, 'end': 6},
        ]        
        transcript ="""Amelia Dai loves playing the violin."""
        matches = self.evaluator._match_detections_to_ground_truth(
            detected_pii, ground_truth_pii, transcript
        )        
        metrics = self.evaluator._calculate_transcript_metrics(
            matches, ground_truth_pii, detected_pii
        )        
        # one match + partial_match
        expected_raw_recall = (1 + 1) / 2
        expected_recall = (1 + 1) / 2
        assert abs(metrics['recall'] - expected_recall) < 0.01
        assert abs(metrics['raw_recall'] -  expected_raw_recall) < 0.01
    
    def test_pii_protection_rate_calculation_duplicated_positions(self):
        """Test character-based PII protection rate calculation using detected spans."""
        
        # Detected PII covers only the first 6 characters
        detected_pii = [
            {'text': 'Amelia', 'entity_type': 'PERSON', 'start': 0, 'end': 6}
        ]
        
        # Ground truth includes overlapping spans: 0–6 and 0–10
        ground_truth_pii = [
            {'start': 0, 'end': 6},   # First name
            {'start': 0, 'end': 10}   # Full name
        ]
        
        # Instantiate evaluator and run protection rate calculation
        protection_rate = self.evaluator._calculate_pii_protection_rate(detected_pii, ground_truth_pii)
        
        # Total unique ground truth characters = positions 0–9 (i.e., 10 chars)
        # Detected coverage = positions 0–5 (i.e., 6 chars)
        expected_rate = 6 / 10
        
        assert abs(protection_rate - expected_rate) < 0.01, \
            f"Expected protection rate {expected_rate:.2f}, got {protection_rate:.2f}"
