"""
Tests for PII extraction functionality.
"""
import json
import pandas as pd
from pathlib import Path
import sys
import os
import unittest

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation.metrics import PIIEvaluator

class TestPIIExtraction(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = PIIEvaluator(matching_mode='business')
        
        # Load test data
        test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        
        with open(os.path.join(test_data_dir, 'simple_transcript.txt'), 'r') as f:
            self.transcript = f.read()
            
        with open(os.path.join(test_data_dir, 'expected_results.json'), 'r') as f:
            self.expected = json.load(f)
            
        # Ground truth PII for the test transcript
        self.ground_truth = {
            'member_first_name': 'Ella',
            'member_full_name': 'Ella Michael Wilson',
            'member_email': 'ella.wilson@example.com',
            'member_phone': '0412 345 678',
            'member_mobile': '0412 345 678',  # Same as phone for our test
            'member_address': '',  # Empty in our test transcript
            'member_number': '',   # Empty in our test transcript
            'agent_first_name': 'Sarah',
            'consultant_first_name': 'Sarah',  # Same as agent for our test
            'case_number': 'CASE-2024-001'
        }
    
    def test_pii_position_extraction(self):
        """Test that PII positions are correctly extracted from transcript."""
        # Test member first name (should find 2 occurrences)
        positions = self.evaluator._find_pii_positions('Ella', self.transcript, 'member_first_name')
        self.assertEqual(len(positions), 2, "Should find 2 occurrences of 'Ella'")
        
        # Test member full name (should find 1 occurrence)
        positions = self.evaluator._find_pii_positions('Ella Michael Wilson', self.transcript, 'member_full_name')
        self.assertEqual(len(positions), 1, "Should find 1 occurrence of full name")
        
        # Test email (should find 1 occurrence)
        positions = self.evaluator._find_pii_positions('ella.wilson@example.com', self.transcript, 'member_email')
        self.assertEqual(len(positions), 1, "Should find 1 occurrence of email")
    
    def test_word_boundary_exclusion(self):
        """Test that names within emails are properly excluded."""
        # Test that 'ella' and 'wilson' within email are NOT found as separate names
        positions = self.evaluator._find_pii_positions('ella', self.transcript, 'member_first_name')
        
        # Should find standalone 'Ella' occurrences but not the one in email
        for start, end in positions:
            text_around = self.transcript[max(0, start-10):end+10]
            self.assertNotIn('@', text_around, f"Found 'ella' too close to email context: {text_around}")
    
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
        
        self.assertEqual(len(pii_items), self.expected['total_pii_count'], 
                        f"Expected {self.expected['total_pii_count']} PII items, got {len(pii_items)}")
    
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
        self.assertEqual(len(successful_matches), 2, 
                        "Smart matching should allow one detection to match multiple overlapping GT items")
    
    def test_pii_protection_rate_calculation(self):
        """Test character-based PII protection rate calculation."""
        # Mock matches covering some PII characters
        matches = [
            type('MockMatch', (), {
                'match_type': 'exact',
                'start_pos': 427,
                'end_pos': 431  # Covers "Ella" (4 chars)
            })(),
            type('MockMatch', (), {
                'match_type': 'partial', 
                'start_pos': 500,
                'end_pos': 505  # Covers 5 more chars
            })()
        ]
        
        # Ground truth PII covering 10 total characters
        ground_truth_pii = [
            {'start': 427, 'end': 431},  # 4 chars
            {'start': 500, 'end': 506}   # 6 chars
        ]
        
        protection_rate = self.evaluator._calculate_pii_protection_rate(matches, ground_truth_pii)
        
        # Should protect 9 out of 10 characters = 0.9
        expected_rate = 9 / 10  # 4 + 5 protected out of 4 + 6 total
        self.assertAlmostEqual(protection_rate, expected_rate, places=2,
                              msg="PII protection rate should be calculated correctly")
    
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
        self.assertLessEqual(metrics['recall'], 1.0, "Recall should be capped at 100%")
        self.assertIn('raw_recall', metrics, "Should include uncapped raw recall for debugging")


def run_tests():
    """Run all PII extraction tests and return success status."""
    try:
        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestPIIExtraction)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Return success status
        return result.wasSuccessful()
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


if __name__ == '__main__':
    unittest.main() 