# This file will contain the evaluation logic and metric calculations
# to compare the performance of the two frameworks. 

"""
Evaluation Framework for PII Deidentification
Handles comparison between detected PII and ground truth data with focus on 100% recall target.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
import re
from dataclasses import dataclass
from collections import defaultdict
import json


@dataclass
class PIIMatch:
    """Represents a match between detected PII and ground truth."""
    ground_truth_value: str
    ground_truth_type: str
    detected_value: str
    detected_type: str
    match_type: str  # 'exact', 'partial', 'over_detection', 'missed'
    confidence: float
    start_pos: int
    end_pos: int


class PIIEvaluator:
    """
    Comprehensive PII evaluation framework focusing on 100% recall target.
    Maps Presidio detections to ground truth PII with sophisticated matching logic.
    """
    
    def __init__(self, matching_mode: str = 'business'):
        """
        Initialize PII Evaluator with configurable matching strategy.
        
        Args:
            matching_mode (str): 
                - 'business' (default): Any PII detection covering ground truth = success 
                  Focus: "Is the PII actually protected?" (production deployment)
                - 'research': Requires exact entity type matching for success
                  Focus: "Is the PII classified correctly?" (academic evaluation)
        """
        # Validate matching mode
        if matching_mode not in ['business', 'research']:
            raise ValueError("matching_mode must be 'business' or 'research'")
            
        self.matching_mode = matching_mode
        
        # Entity type mappings
        self.entity_mappings = {
            'PERSON': ['member_first_name', 'member_middle_name', 'member_last_name', 'consultant_first_name'],
            'EMAIL_ADDRESS': ['member_email'],
            'PHONE_NUMBER': ['member_mobile'],
            'LOCATION': ['member_address'],
            'DATE_TIME': ['member_number'],  # Often detected as dates
            'US_SSN': ['member_number']      # Sometimes detected as SSN
        }
        
        print(f"🔧 PIIEvaluator initialized with '{matching_mode}' matching mode")
        if matching_mode == 'business':
            print("   ✅ Business Focus: Any PII detection over ground truth = SUCCESS")
        else:
            print("   📊 Research Focus: Exact entity type matching required")
        
        # Phone number patterns for Australian numbers
        self.phone_patterns = [
            r'\b0[2-9]\d{4}\s?\d{3}\s?\d{3}\b',  # 0X XXXX XXXX (landline)
            r'\b04\d{4}\s?\d{3}\s?\d{3}\b',      # 04XX XXX XXX (mobile)
            r'\b\+61\s?[2-9]\s?\d{4}\s?\d{4}\b'  # +61 X XXXX XXXX (international)
            r'\b04\d{4}\s?\d{3}\s?\d{3}\b',      # 04XXXX XXX XXX (synthetic data)
        ]
    
    def evaluate_framework_results(self, results_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> Dict:
        """
        Evaluate framework results against ground truth data.
        
        Args:
            results_df: DataFrame with processing results (anonymized transcripts + detections)
            ground_truth_df: Original DataFrame with ground truth PII columns
            
        Returns:
            Comprehensive evaluation metrics dictionary
        """
        print("🔍 Starting PII evaluation...")
        
        # Initialize metrics
        evaluation_results = {
            'overall_metrics': {},
            'per_transcript_metrics': [],
            'per_entity_type_metrics': {},
            'detailed_analysis': {
                'missed_pii': [],
                'over_detections': [],
                'partial_matches': []
            }
        }
        
        # Process each transcript
        for idx, result_row in results_df.iterrows():
            ground_truth_row = ground_truth_df.loc[ground_truth_df['call_id'] == result_row['call_id']].iloc[0]
            
            transcript_metrics = self._evaluate_single_transcript(
                result_row, ground_truth_row
            )
            
            evaluation_results['per_transcript_metrics'].append(transcript_metrics)
        
        # Calculate overall metrics
        evaluation_results['overall_metrics'] = self._calculate_overall_metrics(
            evaluation_results['per_transcript_metrics']
        )
        
        # Calculate per-entity-type metrics
        evaluation_results['per_entity_type_metrics'] = self._calculate_per_entity_metrics(
            evaluation_results['per_transcript_metrics']
        )
        
        # Generate detailed analysis
        evaluation_results['detailed_analysis'] = self._generate_detailed_analysis(
            evaluation_results['per_transcript_metrics']
        )
        
        return evaluation_results
    
    def _evaluate_single_transcript(self, result_row: pd.Series, ground_truth_row: pd.Series) -> Dict:
        """Evaluate a single transcript against its ground truth."""
        transcript_id = result_row['call_id']
        original_text = result_row['original_transcript']
        detected_pii = result_row['pii_detections']
        
        # Extract ground truth PII
        ground_truth_pii = self._extract_ground_truth_pii(ground_truth_row, original_text)
        
        # Match detections to ground truth
        matches = self._match_detections_to_ground_truth(detected_pii, ground_truth_pii, original_text)
        
        # Calculate metrics for this transcript
        metrics = self._calculate_transcript_metrics(matches, ground_truth_pii, detected_pii)
        metrics['transcript_id'] = transcript_id
        
        return metrics
    
    def _extract_ground_truth_pii(self, ground_truth_row: pd.Series, transcript_text: str) -> List[Dict]:
        """Extract ground truth PII with positions in transcript - COUNT ALL OCCURRENCES."""
        ground_truth_pii = []
        
        # Define ground truth fields (customer PII)
        gt_fields = {
            'member_first_name': ground_truth_row['member_first_name'],
            'member_middle_name': ground_truth_row.get('member_middle_name'),  # Optional field
            'member_last_name': ground_truth_row['member_last_name'],
            'member_email': ground_truth_row['member_email'],
            'member_mobile': ground_truth_row['member_mobile'],
            'member_address': ground_truth_row['member_address'],
            'member_number': str(ground_truth_row['member_number'])
        }
        
        # IMPORTANT: Also include consultant names as valid PII to protect
        # In production, any person name should be protected regardless of role
        if 'consultant_first_name' in ground_truth_row:
            gt_fields['consultant_first_name'] = ground_truth_row['consultant_first_name']
        
        # CORRECTED: Count ALL occurrences of each PII item in transcript
        for pii_type, pii_value in gt_fields.items():
            if pd.notna(pii_value) and str(pii_value).strip():
                pii_value = str(pii_value).strip()
                
                # Find ALL positions where this PII appears in transcript
                positions = self._find_pii_positions(pii_value, transcript_text, pii_type)
                
                # Create separate ground truth entry for EACH occurrence
                for start_pos, end_pos in positions:
                    ground_truth_pii.append({
                        'type': pii_type,
                        'value': pii_value,
                        'start': start_pos,
                        'end': end_pos,
                        'occurrence_id': f"{pii_type}_{start_pos}_{end_pos}"  # Unique ID for each occurrence
                    })
        
        # Log for debugging
        if len(ground_truth_pii) > 0:
            print(f"  📍 Found {len(ground_truth_pii)} total PII occurrences in transcript")
            pii_counts = {}
            for item in ground_truth_pii:
                pii_counts[item['type']] = pii_counts.get(item['type'], 0) + 1
            for pii_type, count in pii_counts.items():
                print(f"    - {pii_type}: {count} occurrences")
        
        return ground_truth_pii
    
    def _find_pii_positions(self, pii_value: str, transcript: str, pii_type: str) -> List[Tuple[int, int]]:
        """Find all positions of PII value in transcript with fuzzy matching."""
        positions = []
        
        # Exact match first
        start = 0
        while True:
            pos = transcript.lower().find(pii_value.lower(), start)
            if pos == -1:
                break
            positions.append((pos, pos + len(pii_value)))
            start = pos + 1
        
        # For phone numbers, try different formats
        if pii_type == 'member_mobile':
            # Try with different spacing/formatting
            clean_phone = re.sub(r'[^\d]', '', pii_value)
            for pattern in self.phone_patterns:
                for match in re.finditer(pattern, transcript):
                    if re.sub(r'[^\d]', '', match.group()) == clean_phone:
                        positions.append((match.start(), match.end()))
        
        # For individual name components, ensure word boundaries
        elif pii_type in ['member_first_name', 'member_middle_name', 'member_last_name', 'consultant_first_name']:
            if len(pii_value) > 1:  # Skip very short names to avoid false positives
                start = 0
                while True:
                    pos = transcript.lower().find(pii_value.lower(), start)
                    if pos == -1:
                        break
                    # Check word boundaries for name components
                    if self._is_word_boundary(transcript, pos, len(pii_value)):
                        positions.append((pos, pos + len(pii_value)))
                    start = pos + 1
        
        return list(set(positions))  # Remove duplicates
    
    def _is_word_boundary(self, text: str, start_pos: int, length: int) -> bool:
        """Check if the found text is at word boundaries to avoid partial word matches."""
        end_pos = start_pos + length
        
        # Check start boundary
        start_ok = (start_pos == 0 or not text[start_pos - 1].isalnum())
        
        # Check end boundary  
        end_ok = (end_pos >= len(text) or not text[end_pos].isalnum())
        
        return start_ok and end_ok
    
    def _match_detections_to_ground_truth(self, detected_pii: List[Dict], 
                                         ground_truth_pii: List[Dict], 
                                         transcript: str) -> List[PIIMatch]:
        """Match detected PII to ground truth with configurable matching strategy."""
        matches = []
        gt_matched = set()
        det_matched = set()
        
        # Match detected items to ground truth
        for j, det_item in enumerate(detected_pii):
            best_match = None
            best_score = 0
            
            for i, gt_item in enumerate(ground_truth_pii):
                if i in gt_matched:
                    continue
                    
                # Calculate overlap between detected and ground truth positions
                det_start, det_end = det_item['start'], det_item['end']
                gt_start, gt_end = gt_item['start'], gt_item['end']
                
                # Check for positional overlap
                overlap = not (det_end <= gt_start or det_start >= gt_end)
                if not overlap:
                    continue
                
                # CONFIGURABLE MATCHING LOGIC
                if self.matching_mode == 'business':
                    # BUSINESS: Any PII detection over ground truth = SUCCESS
                    # Business focus: "Is the PII protected?"
                    overlap_size = min(det_end, gt_end) - max(det_start, gt_start)
                    gt_size = gt_end - gt_start
                    overlap_ratio = overlap_size / gt_size if gt_size > 0 else 0
                    
                    if overlap_ratio > 0.1:  # Even small overlap counts as protection
                        match_score = overlap_ratio
                        match_type = 'exact' if overlap_ratio > 0.8 else 'partial'
                        
                        if match_score > best_score:
                            best_match = (i, match_type, match_score)
                            best_score = match_score
                            
                elif self.matching_mode == 'research':
                    # RESEARCH: Require exact entity type matching + good positional overlap
                    # Research focus: "Is the PII classified correctly?"
                    if self._is_compatible_entity_type(det_item['entity_type'], gt_item['type']):
                        overlap_size = min(det_end, gt_end) - max(det_start, gt_start)
                        gt_size = gt_end - gt_start
                        overlap_ratio = overlap_size / gt_size if gt_size > 0 else 0
                        
                        if overlap_ratio > 0.5:  # Require substantial overlap in research mode
                            match_score = overlap_ratio
                            match_type = 'exact' if overlap_ratio > 0.8 else 'partial'
                            
                            if match_score > best_score:
                                best_match = (i, match_type, match_score)
                                best_score = match_score
            
            # Record the best match if found
            if best_match:
                gt_idx, match_type, confidence = best_match
                gt_item = ground_truth_pii[gt_idx]
                
                matches.append(PIIMatch(
                    ground_truth_value=gt_item['value'],
                    ground_truth_type=gt_item['type'],
                    detected_value=det_item['text'],
                    detected_type=det_item['entity_type'],
                    match_type=match_type,
                    confidence=confidence,
                    start_pos=gt_item['start'],
                    end_pos=gt_item['end']
                ))
                
                gt_matched.add(gt_idx)
                det_matched.add(j)
        
        # Handle unmatched ground truth (missed detections)
        for i, gt_item in enumerate(ground_truth_pii):
            if i not in gt_matched:
                matches.append(PIIMatch(
                    ground_truth_value=gt_item['value'],
                    ground_truth_type=gt_item['type'],
                    detected_value='',
                    detected_type='',
                    match_type='missed',
                    confidence=0.0,
                    start_pos=gt_item['start'],
                    end_pos=gt_item['end']
                ))
        
        # Handle unmatched detections (over-detections)
        for j, det_item in enumerate(detected_pii):
            if j not in det_matched:
                # ALL unmatched detections are over-detections
                # Only ground truth-based matches should count as correct
                matches.append(PIIMatch(
                    ground_truth_value='',
                    ground_truth_type='',
                    detected_value=det_item['text'],
                    detected_type=det_item['entity_type'],
                    match_type='over_detection',
                    confidence=det_item['score'],
                    start_pos=det_item['start'],
                    end_pos=det_item['end']
                ))
        
        return matches
    
    def _calculate_transcript_metrics(self, matches: List[PIIMatch], 
                                    ground_truth_pii: List[Dict], 
                                    detected_pii: List[Dict]) -> Dict:
        """Calculate precision, recall, F1 for a single transcript."""
        
        # Count matches by type
        exact_matches = sum(1 for m in matches if m.match_type == 'exact')
        partial_matches = sum(1 for m in matches if m.match_type == 'partial')
        missed = sum(1 for m in matches if m.match_type == 'missed')
        over_detections = sum(1 for m in matches if m.match_type == 'over_detection')
        
        # Calculate true positives (only ground truth matches count)
        true_positives = (exact_matches + 
                         sum(m.confidence for m in matches if m.match_type == 'partial'))
        false_positives = over_detections  # All non-ground-truth detections
        false_negatives = missed
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / len(ground_truth_pii) if len(ground_truth_pii) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'exact_matches': exact_matches,
            'partial_matches': partial_matches,
            'missed': missed,
            'over_detections': over_detections,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_ground_truth': len(ground_truth_pii),
            'total_detected': len(detected_pii),
            'matches': matches
        }
    
    def _calculate_overall_metrics(self, per_transcript_metrics: List[Dict]) -> Dict:
        """Calculate overall metrics across all transcripts."""
        
        # Aggregate counts
        total_tp = sum(m['true_positives'] for m in per_transcript_metrics)
        total_fp = sum(m['false_positives'] for m in per_transcript_metrics)
        total_fn = sum(m['false_negatives'] for m in per_transcript_metrics)
        
        # Calculate overall metrics
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        return {
            'total_transcripts': len(per_transcript_metrics),
            'total_true_positives': total_tp,
            'total_false_positives': total_fp,
            'total_false_negatives': total_fn,
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_f1_score': overall_f1,
            'avg_precision': np.mean([m['precision'] for m in per_transcript_metrics]),
            'avg_recall': np.mean([m['recall'] for m in per_transcript_metrics]),
            'avg_f1_score': np.mean([m['f1_score'] for m in per_transcript_metrics]),
            'recall_target_achievement': overall_recall >= 0.99,  # 99%+ recall target
        }
    
    def _calculate_per_entity_metrics(self, per_transcript_metrics: List[Dict]) -> Dict:
        """Calculate metrics broken down by entity type."""
        entity_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        for transcript_metrics in per_transcript_metrics:
            for match in transcript_metrics['matches']:
                if match.match_type in ['exact', 'partial']:
                    entity_metrics[match.ground_truth_type]['tp'] += match.confidence
                elif match.match_type == 'missed':
                    entity_metrics[match.ground_truth_type]['fn'] += 1
                elif match.match_type == 'over_detection':
                    entity_metrics[match.detected_type]['fp'] += 1
        
        # Calculate metrics for each entity type
        final_metrics = {}
        for entity_type, counts in entity_metrics.items():
            tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            final_metrics[entity_type] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn
            }
        
        return dict(final_metrics)
    
    def _generate_detailed_analysis(self, per_transcript_metrics: List[Dict]) -> Dict:
        """Generate detailed analysis of evaluation results."""
        missed_pii = []
        over_detections = []
        partial_matches = []
        
        for transcript_metrics in per_transcript_metrics:
            transcript_id = transcript_metrics['transcript_id']
            
            for match in transcript_metrics['matches']:
                if match.match_type == 'missed':
                    missed_pii.append({
                        'transcript_id': transcript_id,
                        'pii_type': match.ground_truth_type,
                        'pii_value': match.ground_truth_value,
                        'position': f"{match.start_pos}-{match.end_pos}"
                    })
                elif match.match_type == 'over_detection':
                    over_detections.append({
                        'transcript_id': transcript_id,
                        'detected_type': match.detected_type,
                        'detected_value': match.detected_value,
                        'confidence': match.confidence,
                        'position': f"{match.start_pos}-{match.end_pos}"
                    })
                elif match.match_type == 'partial':
                    partial_matches.append({
                        'transcript_id': transcript_id,
                        'ground_truth_type': match.ground_truth_type,
                        'ground_truth_value': match.ground_truth_value,
                        'detected_type': match.detected_type,
                        'detected_value': match.detected_value,
                        'confidence': match.confidence,
                        'position': f"{match.start_pos}-{match.end_pos}"
                    })
        
        return {
            'missed_pii': missed_pii,
            'over_detections': over_detections,
            'partial_matches': partial_matches,
            'summary': {
                'total_missed': len(missed_pii),
                'total_over_detections': len(over_detections),
                'total_partial_matches': len(partial_matches)
            }
        }
    
    def print_evaluation_summary(self, results: Dict):
        """Print a comprehensive evaluation summary."""
        overall = results['overall_metrics']
        analysis = results['detailed_analysis']
        
        print("\n" + "="*60)
        print("🎯 PII DEIDENTIFICATION EVALUATION RESULTS")
        print("="*60)
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   Precision: {overall['overall_precision']:.3f}")
        print(f"   Recall:    {overall['overall_recall']:.3f} {'✅' if overall.get('recall_target_achievement', False) else '❌'}")
        print(f"   F1-Score:  {overall['overall_f1_score']:.3f}")
        
        print(f"\n📈 DETAILED COUNTS:")
        print(f"   True Positives:  {overall['total_true_positives']:.1f}")
        print(f"   False Positives: {overall['total_false_positives']}")
        print(f"   False Negatives: {overall['total_false_negatives']}")
        
        print(f"\n🔍 ENTITY TYPE BREAKDOWN:")
        for entity_type, metrics in results['per_entity_type_metrics'].items():
            print(f"   {entity_type:20} | P: {metrics['precision']:.3f} | R: {metrics['recall']:.3f} | F1: {metrics['f1_score']:.3f}")
        
        print(f"\n⚠️  ISSUES IDENTIFIED:")
        print(f"   Missed PII:       {analysis['summary']['total_missed']}")
        print(f"   Over-detections:  {analysis['summary']['total_over_detections']}")
        print(f"   Partial matches:  {analysis['summary']['total_partial_matches']}")
        
        print(f"\n🎯 RECALL TARGET: {'✅ ACHIEVED' if overall.get('recall_target_achievement', False) else '❌ NOT ACHIEVED'}")
        print("="*60)

    def evaluate_single_transcript_public(self, original_text: str, detected_pii: List[Dict], 
                                         ground_truth_pii: Dict, call_id: str = "demo") -> Dict:
        """
        PUBLIC METHOD: Evaluate a single transcript with consistent metric calculation
        For use by demos and monitoring - ensures consistent evaluation logic across all tools
        
        Args:
            original_text: Original transcript text
            detected_pii: List of detected PII from framework
            ground_truth_pii: Dict with ground truth PII values
            call_id: Identifier for this transcript
            
        Returns:
            Dict with comprehensive evaluation metrics including total occurrences
        """
        
        # Create mock series with ground truth data
        ground_truth_series = pd.Series(ground_truth_pii)
        
        # Extract ground truth with ALL occurrences counted
        gt_pii_occurrences = self._extract_ground_truth_pii(ground_truth_series, original_text)
        
        # Match detections to ground truth occurrences
        matches = self._match_detections_to_ground_truth(detected_pii, gt_pii_occurrences, original_text)
        
        # Calculate metrics using actual occurrence counts
        transcript_metrics = self._calculate_transcript_metrics(matches, gt_pii_occurrences, detected_pii)
        
        # Add call ID and additional context
        transcript_metrics['call_id'] = call_id
        transcript_metrics['total_pii_occurrences'] = len(gt_pii_occurrences)  # True denominator for recall
        transcript_metrics['total_unique_pii_types'] = len(set(item['type'] for item in gt_pii_occurrences))
        
        # Breakdown by PII type
        pii_type_breakdown = {}
        for item in gt_pii_occurrences:
            pii_type = item['type']
            if pii_type not in pii_type_breakdown:
                pii_type_breakdown[pii_type] = 0
            pii_type_breakdown[pii_type] += 1
        transcript_metrics['pii_occurrence_breakdown'] = pii_type_breakdown
        
        return transcript_metrics

    def _is_compatible_entity_type(self, detected_type: str, ground_truth_type: str) -> bool:
        """
        Check if detected entity type is compatible with ground truth PII type.
        Used in research matching mode for precise evaluation.
        
        Args:
            detected_type: Entity type from detection (e.g., 'PERSON', 'EMAIL_ADDRESS')
            ground_truth_type: Ground truth PII type (e.g., 'member_full_name')
            
        Returns:
            bool: True if types are compatible
        """
        # Check if ground truth type is in the list for this detected type
        compatible_gt_types = self.entity_mappings.get(detected_type, [])
        return ground_truth_type in compatible_gt_types


def create_evaluation_demo():
    """Create a simple demo of the evaluation framework."""
    print("🧪 PII Evaluation Framework Demo")
    print("\nKey Features:")
    print("✅ Handles exact and partial matches")
    print("✅ Maps Presidio entities to ground truth PII types") 
    print("✅ Focuses on 100% recall target")
    print("✅ Provides detailed analysis of missed/over-detected PII")
    print("✅ Supports Australian phone number formats")
    print("✅ Generates comprehensive metrics")


if __name__ == "__main__":
    create_evaluation_demo() 