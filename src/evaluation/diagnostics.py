"""
Diagnostic Analysis Utilities for PII Deidentification
Provides high-level analysis and visualization tools for understanding framework performance.
"""

import pandas as pd
import html
import sys
from typing import Dict, List, Union
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from evaluation.metrics import PIIEvaluator


def get_transcript_cases_by_performance(results_df: pd.DataFrame, 
                                      ground_truth_df: pd.DataFrame, 
                                      transcript_column: str = 'original_transcript',
                                      metric: str = 'recall', 
                                      n_cases: int = 3, 
                                      ascending: bool = True,
                                      matching_mode: str = 'business') -> List[Dict]:
    """
    Get n transcript cases ranked by performance metric.
    
    Args:
        results_df: DataFrame with processing results
        ground_truth_df: DataFrame with ground truth data
        transcript_column: Column name for transcript to be dedacted
        metric: 'recall', 'precision', 'f1_score' - which metric to rank by
        n_cases: Number of cases to return (default 3)
        ascending: True for worst performers, False for best performers
        matching_mode: 'business' or 'research' evaluation mode
        
    Returns:
        List of transcript data dicts ready for diagnostic HTML table
        
    Examples:
        # Worst 5 recall cases (missed PII)
        worst_recall = get_transcript_cases_by_performance(
            results_df, df, metric='recall', n_cases=5, ascending=True
        )
        
        # Best 3 precision cases (accurate detection)
        best_precision = get_transcript_cases_by_performance(
            results_df, df, metric='precision', n_cases=3, ascending=False
        )
        
        # Middle performers for F1 (balanced analysis)
        middle_f1 = get_transcript_cases_by_performance(
            results_df, df, metric='f1_score', n_cases=10, ascending=True
        )[3:7]  # Skip worst 3, take next 4
    """
    
    print(f"🔍 ANALYZING TRANSCRIPT PERFORMANCE BY {metric.upper()}")
    print("=" * 60)
    
    # Initialize evaluator with matching mode
    try:
        evaluator = PIIEvaluator(matching_mode=matching_mode)
    except TypeError:
        evaluator = PIIEvaluator()
        evaluator.matching_mode = matching_mode
    
    # Calculate performance for all transcripts
    transcript_performance = []
    
    for idx, result_row in results_df.iterrows():
        call_id = result_row['call_id']
        
        # Get corresponding ground truth
        try:
            gt_row = ground_truth_df[ground_truth_df['call_id'] == call_id].iloc[0]
        except IndexError:
            print(f"⚠️ No ground truth found for {call_id}, skipping")
            continue
        
        # Prepare ground truth for evaluation (using new structure)
        ground_truth = {
            'member_first_name': gt_row['member_first_name'],
            'member_full_name': gt_row['member_full_name'],  # NEW: Full name field
            'member_email': gt_row['member_email'],
            'member_mobile': gt_row['member_mobile'],
            'member_address': gt_row['member_address'],
            'member_number': str(gt_row['member_number']),
            'consultant_first_name': gt_row['consultant_first_name']
        }
        
        try:
            # Get precise metrics from centralized evaluation
            eval_result = evaluator.evaluate_single_transcript_public(
                original_text=result_row[transcript_column],
                detected_pii=result_row['pii_detections'],
                ground_truth_pii=ground_truth,
                call_id=call_id
            )
            
            performance = {
                'call_id': call_id,
                'recall': eval_result['recall'],
                'precision': eval_result['precision'],
                'f1_score': eval_result['f1_score'],
                'total_pii_occurrences': eval_result['total_pii_occurrences'],
                'detected_count': len(result_row['pii_detections']),
                'exact_matches': eval_result['exact_matches'],
                'eval_result': eval_result  # Store full result for later use
            }
            
        except Exception as e:
            print(f"⚠️ Error evaluating {call_id}: {e}")
            # Fallback to basic calculation
            detected_count = len(result_row['pii_detections'])
            performance = {
                'call_id': call_id,
                'recall': None,
                'precision': None,
                'f1_score': None,
                'total_pii_occurrences': None,
                'detected_count': detected_count,
                'exact_matches': None,
                'eval_result': None
            }
        
        transcript_performance.append(performance)
    
    # Validate metric parameter
    if metric not in ['recall', 'precision', 'f1_score']:
        raise ValueError("metric must be 'recall', 'precision', or 'f1_score'")
    
    # Sort by selected metric
    transcript_performance.sort(key=lambda x: x[metric], reverse=not ascending)
    
    # Select top n_cases
    selected_cases = transcript_performance[:n_cases]
    
    # Print summary
    direction = "WORST" if ascending else "BEST"
    print(f"\n📊 {direction} {n_cases} PERFORMERS BY {metric.upper()}:")
    
    for i, perf in enumerate(selected_cases, 1):
        print(f"  {i}. Call {perf['call_id']}: "
              f"{metric}={perf[metric]:.1%}, "
              f"Recall={perf['recall']:.1%}, "
              f"Precision={perf['precision']:.1%}, "
              f"F1={perf['f1_score']:.1%}")
    
    # Prepare data structure for HTML diagnostic table
    cases_data = []
    
    for perf in selected_cases:
        call_id = perf['call_id']
        
        # Get full data
        result_row = results_df[results_df['call_id'] == call_id].iloc[0]
        gt_row = ground_truth_df[ground_truth_df['call_id'] == call_id].iloc[0]
        
        case_data = {
            'call_id': call_id,
            'original_transcript': result_row['original_transcript'],
            'anonymized_transcript': result_row['anonymized_transcript'],
            'detected_pii': result_row['pii_detections'],
            'member_first_name': gt_row['member_first_name'],
            'member_full_name': gt_row['member_full_name'],  # NEW: Full name field
            'member_email': gt_row['member_email'],
            'member_mobile': gt_row['member_mobile'],
            'member_address': gt_row['member_address'],
            'member_number': str(gt_row['member_number']),
            'consultant_first_name': gt_row['consultant_first_name'],
            # Add performance metrics for reference
            'performance_metrics': perf
        }
        if 'normalized_transcript' in result_row and pd.notna(result_row['normalized_transcript']):
            case_data['normalized_transcript'] = result_row['normalized_transcript']
        
        cases_data.append(case_data)
    
    print(f"\n✅ Prepared {len(cases_data)} cases for analysis")
    return cases_data


def create_diagnostic_html_table_configurable(transcript_data: List[Dict], 
                                            transcript_column: str = 'original_transcript',
                                            title: str = "PII Masking Anaysis", 
                                            description: str = "", 
                                            matching_mode: str = 'business') -> str:
    """
    Create configurable diagnostic HTML table for PII analysis.
    
    Args:
        transcript_data: List of transcript data dictionaries
        transcript_column: Column name for transcript to be dedacted
        title: Title for the analysis table
        description: Description text to display
        matching_mode: 'business' (default) = any PII detection = success
                      'research' = exact type matching required
                      
    Returns:
        HTML string for display in notebooks
        
    Notes:
        - Business-focused (business): Any PII detection covering ground truth = SUCCESS
        - Research-focused (research): Exact entity type matching required for SUCCESS
    """
    
    # Initialize evaluator with matching mode
    evaluator = PIIEvaluator(matching_mode=matching_mode)
    
    # Show mode information
    mode_info = {
        'business': "🏢 <strong>Business Mode:</strong> Any PII detection covering ground truth = SUCCESS",
        'research': "🔬 <strong>Research Mode:</strong> Exact entity type matching required for SUCCESS"
    }
    
    html_content = f"""
    <style>
    .diagnostic-table {{
        border-collapse: collapse;
        width: 100%;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin: 20px 0;
    }}
    
    .mode-info {{
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #2196f3;
    }}
    
    .diagnostic-table th {{
        background-color: #2E86AB;
        color: white;
        padding: 12px 8px;
        text-align: center;
        font-weight: bold;
        border: 1px solid #ddd;
        font-size: 11px;
    }}
    
    .diagnostic-table td {{
        padding: 10px 8px;
        border: 1px solid #ddd;
        vertical-align: top;
        font-size: 10px;
        line-height: 1.4;
    }}
    
    .metrics-col {{ background-color: #f8f9fa; font-weight: 500; }}
    .original-col {{ background-color: #fff8dc; }}
    .cleaned-col {{ background-color: #f0f8ff; }}
    
    .missed-pii {{ background-color: #ffcccc; padding: 2px 4px; border-radius: 3px; }}
    .detected-pii-tp {{ background-color: #ccffcc; padding: 2px 4px; border-radius: 3px; }}
    .detected-pii-fp {{ background-color: #fff3cd; padding: 2px 4px; border-radius: 3px; border: 1px solid #ffc107; }}
    </style>
    
    <h3>{title}</h3>
    <div class="mode-info">{mode_info[matching_mode]}</div>
    <p>{description}</p>
    
    <table class="diagnostic-table">
        <thead>
            <tr>
                <th style="width: 25%;">📊 Metrics & Performance</th>
                <th style="width: 37.5%;">📋 Original Transcript</th>
                <th style="width: 37.5%;">🛡️ Cleaned Transcript</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for data in transcript_data:
        call_id = data['call_id']
        original = data[transcript_column]
        anonymized = data['anonymized_transcript']
        detected_pii = data['detected_pii']
        
        # Get ground truth for centralized evaluation (using new structure)
        ground_truth = {
            'member_first_name': data['member_first_name'],
            'member_full_name': data['member_full_name'],  # NEW: Full name field
            'member_email': data['member_email'],
            'member_mobile': data['member_mobile'],
            'member_address': data['member_address'],
            'member_number': data['member_number'],
            'consultant_first_name': data['consultant_first_name']
        }
        
        # CENTRALIZED EVALUATION with selected matching mode
        eval_result = evaluator.evaluate_single_transcript_public(
            original_text=original,
            detected_pii=detected_pii,
            ground_truth_pii=ground_truth,
            call_id=call_id
        )
        
        # Extract metrics
        total_pii_occurrences = eval_result['total_pii_occurrences']
        exact_matches = eval_result['exact_matches']
        partial_matches = eval_result['partial_matches'] 
        missed_matches = eval_result['missed']  # NEW: Add missed matches
        recall = eval_result['recall']
        precision = eval_result['precision']
        pii_protection_rate = eval_result['pii_protection_rate']  # NEW: PII protection rate
        
        # Create highlighted original transcript (missed PII in red)
        highlighted_original = html.escape(original)
        for match in eval_result['matches']:
            if match.match_type == 'missed':
                missed_text = match.ground_truth_value
                highlighted_original = highlighted_original.replace(
                    html.escape(missed_text), 
                    f'<span class="missed-pii">{html.escape(missed_text)}</span>'
                )
        
        # Create highlighted cleaned transcript - distinguish TP vs FP
        highlighted_cleaned = html.escape(anonymized)
        
        # Create mapping of detected PII to their match types
        detection_to_match_type = {}
        for match in eval_result['matches']:
            if match.detected_value:  # Only for actual detections (not missed)
                detection_to_match_type[match.detected_value] = match.match_type
        
        # Enhanced placeholder list including custom recognizers
        placeholders = ['&lt;PERSON&gt;', '&lt;EMAIL_ADDRESS&gt;', '&lt;PHONE_NUMBER&gt;', 
                       '&lt;LOCATION&gt;', '&lt;DATE_TIME&gt;', '&lt;US_SSN&gt;', '&lt;MEMBER_NUMBER&gt;', '&lt;AU_ADDRESS&gt;']
        
        # Determine if each placeholder represents TP or FP
        for placeholder in placeholders:
            # Find corresponding detections for this placeholder type
            placeholder_entity_type = placeholder.replace('&lt;', '').replace('&gt;', '')
            
            # Check if any detection of this type exists and what its match type is
            is_true_positive = False
            is_false_positive = False
            
            for detection in detected_pii:
                if detection['entity_type'] == placeholder_entity_type:
                    # Find the match type for this specific detection
                    for match in eval_result['matches']:
                        if (match.detected_type == detection['entity_type'] and 
                            match.detected_value == detection['text']):
                            if match.match_type in ['exact', 'partial']:
                                is_true_positive = True
                            elif match.match_type == 'over_detection':
                                is_false_positive = True
                            break
            
            # Apply highlighting based on classification
            if is_true_positive and not is_false_positive:
                # Pure True Positive - green
                highlighted_cleaned = highlighted_cleaned.replace(
                    placeholder,
                    f'<span class="detected-pii-tp">{placeholder}</span>'
                )
            elif is_false_positive and not is_true_positive:
                # Pure False Positive - yellow
                highlighted_cleaned = highlighted_cleaned.replace(
                    placeholder,
                    f'<span class="detected-pii-fp">{placeholder}</span>'
                )
            elif is_true_positive and is_false_positive:
                # Mixed case - default to TP (green) but could be improved
                highlighted_cleaned = highlighted_cleaned.replace(
                    placeholder,
                    f'<span class="detected-pii-tp">{placeholder}</span>'
                )
            else:
                # No matching detection found - default to TP styling
                highlighted_cleaned = highlighted_cleaned.replace(
                    placeholder,
                    f'<span class="detected-pii-tp">{placeholder}</span>'
                )
        
        # Performance status (adjusted for matching mode)
        recall_threshold = 0.90 if matching_mode == 'business' else 0.85  # business mode should achieve higher recall
        
        if recall >= recall_threshold and precision >= 0.8:
            status = "🟢 Excellent Protection"
            status_color = "#28a745"
        elif recall >= (recall_threshold - 0.1):
            status = "🟡 Good Protection"
            status_color = "#ffc107"
        else:
            status = "🔴 Needs Improvement"
            status_color = "#dc3545"
        
        # Build metrics content
        metrics_content = f"""
        <strong>📋 CALL ID:</strong> {call_id}<br/>
        <strong>🎯 Total PII Occurrences:</strong> {total_pii_occurrences}<br/>
        {f'<strong>🔍 PII Detected:</strong> {len(detected_pii)}<br/>' if False else ''}
        {f'<strong>✅ Exact Matches:</strong> {exact_matches}<br/>' if False else ''}
        {f'<strong>⚡ Partial Matches:</strong> {partial_matches}<br/>' if False else ''}        
        {f'<strong>🔍 Missed Matches:</strong> {missed_matches}<br/>' if False else ''}
        <hr style="margin: 8px 0;"/>
        <strong>📈 PERFORMANCE ({matching_mode.upper()}):</strong><br/>
        &nbsp;&nbsp;• Recall: <strong>{recall:.1%}</strong><br/>
        &nbsp;&nbsp;• Precision: <strong>{precision:.1%}</strong><br/>
        &nbsp;&nbsp;• 🛡️ PII Protection: <strong>{pii_protection_rate:.1%}</strong><br/>
        <hr style="margin: 8px 0;"/>
        <strong style="color: {status_color};">🎯 STATUS:</strong><br/>
        <span style="color: {status_color};">{status}</span><br/>
        <hr style="margin: 8px 0;"/>
        """
        
        html_content += f"""
        <tr>
            <td class="metrics-col">{metrics_content}</td>
            <td class="original-col">{highlighted_original}</td>
            <td class="cleaned-col">{highlighted_cleaned}</td>
        </tr>
        """
    
    html_content += """
        </tbody>
    </table>
    """
    
    return html_content 


def analyze_missed_pii_categories(results_df: pd.DataFrame, 
                                ground_truth_df: pd.DataFrame,
                                matching_mode: str = 'business') -> Dict:
    """
    Analyze missed PII by categories to identify improvement opportunities.
    
    Args:
        results_df: DataFrame with processing results
        ground_truth_df: DataFrame with ground truth data
        matching_mode: 'business' or 'research' evaluation mode
        
    Returns:
        Dictionary with category analysis including:
        - missed_by_category: Count of missed PII by type
        - transcripts_with_misses: Examples where each category was missed
        - transcripts_with_detections: Examples where each category was detected
        - improvement_insights: Patterns and recommendations
    """
    print("🔍 ANALYZING MISSED PII BY CATEGORIES")
    print("=" * 50)
    
    evaluator = PIIEvaluator(matching_mode=matching_mode)
    
    # Track missed PII by category
    missed_by_category = {}
    transcripts_with_misses = {}
    transcripts_with_detections = {}
    
    # Track all PII categories seen
    all_categories = set()
    
    for idx, result_row in results_df.iterrows():
        call_id = result_row['call_id']
        
        try:
            gt_row = ground_truth_df[ground_truth_df['call_id'] == call_id].iloc[0]
        except IndexError:
            print(f"⚠️ No ground truth found for {call_id}, skipping")
            continue
        
        # Prepare ground truth for evaluation (using new structure)
        ground_truth = {
            'member_first_name': gt_row['member_first_name'],
            'member_full_name': gt_row['member_full_name'],  # NEW: Full name field
            'member_email': gt_row['member_email'],
            'member_mobile': gt_row['member_mobile'],
            'member_address': gt_row['member_address'],
            'member_number': str(gt_row['member_number']),
            'consultant_first_name': gt_row['consultant_first_name']
        }
        
        # Get evaluation results
        eval_result = evaluator.evaluate_single_transcript_public(
            original_text=result_row['original_transcript'],
            detected_pii=result_row['pii_detections'],
            ground_truth_pii=ground_truth,
            call_id=call_id
        )
        
        # Analyze matches for categories
        for match in eval_result['matches']:
            category_key = match.ground_truth_type if match.ground_truth_type else match.detected_type
            all_categories.add(category_key)
            
            if match.match_type == 'missed':
                # Track missed PII
                if category_key not in missed_by_category:
                    missed_by_category[category_key] = 0
                    transcripts_with_misses[category_key] = []
                
                missed_by_category[category_key] += 1
                transcripts_with_misses[category_key].append({
                    'call_id': call_id,
                    'missed_value': match.ground_truth_value,
                    'original_text': result_row['original_transcript'][:200] + "...",
                    'context': result_row['original_transcript'][max(0, match.start_pos-50):match.end_pos+50]
                })
            
            elif match.match_type in ['exact', 'partial']:
                # Track successful detections for comparison
                if category_key not in transcripts_with_detections:
                    transcripts_with_detections[category_key] = []
                
                transcripts_with_detections[category_key].append({
                    'call_id': call_id,
                    'detected_value': match.detected_value,
                    'ground_truth_value': match.ground_truth_value,
                    'overlap_ratio': match.overlap_ratio,
                    'original_text': result_row['original_transcript'][:200] + "...",
                    'context': result_row['original_transcript'][max(0, match.start_pos-50):match.end_pos+50]
                })
    
    # Calculate improvement insights
    improvement_insights = {}
    
    for category in all_categories:
        missed_count = missed_by_category.get(category, 0)
        detected_count = len(transcripts_with_detections.get(category, []))
        total_count = missed_count + detected_count
        
        if total_count > 0:
            miss_rate = missed_count / total_count
            improvement_insights[category] = {
                'miss_rate': miss_rate,
                'total_occurrences': total_count,
                'missed_count': missed_count,
                'detected_count': detected_count,
                'priority': 'HIGH' if miss_rate > 0.3 else 'MEDIUM' if miss_rate > 0.1 else 'LOW'
            }
    
    # Print summary
    print(f"\n📊 MISSED PII SUMMARY:")
    # Sort by recall (1 - miss_rate) in ascending order to show worst performers first
    for category, count in sorted(missed_by_category.items(), key=lambda x: 1 - improvement_insights[x[0]]['miss_rate']):
        total = improvement_insights[category]['total_occurrences']
        miss_rate = improvement_insights[category]['miss_rate']
        recall = 1 - miss_rate
        priority = improvement_insights[category]['priority']
        print(f"  {category:20} | Recall: {recall:.1%} | Missed: {count:3d}/{total:3d} | Priority: {priority}")
    
    return {
        'missed_by_category': missed_by_category,
        'transcripts_with_misses': transcripts_with_misses,
        'transcripts_with_detections': transcripts_with_detections,
        'improvement_insights': improvement_insights,
        'all_categories': sorted(list(all_categories))
    }


def analyze_confidence_vs_correctness(results_df: pd.DataFrame, 
                                    ground_truth_df: pd.DataFrame,
                                    transcript_column: str = 'original_transcript',
                                    matching_mode: str = 'business') -> Dict:
    """
    Analyze confidence levels vs correctness to understand model thinking.
    
    Args:
        results_df: DataFrame with processing results
        transcript_column: Column name for transcript to be dedacted
        ground_truth_df: DataFrame with ground truth data
        matching_mode: 'business' or 'research' evaluation mode
        
    Returns:
        Dictionary with confidence analysis including:
        - high_confidence_correct: Cases where model is confident and correct
        - high_confidence_wrong: Cases where model is confident but wrong
        - low_confidence_correct: Cases where model is uncertain but correct
        - low_confidence_wrong: Cases where model is uncertain and wrong
        - confidence_thresholds: Analysis of optimal confidence thresholds
    """
    print("🔍 ANALYZING CONFIDENCE vs CORRECTNESS")
    print("=" * 50)
    
    evaluator = PIIEvaluator(matching_mode=matching_mode)
    
    # Confidence thresholds
    HIGH_CONFIDENCE = 0.8
    LOW_CONFIDENCE = 0.5
    
    # Results buckets
    high_confidence_correct = []
    high_confidence_wrong = []
    low_confidence_correct = []
    low_confidence_wrong = []
    
    # Confidence distribution analysis
    confidence_scores = []
    
    for idx, result_row in results_df.iterrows():
        call_id = result_row['call_id']
        
        try:
            gt_row = ground_truth_df[ground_truth_df['call_id'] == call_id].iloc[0]
        except IndexError:
            continue
        
        # Prepare ground truth for evaluation (using new structure)
        ground_truth = {
            'member_first_name': gt_row['member_first_name'],
            'member_full_name': gt_row['member_full_name'],  # NEW: Full name field
            'member_email': gt_row['member_email'],
            'member_mobile': gt_row['member_mobile'],
            'member_address': gt_row['member_address'],
            'member_number': str(gt_row['member_number']),
            'consultant_first_name': gt_row['consultant_first_name']
        }
        
        # Get evaluation results
        eval_result = evaluator.evaluate_single_transcript_public(
            original_text=result_row[transcript_column],
            detected_pii=result_row['pii_detections'],
            ground_truth_pii=ground_truth,
            call_id=call_id
        )
        
        # Analyze each detection's confidence vs correctness
        for detection in result_row['pii_detections']:
            confidence = detection['score']
            confidence_scores.append(confidence)
            
            # Find corresponding match
            match_found = False
            is_correct = False
            
            for match in eval_result['matches']:
                if match.detected_value == detection['text']:
                    match_found = True
                    is_correct = match.match_type in ['exact', 'partial']
                    break
            
            # If no match found, it's an over-detection (wrong)
            if not match_found:
                is_correct = False
            
            # Create analysis entry
            analysis_entry = {
                'call_id': call_id,
                'detected_value': detection['text'],
                'detected_type': detection['entity_type'],
                'model_confidence': confidence,  
                'is_correct': is_correct,
                'context': result_row['original_transcript'][
                    max(0, detection['start']-50):detection['end']+50
                ]
            }
            
            # Categorize by confidence and correctness
            if confidence >= HIGH_CONFIDENCE:
                if is_correct:
                    high_confidence_correct.append(analysis_entry)
                else:
                    high_confidence_wrong.append(analysis_entry)
            elif confidence <= LOW_CONFIDENCE:
                if is_correct:
                    low_confidence_correct.append(analysis_entry)
                else:
                    low_confidence_wrong.append(analysis_entry)
    
    # Calculate confidence threshold analysis
    confidence_thresholds = {}
    for threshold in [0.3, 0.5, 0.7, 0.8, 0.9]:
        above_threshold = [score for score in confidence_scores if score >= threshold]
        below_threshold = [score for score in confidence_scores if score < threshold]
        
        confidence_thresholds[threshold] = {
            'above_count': len(above_threshold),
            'below_count': len(below_threshold),
            'percentage_above': len(above_threshold) / len(confidence_scores) * 100 if confidence_scores else 0
        }
    
    # Print summary
    print(f"\n📊 CONFIDENCE vs CORRECTNESS SUMMARY:")
    print(f"  High Confidence + Correct:    {len(high_confidence_correct):3d} cases")
    print(f"  High Confidence + Wrong:      {len(high_confidence_wrong):3d} cases")
    print(f"  Low Confidence + Correct:     {len(low_confidence_correct):3d} cases")
    print(f"  Low Confidence + Wrong:       {len(low_confidence_wrong):3d} cases")
    
    # Calculate insights
    total_detections = len(confidence_scores)
    if total_detections > 0:
        high_conf_accuracy = len(high_confidence_correct) / (len(high_confidence_correct) + len(high_confidence_wrong)) if (len(high_confidence_correct) + len(high_confidence_wrong)) > 0 else 0
        low_conf_accuracy = len(low_confidence_correct) / (len(low_confidence_correct) + len(low_confidence_wrong)) if (len(low_confidence_correct) + len(low_confidence_wrong)) > 0 else 0
        
        print(f"\n🎯 INSIGHTS:")
        print(f"  High Confidence Accuracy: {high_conf_accuracy:.1%}")
        print(f"  Low Confidence Accuracy:  {low_conf_accuracy:.1%}")
        print(f"  Avg Confidence Score:     {sum(confidence_scores)/len(confidence_scores):.3f}")
    
    return {
        'high_confidence_correct': high_confidence_correct,
        'high_confidence_wrong': high_confidence_wrong,
        'low_confidence_correct': low_confidence_correct,
        'low_confidence_wrong': low_confidence_wrong,
        'confidence_thresholds': confidence_thresholds,
        'summary_stats': {
            'total_detections': total_detections,
            'avg_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            'high_conf_accuracy': high_conf_accuracy if total_detections > 0 else 0,
            'low_conf_accuracy': low_conf_accuracy if total_detections > 0 else 0
        }
    } 

def debug_pii_counting_discrepancy(transcript_text: str, ground_truth_data: Dict, call_id: str = "test") -> Dict:
    """
    Debug function to trace PII counting discrepancies.
    
    Args:
        transcript_text: The original transcript text
        ground_truth_data: Dictionary with ground truth PII fields
        call_id: Identifier for this test case
        
    Returns:
        Dictionary with detailed counting analysis
    """
    print(f"🔍 DEBUGGING PII COUNTING FOR CALL {call_id}")
    print(f"=" * 60)
    print(f"📄 TRANSCRIPT:")
    print(f"   {transcript_text}")
    print(f"\n🎯 GROUND TRUTH DATA:")
    for field, value in ground_truth_data.items():
        if pd.notna(value) and str(value).strip():
            print(f"   {field}: '{value}'")
    
    # Initialize evaluator to use its extraction logic
    evaluator = PIIEvaluator(matching_mode='business')
    
    # Create a mock pandas Series for the ground truth
    gt_series = pd.Series(ground_truth_data)
    
    # Call the actual extraction function being used
    extracted_pii = evaluator._extract_ground_truth_pii(gt_series, transcript_text)
    
    print(f"\n🤖 SYSTEM EXTRACTED PII ({len(extracted_pii)} total):")
    for i, pii_item in enumerate(extracted_pii, 1):
        start, end = pii_item['start'], pii_item['end']
        extracted_text = transcript_text[start:end]
        context = transcript_text[max(0, start-15):end+15]
        print(f"   {i:2d}. {pii_item['type']:20} | '{pii_item['value']:15}' | Pos: {start:3d}-{end:3d} | Found: '{extracted_text}' | Context: ...{context}...")
    
    # Manual verification against expected PII
    print(f"\n✋ MANUAL VERIFICATION:")
    print(f"   Expected PII occurrences from your manual count:")
    
    expected_pii = [
        ("consultant_first_name", "Ava", "appears once"),
        ("member_mobile", "041648 996 374", "appears once"),
        ("member_email", "ella.wilson@example.com", "appears once"),
        ("member_number", "95924617", "appears once"),
        ("member_address", "34 Church Street, Adelaide SA 5000", "appears once"),
        ("member_first_name", "Ella", "appears TWICE"),
        ("member_full_name", "Ella Michael Wilson", "appears once"),  # NEW: Full name
    ]
    
    manual_count = 0
    for pii_type, pii_value, note in expected_pii:
        if pii_type == "member_first_name" and pii_value == "Ella":
            manual_count += 2  # Ella appears twice
        else:
            manual_count += 1
        print(f"   • {pii_type:20} | '{pii_value}' | {note}")
    
    print(f"\n📊 COUNT COMPARISON:")
    print(f"   Manual count (expected):  {manual_count}")
    print(f"   System count (actual):    {len(extracted_pii)}")
    print(f"   Discrepancy:              {len(extracted_pii) - manual_count}")
    print(f"\n🔍 NEW STRUCTURE BENEFIT:")
    print(f"   ✅ Using member_first_name + member_full_name structure")
    print(f"   ✅ More realistic PII counting (8 items instead of 11)")
    print(f"   ✅ Eliminates artificial middle/last name separation")
    
    # Identify potential issues
    print(f"\n🔍 ISSUE ANALYSIS:")
    if len(extracted_pii) > manual_count:
        print(f"   ❌ OVER-COUNTING: System found {len(extracted_pii) - manual_count} extra PII occurrences")
        print(f"   Possible causes:")
        print(f"   • Duplicate position matches")
        print(f"   • Spurious text matches")
        print(f"   • Incorrect word boundary detection")
        
        # Check for duplicates
        positions = [(pii['start'], pii['end']) for pii in extracted_pii]
        unique_positions = set(positions)
        if len(positions) != len(unique_positions):
            print(f"   • FOUND DUPLICATE POSITIONS!")
            
        # Check for overlapping matches
        values = [pii['value'] for pii in extracted_pii]
        from collections import Counter
        value_counts = Counter(values)
        for value, count in value_counts.items():
            if count > 1:
                expected_count = 2 if value == "Ella" else 1
                if count > expected_count:
                    print(f"   • '{value}' found {count} times (expected: {expected_count})")
                    
    elif len(extracted_pii) < manual_count:
        print(f"   ❌ UNDER-COUNTING: System missed {manual_count - len(extracted_pii)} PII occurrences")
    else:
        print(f"   ✅ COUNTS MATCH: System counting appears correct")
    
    return {
        'transcript': transcript_text,
        'ground_truth': ground_truth_data,
        'extracted_pii': extracted_pii,
        'manual_count': manual_count,
        'system_count': len(extracted_pii),
        'discrepancy': len(extracted_pii) - manual_count,
        'call_id': call_id
    }


def test_transcript_counting():
    """Test the specific transcript case where manual count = 9 but system reports 11."""
    
    # Your test transcript (removing ** highlighting)
    test_transcript = """Agent: Hi, this is Ava from Bricks Health Insurance. Agent: May I have your mobile number? Customer: 041648 996 374. Agent: And your email address, please? Customer: ella.wilson@example.com. Agent: Could I please have your Bricks membership number? Customer: 95924617. Agent: Finally, could you provide your residential address? Customer: 34 Church Street, Adelaide SA 5000. Agent: Could you confirm your full name, please? Customer: Ella Michael Wilson. Agent: Thank you for verifying, Ella. How can I assist you today?"""
    
    # Ground truth data (using NEW structure with member_first_name + member_full_name)
    test_ground_truth = {
        'consultant_first_name': 'Ava',
        'member_first_name': 'Ella',
        'member_full_name': 'Ella Michael Wilson',  # NEW: Full name field
        'member_email': 'ella.wilson@example.com',
        'member_mobile': '041648 996 374',
        'member_address': '34 Church Street, Adelaide SA 5000',
        'member_number': '95924617'
    }
    
    # Run the debug analysis
    debug_result = debug_pii_counting_discrepancy(
        transcript_text=test_transcript,
        ground_truth_data=test_ground_truth,
        call_id="DEBUG_TEST"
    )
    
    return debug_result 

def create_three_stage_html_table(
    transcript_data: List[Dict],
    title: str = "Three-Stage Workflow Analysis",
    description: str = "",
    matching_mode: str = 'business',
    show_original: bool = True
) -> str:
    """
    Create a diagnostic HTML table for the three-stage workflow.
    
    This function is a wrapper around create_diagnostic_html_table_configurable
    that adapts the data for three-stage workflow display.
    
    Args:
        transcript_data: List of transcript data dictionaries (must include 'stage_a_original', 'stage_b_normalized', 'stage_c_cleaned')
        title: Title for the analysis table
        description: Description text to display
        matching_mode: 'business' (default) or 'research'
        show_original: Whether to show the original transcript column
        
    Returns:
        HTML string for display in notebooks
    """
    
    # Adapt the data to work with the existing configurable function
    adapted_data = []
    
    for data in transcript_data:
        # Create adapted data structure that matches what create_diagnostic_html_table_configurable expects
        adapted_item = {
            'call_id': data['call_id'],
            # Use normalized transcript as the "original" for evaluation (since that's what we evaluate against)
            'original': data.get('stage_b_normalized', ''),
            # Use cleaned transcript as the "anonymized" 
            'anonymized': data.get('stage_c_cleaned', ''),
            # Keep the same PII detections
            'detected_pii': data.get('pii_detections', data.get('detected_pii', [])),
            # Keep all ground truth fields
            'member_first_name': data['member_first_name'],
            'member_full_name': data['member_full_name'],
            'member_email': data['member_email'],
            'member_mobile': data['member_mobile'],
            'member_address': data['member_address'],
            'member_number': data['member_number'],
            'consultant_first_name': data['consultant_first_name']
        }
        adapted_data.append(adapted_item)
    
    # Call the existing function with adapted data
    base_html = create_diagnostic_html_table_configurable(
        transcript_data=adapted_data,
        title=title,
        description=description,
        matching_mode=matching_mode
    )
    
    # If we need to show the original transcript, we need to modify the HTML
    if show_original:
        # Replace the table header to include original column
        base_html = base_html.replace(
            '<th style="width: 25%;">📊 Metrics & Performance</th>',
            '<th style="width: 20%;">📊 Metrics & Performance</th>'
        )
        base_html = base_html.replace(
            '<th style="width: 37.5%;">📋 Original Transcript</th>',
            '<th style="width: 20%;">📋 Original Transcript</th>'
        )
        base_html = base_html.replace(
            '<th style="width: 37.5%;">🛡️ Cleaned Transcript</th>',
            '<th style="width: 30%;">🔄 Normalized Transcript</th><th style="width: 30%;">🛡️ Cleaned Transcript</th>'
        )
        
        # Add original transcript column to each row
        for data in transcript_data:
            call_id = data['call_id']
            original_text = data.get('stage_a_original', '')
            
            # Find the row for this call_id and insert the original transcript
            # This is a simple replacement - in a more complex scenario, we might need more sophisticated HTML parsing
            row_start = base_html.find(f'<td class="metrics-col">')
            if row_start != -1:
                # Find the end of the metrics column
                metrics_end = base_html.find('</td>', row_start) + 5
                # Insert original transcript column after metrics
                original_cell = f'<td class="original-col">{html.escape(original_text)}</td>'
                base_html = base_html[:metrics_end] + original_cell + base_html[metrics_end:]
    
    return base_html 
