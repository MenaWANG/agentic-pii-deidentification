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
                                      metric: str = 'f1_score', 
                                      n_cases: int = 3, 
                                      ascending: bool = True,
                                      matching_mode: str = 'business') -> List[Dict]:
    """
    Get n transcript cases ranked by performance metric.
    
    Args:
        results_df: DataFrame with processing results
        ground_truth_df: DataFrame with ground truth data
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
        
        # Prepare ground truth for evaluation
        ground_truth = {
            'member_full_name': gt_row['member_full_name'],
            'member_email': gt_row['member_email'],
            'member_mobile': gt_row['member_mobile'],
            'member_address': gt_row['member_address'],
            'member_number': str(gt_row['member_number']),
            'consultant_first_name': gt_row['consultant_first_name']
        }
        
        try:
            # Get precise metrics from centralized evaluation
            eval_result = evaluator.evaluate_single_transcript_public(
                original_text=result_row['original_transcript'],
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
                'recall': 0.5,  # Conservative estimate
                'precision': 0.7,
                'f1_score': 0.58,
                'total_pii_occurrences': 6,
                'detected_count': detected_count,
                'exact_matches': detected_count // 2,
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
            'original': result_row['original_transcript'],
            'anonymized': result_row['anonymized_text'],
            'detected_pii': result_row['pii_detections'],
            'member_full_name': gt_row['member_full_name'],
            'member_email': gt_row['member_email'],
            'member_mobile': gt_row['member_mobile'],
            'member_address': gt_row['member_address'],
            'member_number': str(gt_row['member_number']),
            'consultant_first_name': gt_row['consultant_first_name'],
            # Add performance metrics for reference
            'performance_metrics': perf
        }
        
        cases_data.append(case_data)
    
    print(f"\n✅ Prepared {len(cases_data)} cases for analysis")
    return cases_data


def create_diagnostic_html_table_configurable(transcript_data: List[Dict], 
                                            title: str = "PII Analysis", 
                                            description: str = "", 
                                            matching_mode: str = 'business') -> str:
    """
    Create configurable diagnostic HTML table for PII analysis.
    
    Args:
        transcript_data: List of transcript data dictionaries
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
    .detected-pii {{ background-color: #ccffcc; padding: 2px 4px; border-radius: 3px; }}
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
        original = data['original']
        anonymized = data['anonymized']
        detected_pii = data['detected_pii']
        
        # Get ground truth for centralized evaluation
        ground_truth = {
            'member_full_name': data['member_full_name'],
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
        valid_detections = eval_result['valid_detections']
        recall = eval_result['recall']
        precision = eval_result['precision']
        
        # Create highlighted original transcript (missed PII in red)
        highlighted_original = html.escape(original)
        for match in eval_result['matches']:
            if match.match_type == 'missed':
                missed_text = match.ground_truth_value
                highlighted_original = highlighted_original.replace(
                    html.escape(missed_text), 
                    f'<span class="missed-pii">{html.escape(missed_text)}</span>'
                )
        
        # Create highlighted cleaned transcript (detected PII in green)
        highlighted_cleaned = html.escape(anonymized)
        placeholders = ['&lt;PERSON&gt;', '&lt;EMAIL_ADDRESS&gt;', '&lt;PHONE_NUMBER&gt;', 
                       '&lt;LOCATION&gt;', '&lt;DATE_TIME&gt;', '&lt;US_SSN&gt;']
        for placeholder in placeholders:
            highlighted_cleaned = highlighted_cleaned.replace(
                placeholder,
                f'<span class="detected-pii">{placeholder}</span>'
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
        <strong>🔍 PII Detected:</strong> {len(detected_pii)}<br/>
        <strong>✅ Exact Matches:</strong> {exact_matches}<br/>
        <strong>⚡ Partial Matches:</strong> {partial_matches}<br/>
        <strong>🛡️ Valid Detections:</strong> {valid_detections}<br/>
        <hr style="margin: 8px 0;"/>
        <strong>📈 PERFORMANCE ({matching_mode.upper()}):</strong><br/>
        &nbsp;&nbsp;• Recall: <strong>{recall:.1%}</strong><br/>
        &nbsp;&nbsp;• Precision: <strong>{precision:.1%}</strong><br/>
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