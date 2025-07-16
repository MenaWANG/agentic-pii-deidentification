"""
Utility functions for data loading, preparation, and combined processing workflows.
"""
from typing import Dict, List, Optional
import sys
from pathlib import Path
import time
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.text_normaliser import TextNormaliser
from utils.text_sweeper import TextSweeper
from baseline.presidio_framework import PurePresidioFramework


def process_transcript_with_normalization(text: str, 
                                        normalizer: TextNormaliser = None,
                                        presidio_framework: PurePresidioFramework = None,
                                        sweeper: TextSweeper = None,
                                        email_username_words: int = 2,
                                        apply_sweeping: bool = False,
                                        sweep_months: bool = True,
                                        sweep_ordinals: bool = True,
                                        custom_sweeping_dict: Optional[Dict[str, List[str]]] = None) -> Dict:
    """
    Complete multi-stage workflow: Raw → Normalize → (Optional: Sweep) → Clean
    
    Combines TextNormaliser preprocessing with optional TextSweeper layer and 
    PurePresidioFramework processing to provide enhanced PII detection on normalized text.
    
    Args:
        text (str): Original transcript text
        normalizer (TextNormaliser, optional): Pre-initialized normalizer. Creates new if None.
        presidio_framework (PurePresidioFramework, optional): Pre-initialized framework. Creates new if None.
        sweeper (TextSweeper, optional): Pre-initialized sweeper. Creates new if None.
        email_username_words (int): Number of words to consider for email username normalization
        apply_sweeping (bool): Whether to apply the sweeping layer after normalization
        sweep_months (bool): Whether to replace month names with <MONTH>
        sweep_ordinals (bool): Whether to replace ordinal numbers with <GENERIC_NUMBER>
        custom_sweeping_dict (Dict[str, List[str]], optional): Custom entity replacements
        
    Returns:
        Dict: Combined results with structure:
        {
            'original_text': str,           # Original input text
            'normalized_text': str,         # After TextNormaliser processing
            'swept_text': str,              # After TextSweeper processing (if enabled)
            'anonymized_text': str,         # After Presidio anonymization
            'pii_detections': List[Dict],   # PII detections found
            'processing_time': float,       # Total processing time
            'normalization_time': float,    # Time spent on normalization
            'sweeping_time': float,         # Time spent on sweeping (if enabled)
            'presidio_time': float,         # Time spent on Presidio processing
            'pii_count': int,               # Number of PII entities detected
            'custom_detections': Dict,      # Custom entity counts
            'normalization_applied': bool,  # Whether normalization was applied
            'sweeping_applied': bool,       # Whether sweeping was applied
            'workflow': str,                # Workflow identifier
        }
    """
    
    start_time = time.time()
    
    # Step 1: Initialize components if not provided
    if normalizer is None:
        normalizer = TextNormaliser(email_username_words=email_username_words)
    
    if presidio_framework is None:
        presidio_framework = PurePresidioFramework(enable_mlflow=False)
        
    if sweeper is None and apply_sweeping:
        sweeper = TextSweeper()
    
    # Step 2: Normalization (Raw → Normalized)
    norm_start = time.time()
    normalized_text = normalizer.normalize_text(text)
    norm_time = time.time() - norm_start
    
    # Step 3 (Optional): Sweeping (Normalized → Swept)
    sweep_time = 0
    swept_text = normalized_text
    
    if apply_sweeping and sweeper is not None:
        sweep_start = time.time()
        swept_text = sweeper.sweep_text(
            normalized_text,
            sweep_months=sweep_months,
            sweep_ordinals=sweep_ordinals,
            custom_dict=custom_sweeping_dict
        )
        sweep_time = time.time() - sweep_start
    
    # Step 4: PII Detection & Anonymization (Normalized/Swept → Clean)
    presidio_start = time.time()
    # Use swept text if sweeping was applied, otherwise use normalized text
    presidio_input = swept_text if apply_sweeping else normalized_text
    presidio_results = presidio_framework.process_transcript(presidio_input)
    presidio_time = time.time() - presidio_start
    
    total_time = time.time() - start_time
    
    # Step 5: Combine results
    combined_results = {
        # Multi-stage workflow outputs
        'original_text': text,
        'normalized_text': normalized_text,
        'anonymized_text': presidio_results['anonymized_text'],
        
        # Detection results (from Presidio on normalized/swept text)
        'pii_detections': presidio_results['pii_detections'],
        'pii_count': presidio_results['pii_count'],
        
        # Timing breakdown
        'processing_time': total_time,
        'normalization_time': norm_time,
        'presidio_time': presidio_time,
        
        # Additional metadata
        'workflow': 'normalization_sweeping_presidio' if apply_sweeping else 'normalization_presidio',
        'normalization_applied': True,
        'custom_detections': presidio_results.get('custom_detections', None)
    }
    
    # Add sweeping-related fields if it was applied
    if apply_sweeping:
        combined_results['swept_text'] = swept_text
        combined_results['sweeping_time'] = sweep_time
        combined_results['sweeping_applied'] = True
    else:
        combined_results['sweeping_applied'] = False
    
    return combined_results


def process_dataset_with_normalization(raw_df: pd.DataFrame, 
                                       id_column: str = "call_id",
                                       apply_sweeping: bool = False,
                                       sweep_months: bool = True,
                                       sweep_ordinals: bool = True,
                                       custom_sweeping_dict: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
        """
        Process the entire dataset of call transcripts.
        This is for quick demo notebook only. 
        
        Args:
            raw_df: DataFrame containing the raw data
            id_column: Column name containing the call_id
            apply_sweeping: Whether to apply the sweeping layer after normalization
            sweep_months: Whether to replace month names with <MONTH>
            sweep_ordinals: Whether to replace ordinal numbers with <GENERIC_NUMBER>
            custom_sweeping_dict: Custom entity replacements dictionary
            
        Returns:
            DataFrame with anonymized transcripts and analysis
        """
        workflow_name = "Normalization + Sweeping + Presidio" if apply_sweeping else "Normalization + Presidio"
        print(f"🚀 Starting Integrated {workflow_name} Framework processing...")
        
        # Initialize components once for reuse
        normalizer = TextNormaliser()
        presidio_framework = PurePresidioFramework(enable_mlflow=False)
        sweeper = TextSweeper() if apply_sweeping else None
        
        results = []
        processing_time = 0
        presidio_time = 0
        normalization_time = 0
        sweeping_time = 0
        
        for idx, row in raw_df.iterrows():
            print(f"Processing transcript {idx+1}/{len(raw_df)}...", end='\r')
            original_transcript = row['call_transcript']
            
            result = process_transcript_with_normalization(
                text=original_transcript,
                normalizer=normalizer,
                presidio_framework=presidio_framework,
                sweeper=sweeper,
                apply_sweeping=apply_sweeping,
                sweep_months=sweep_months,
                sweep_ordinals=sweep_ordinals,
                custom_sweeping_dict=custom_sweeping_dict
            )
            
            result[id_column] = row[id_column]
            processing_time += result['processing_time']
            presidio_time += result['presidio_time']
            normalization_time += result['normalization_time']
            
            if apply_sweeping and 'sweeping_time' in result:
                sweeping_time += result['sweeping_time']
                
            results.append(result)
            
        cleaned_df = pd.DataFrame(results)
        column_renames = {
            'original_text': 'original_transcript', 
            'normalized_text': 'normalized_transcript', 
            'anonymized_text': 'anonymized_transcript'
        }
        
        if apply_sweeping:
            column_renames['swept_text'] = 'swept_transcript'
            
        cleaned_df.rename(columns=column_renames, inplace=True)
        
        final_metrics = _calculate_processing_metrics(
            cleaned_df, 
            processing_time, 
            presidio_time, 
            normalization_time,
            sweeping_time if apply_sweeping else 0,
            apply_sweeping
        )
        
        print("\n✅ Processing complete! Final metrics:")
        # Print all metrics
        metrics_to_print = list(final_metrics.items())
        for metric, value in metrics_to_print:
            print(f" • {metric}: {value}")
        
        return cleaned_df
        
def _calculate_processing_metrics(results_df: pd.DataFrame, 
                                    total_processing_time: float,
                                    presidio_time: float,
                                    normalization_time: float,
                                    sweeping_time: float = 0,
                                    sweeping_applied: bool = False) -> Dict:
    """Calculate final processing metrics with detailed timing information."""
    num_transcripts = len(results_df)
    avg_time_per_transcript = total_processing_time / num_transcripts if num_transcripts > 0 else 0
    
    # Calculate 1M transcript processing estimates
    time_for_1m_seconds = avg_time_per_transcript * 1_000_000
    time_for_1m_minutes = time_for_1m_seconds / 60
    time_for_1m_hours = time_for_1m_minutes / 60
    time_for_1m_days = time_for_1m_hours / 24
    
    # Format time estimate for 1M transcripts
    if time_for_1m_days >= 1:
        time_1m_estimate = f"{time_for_1m_days:.2f} days"
    elif time_for_1m_hours >= 1:
        time_1m_estimate = f"{time_for_1m_hours:.2f} hours"
    elif time_for_1m_minutes >= 1:
        time_1m_estimate = f"{time_for_1m_minutes:.2f} minutes"
    else:
        time_1m_estimate = f"{time_for_1m_seconds:.4f} seconds"
    
    metrics = {
        'total_transcripts': num_transcripts,
        'total_pii_detected': results_df['pii_count'].sum(),
        'avg_pii_per_transcript': round(results_df['pii_count'].mean(), 2),
        'total_processing_time_seconds': round(total_processing_time, 4),
        'presidio_processing_time_seconds': round(presidio_time, 4),  # Fixed typo in 'presidio'
        'normalization_processing_time_seconds': round(normalization_time, 4),
        'workflow_stages': 'Normalization + Sweeping + Presidio' if sweeping_applied else 'Normalization + Presidio',
        'avg_processing_time_per_transcript_seconds': round(avg_time_per_transcript, 4),
        'estimated_time_for_1m_transcripts': time_1m_estimate
    }
    
    # Add sweeping metrics if applicable
    if sweeping_applied:
        metrics['sweeping_processing_time_seconds'] = round(sweeping_time, 4)
        metrics['sweeping_percentage_of_total'] = round((sweeping_time / total_processing_time) * 100, 2) if total_processing_time > 0 else 0
    
    return metrics     