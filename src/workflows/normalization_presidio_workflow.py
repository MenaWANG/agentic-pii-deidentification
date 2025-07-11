"""
Utility functions for data loading, preparation, and combined processing workflows.
"""
from typing import Dict
import sys
from pathlib import Path
import time
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.text_normaliser import TextNormaliser
from baseline.presidio_framework import PurePresidioFramework


def process_transcript_with_normalization(text: str, 
                                        normalizer: TextNormaliser = None,
                                        presidio_framework: PurePresidioFramework = None,
                                        email_username_words: int = 2) -> Dict:
    """
    Complete three-stage workflow: Raw → Normalize → Clean
    
    Combines TextNormaliser preprocessing with PurePresidioFramework processing
    to provide enhanced PII detection on normalized text.
    
    Args:
        text (str): Original transcript text
        normalizer (TextNormaliser, optional): Pre-initialized normalizer. Creates new if None.
        presidio_framework (PurePresidioFramework, optional): Pre-initialized framework. Creates new if None.
        email_username_words (int): Number of words to consider for email username normalization
        
    Returns:
        Dict: Combined results with structure:
        {
            'original_text': str,           # Original input text
            'normalized_text': str,         # After TextNormaliser processing
            'anonymized_text': str,         # After Presidio anonymization
            'pii_detections': List[Dict],   # PII detections found
            'processing_time': float,       # Total processing time
            'normalization_time': float,    # Time spent on normalization
            'presidio_time': float,         # Time spent on Presidio processing
            'pii_count': int,               # Number of PII entities detected
            'custom_detections': Dict,      # Custom entity counts
            'normalization_applied': bool,  # Whether normalization was applied
            'workflow': str,                # Workflow identifier
        }
    """

    
    start_time = time.time()
    
    # Step 1: Initialize components if not provided
    if normalizer is None:
        normalizer = TextNormaliser(email_username_words=email_username_words)
    
    if presidio_framework is None:
        presidio_framework = PurePresidioFramework(enable_mlflow=False)
    
    # Step 2: Normalization (Raw → Normalized)
    norm_start = time.time()
    normalized_text = normalizer.normalize_text(text)
    norm_time = time.time() - norm_start
    
    # Step 3: PII Detection & Anonymization (Normalized → Clean)
    presidio_start = time.time()
    presidio_results = presidio_framework.process_transcript(normalized_text)
    presidio_time = time.time() - presidio_start
    
    total_time = time.time() - start_time
    
    # Step 4: Combine results
    combined_results = {
        # Three-stage workflow outputs
        'original_text': text,
        'normalized_text': normalized_text,
        'anonymized_text': presidio_results['anonymized_text'],
        
        # Detection results (from Presidio on normalized text)
        'pii_detections': presidio_results['pii_detections'],
        'pii_count': presidio_results['pii_count'],
        
        # Timing breakdown
        'processing_time': total_time,
        'normalization_time': norm_time,
        'presidio_time': presidio_time,
        
        # Additional metadata
        'workflow': 'normalization_presidio',
        'normalization_applied': True,
        'custom_detections': presidio_results.get('custom_detections', None)
    }
    
    return combined_results


def process_dataset_with_normalization(raw_df: pd.DataFrame, 
                                       id_column: str = "call_id") -> pd.DataFrame:
        """
        Process the entire dataset of call transcripts.
        This is for quick demo notebook only. 
        
        Args:
            raw_df: DataFrame containing the raw data
            id_column: Column name containing the call_id
            
        Returns:
            DataFrame with anonymized transcripts and analysis
        """
        print("🚀 Starting Integrated Normalization + Presidio Framework processing...")
        
        results = []
        processing_time = 0
        presidio_time = 0
        normalization_time = 0
        for idx, row in raw_df.iterrows():
            print(f"Processing transcript {idx+1}/{len(raw_df)}...", end='\r')
            original_transcript = row['call_transcript']
            result = process_transcript_with_normalization(text=original_transcript,
                                                                       normalizer=TextNormaliser(),
                                                                       presidio_framework=PurePresidioFramework(enable_mlflow=False))
            result[id_column] = row[id_column]
            processing_time += result['processing_time']
            presidio_time += result['presidio_time']
            normalization_time += result['normalization_time']
            results.append(result)
            
        cleaned_df = pd.DataFrame(results)
        cleaned_df.rename(columns={'original_text': 'original_transcript', 'normalized_text': 'normalized_transcript', 'anonymized_text': 'anonymized_transcript'}, inplace=True)
        
        final_metrics = _calculate_processing_metrics(cleaned_df, processing_time, presidio_time, normalization_time)
        print("\n✅ Processing complete! Final metrics:")
        # Print all metrics except the last two (which are wordy distributions)
        metrics_to_print = list(final_metrics.items())
        for metric, value in metrics_to_print:
            print(f" • {metric}: {value}")
        
        return cleaned_df
        
def _calculate_processing_metrics(results_df: pd.DataFrame, 
                                    total_processing_time: float,
                                    presidio_time: float,
                                    normalization_time: float) -> Dict:
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
    
    return {
        'total_transcripts': num_transcripts,
        'total_pii_detected': results_df['pii_count'].sum(),
        'avg_pii_per_transcript': round(results_df['pii_count'].mean(), 2),
        'total_processing_time_seconds': round(total_processing_time, 4),
        'procedio_processing_time_seconds': round(presidio_time, 4), 
        'normalization_processing_time_seconds': round(normalization_time, 4),
        'avg_processing_time_per_transcript_seconds': round(avg_time_per_transcript, 4),
        'estimated_time_for_1m_transcripts': time_1m_estimate
    }                  




