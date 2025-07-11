"""
Utility functions for data loading, preparation, and combined processing workflows.
"""
from typing import Dict
import sys
from pathlib import Path
import time

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


def create_combined_processing_demo():
    """
    Demo function showing the three-stage workflow with sample data.
    """
    print("🔄 Combined Processing Demo: Raw → Normalize → Clean")
    print("=" * 60)
    
    # Sample data that benefits from normalization
    test_cases = [
        "Hi c h l o e, call zero four one two three four five six seven eight",
        "Contact j o h n dot d o e at g m a i l dot c o m for member eight eight eight eight one one one one",
        "My mobile is zero four nine nine eight eight seven seven six six"
    ]
    
    # Initialize components once for efficiency
    normalizer = TextNormaliser()
    presidio = PurePresidioFramework(enable_mlflow=False)
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}:")
        print(f"Raw: {test_text}")
        
        result = process_transcript_with_normalization(
            test_text, 
            normalizer=normalizer, 
            presidio_framework=presidio
        )
        
        print(f"Normalized: {result['normalized_transcript']}")
        print(f"Anonymized: {result['anonymized_transcript']}")
        print(f"PII Found: {result['pii_count']} entities")
        print(f"Timing: {result['normalization_time']:.3f}s norm + {result['presidio_time']:.3f}s presidio = {result['processing_time']:.3f}s total")
        
        if result['pii_detections']:
            for detection in result['pii_detections']:
                print(f"  - {detection['entity_type']}: '{detection['text']}' (confidence: {detection['confidence']:.2f})")


if __name__ == "__main__":
    # Run demo
    create_combined_processing_demo()
