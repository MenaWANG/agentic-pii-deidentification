"""
Pure Microsoft Presidio Implementation - Version A
This framework implements the baseline PII deidentification using only Microsoft Presidio
with custom recognizers for member numbers and enhanced address detection.
"""

import pandas as pd
import time
from typing import Dict, List, Tuple
from datetime import datetime
import json
import re

from presidio_analyzer import AnalyzerEngine, RecognizerResult, Pattern
from presidio_analyzer.recognizer_registry import RecognizerRegistry
from presidio_analyzer.pattern_recognizer import PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
import logging

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available - running without experiment tracking")


class PurePresidioFramework:
    """
    Pure Microsoft Presidio implementation for PII deidentification.
    Includes custom recognizers for 8-digit member numbers and improved australian address detection.
    """
    
    def __init__(self, enable_mlflow: bool = True):
        """Initialize the Presidio Framework with custom recognizers."""
        logging.getLogger('presidio-analyzer').setLevel(logging.ERROR)
        self.enable_mlflow = enable_mlflow and MLFLOW_AVAILABLE
        
        # Initialize Presidio engines with custom recognizers
        self.analyzer = self._setup_analyzer()
        self.anonymizer = AnonymizerEngine()
        
        # Statistics tracking
        self.stats = {
            'total_transcripts': 0,
            'total_pii_detected': 0,
            'processing_time': 0,
            'pii_types': {},
            'custom_detections': {
                'member_numbers': 0,
                'addresses': 0
            }
        }
        
        if self.enable_mlflow:
            self._setup_mlflow()
    
    # def _create_member_number_recognizer(self) -> PatternRecognizer:
    #     """Create custom recognizer for 8-digit member numbers."""
    #     # Pattern for exactly 8 digits (member numbers)
    #     member_number_patterns = [
    #         Pattern(
    #             name="member_number_8_digits",
    #             regex=r"\b\d{8}\b",
    #             score=0.9
    #         ),
    #         Pattern(
    #             name="member_number_with_spaces",
    #             regex=r"\b\d{4}\s?\d{4}\b",
    #             score=0.85
    #         )
    #     ]
        
    #     return PatternRecognizer(
    #         supported_entity="MEMBER_NUMBER",
    #         patterns=member_number_patterns,
    #         name="member_number_recognizer"
    #     )

    def _create_australian_phone_recognizer(self) -> PatternRecognizer:
        """Create custom recognizer for Australian phone numbers."""
        # Australian phone number patterns
        australian_phone_patterns = [
            Pattern(
                name="au_landline",
                regex=r'\b0[2-9]\s?\d{4}\s?\d{4}\b',       # 0X XXXX XXXX (landline)
                score=0.85
            ),
            Pattern(
                name="au_mobile_standard",
                regex=r'\b04\d{2}\s?\d{3}\s?\d{3}\b',      # 04XX XXX XXX (mobile)
                score=0.9
            ),
            Pattern(
                name="au_mobile_compact",
                regex=r'\b04\d{8}\b',                      # 04XXXXXXXX (mobile)
                score=0.9
            ),
            Pattern(
                name="au_international",
                regex=r'\b\+61\s?[2-9]\s?\d{4}\s?\d{4}\b', # +61 X XXXX XXXX (international)
                score=0.95
            ),
            Pattern(
                name="au_synthetic",
                regex=r'\b04\d{4}\s?\d{3}\s?\d{3}\b',      # 04XXXX XXX XXX (synthetic data)
                score=0.7
            )
        ]
        
        return PatternRecognizer(
            supported_entity="AU_PHONE_NUMBER",
            patterns=australian_phone_patterns,
            name="australian_phone_recognizer",
            context=["phone", "mobile", "cell", "number", "call"]
        )
    
    def _create_generic_number_recognizer(self) -> PatternRecognizer:
        """Create custom recognizer for any sequence of digits."""
        number_patterns = [
            Pattern(
                name="eight_digit_sequence",
                regex=r"\b\d{8}\b",  # Specifically target 8-digit sequences
                score=1.0 
            ),
            Pattern(
                name="any_digit_sequence",
                regex=r"\b\d{1,16}\b",  # Aggresively masking 1-16 digit sequences 
                score=0.9  
            ),
            Pattern(
                name="formatted_numbers",
                regex=r"\b\d{1,3}(,\d{3})+\b",  # Matches numbers with commas like 1,000,000
                score=0.65
            ),
            Pattern(
                name="decimal_numbers",
                regex=r"\b\d+\.\d+\b",  # Matches decimal numbers like 123.45
                score=0.65
            )
        ]
        
        return PatternRecognizer(
            supported_entity="GENERIC_NUMBER",
            patterns=number_patterns,
            name="generic_number_recognizer",
            context=["number", "digit", "id", "member", "account"]  
        )

    def _create_enhanced_address_recognizer(self) -> PatternRecognizer:
        """Create enhanced address recognizer for Australian addresses."""
        # Australian address patterns
        australian_address_patterns = [
            Pattern(
                name="australian_street_address",
                regex=r"\b\d{1,3}\s+(?:[A-Za-z]+\s+){1,5}(Street|St|Road|Rd|Avenue|Ave|Drive|Dr|Lane|Ln|Boulevard|Blvd|Circuit|Cct|Court|Ct|Place|Pl|Way|Crescent|Cres)\b,?\s*(?:[A-Za-z]+\s+){1,3}(NSW|VIC|QLD|WA|SA|TAS|ACT|NT)\s+\d{4}\b",
                score=0.95
            ),
            Pattern(
                name="australian_street_simple",
                regex=r"\b\d{1,3}\s+(?:[A-Za-z]+\s+){1,5}(Street|St|Road|Rd|Avenue|Ave|Drive|Dr|Lane|Ln|Boulevard|Blvd|Circuit|Cct|Court|Ct|Place|Pl|Way|Crescent|Cres)\b",
                score=0.7
            )
        ]
        
        return PatternRecognizer(
            supported_entity="AU_ADDRESS",
            patterns=australian_address_patterns,
            name="australian_address_recognizer"
        )
    
    def _setup_analyzer(self) -> AnalyzerEngine:
        """Setup analyzer with custom recognizers and Australian localization."""
        # Configure NLP engine
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
        
        nlp_engine_provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = nlp_engine_provider.create_engine()
        
        # Create registry and add custom recognizers
        registry = RecognizerRegistry()
        
        # Add default recognizers
        registry.load_predefined_recognizers(nlp_engine=nlp_engine, languages=["en"])
        
        # Add custom recognizers
        # member_number_recognizer = self._create_member_number_recognizer()
        enhanced_address_recognizer = self._create_enhanced_address_recognizer()
        generic_number_recognizer = self._create_generic_number_recognizer()
        au_phone_recognizer = self._create_australian_phone_recognizer()
        
        
        # registry.add_recognizer(member_number_recognizer)
        registry.add_recognizer(enhanced_address_recognizer)
        registry.add_recognizer(generic_number_recognizer)
        registry.add_recognizer(au_phone_recognizer)
        
        # Create analyzer with custom registry
        analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            registry=registry
        )
        
        # print("✅  Presidio Framework initialized with custom recognizers:")
        # print("   📍 8-digit member number recognizer")
        # print("   🏠 Enhanced Australian address recognizer")
        
        return analyzer
    
    def _setup_mlflow(self):
        """Setup MLflow experiment tracking."""
        if not self.enable_mlflow:
            return
            
        try:
            mlflow.set_experiment("presidio-baseline-pii-deidentification")
            print("✅ MLflow experiment tracking enabled")
        except Exception as e:
            print(f"⚠️ MLflow setup failed: {e}")
            self.enable_mlflow = False
    
    def process_transcript(self, text: str) -> Dict:
        """
        Process a single transcript using Presidio with custom recognizers.
        
        Args:
            text: Raw call transcript text
            
        Returns:
            Dictionary containing anonymized text and detection metadata
            {
            'original_text': str,           # Original input text
            'anonymized_text': str,         # After Presidio anonymization
            'pii_detections': List[Dict],   # PII detections found
            'processing_time': float,       # Total processing time
            'pii_count': int,               # Number of PII entities detected
            'workflow': str,                # Workflow identifier
            'custom_detections': Dict,      # Custom entity counts
            }
        """
        start_time = time.time()
        
        # Presidio analysis with custom entities
        analyzer_results = self.analyzer.analyze(
            text=text, 
            language='en',
            entities=[
                'PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 
                'LOCATION', 'CREDIT_CARD', 
                'GENERIC_NUMBER',  # Custom entity for generic numbers
                'AU_PHONE_NUMBER',  # Custom entity for Australian phone numbers                   
                'MEMBER_NUMBER',  # Custom entity
                'AU_ADDRESS'      # Custom entity
                # 'DATE_TIME','ORGANIZATION','IP_ADDRESS', # disabled 
            ]
        )
        
        # Direct Presidio anonymization
        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results
        )
        
        processing_time = time.time() - start_time
        
        # Extract PII information for analysis with custom entity tracking
        pii_detections = []
        custom_stats = {'member_numbers': 0, 'addresses': 0}
        
        for result in analyzer_results:
            pii_detections.append({
                'entity_type': result.entity_type,
                'start': result.start,
                'end': result.end,
                'score': result.score,
                'text': text[result.start:result.end]
            })
            
            # Track custom detections
            if result.entity_type == 'MEMBER_NUMBER':
                custom_stats['member_numbers'] += 1
            elif result.entity_type in ['AU_ADDRESS', 'LOCATION']:
                custom_stats['addresses'] += 1
        
        return {
            'anonymized_text': anonymized_result.text,
            'original_text': text,
            'pii_detections': pii_detections,
            'processing_time': processing_time,
            'pii_count': len(pii_detections),
            'custom_detections': custom_stats,
            'workflow': 'pure_presidio'  
        }
    
    def process_dataset(self, csv_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Process the entire dataset of call transcripts.
        This is for quick demo notebook only. 
        TODO: parameterize this method to be more generic, like the normalization_presidio_workflow
        
        Args:
            csv_path: Path to the synthetic_call_transcripts.csv file
            output_path: Optional path to save results
            
        Returns:
            DataFrame with anonymized transcripts and analysis
        """
        print("🚀 Starting Pure Presidio Framework processing...")
        
        if self.enable_mlflow:
            mlflow.start_run(run_name=f"presidio-baseline-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        try:
            # Load data
            df = pd.read_csv(csv_path)
            print(f"📊 Loaded {len(df)} call transcripts")
            
            # Process each transcript
            results = []
            total_start_time = time.time()
            
            for idx, row in df.iterrows():
                print(f"Processing transcript {idx+1}/{len(df)}...", end='\r')
                
                result = self.process_transcript(row['call_transcript'])
                
                # Add original data for reference (excluding ground truth during processing)
                result.update({
                    'call_id': row['call_id'],
                    'consultant_first_name': row['consultant_first_name'],
                    'original_transcript': row['call_transcript'],
                    'anonymized_transcript': result['anonymized_text'],
                })
                
                results.append(result)
                
                # Update statistics
                self._update_stats(result)
            
            total_processing_time = time.time() - total_start_time
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Log final metrics
            final_metrics = self._calculate_processing_metrics(results_df, total_processing_time)
            print("\n✅ Processing complete! Final metrics:")
            # Print all metrics except the last two (which are wordy distributions)
            metrics_to_print = list(final_metrics.items())[:-2]
            for metric, value in metrics_to_print:
                print(f"  • {metric}: {value}")
            
            # Save results
            if output_path:
                results_df.to_csv(output_path, index=False)
                # print(f"💾 Results saved to {output_path}")
            
            if self.enable_mlflow:
                self._log_mlflow_metrics(final_metrics, results_df)
            
            return results_df
            
        except Exception as e:
            print(f"❌ Error during processing: {e}")
            raise
        finally:
            if self.enable_mlflow:
                mlflow.end_run()
    
    def _update_stats(self, result: Dict):
        """Update processing statistics."""
        self.stats['total_transcripts'] += 1
        self.stats['total_pii_detected'] += result['pii_count']
        self.stats['processing_time'] += result['processing_time']
        
        # Track PII types
        for detection in result['pii_detections']:
            entity_type = detection['entity_type']
            if entity_type not in self.stats['pii_types']:
                self.stats['pii_types'][entity_type] = 0
            self.stats['pii_types'][entity_type] += 1
        
        # Track custom detections
        for detection_type, count in result['custom_detections'].items():
            self.stats['custom_detections'][detection_type] += count
    
    def _calculate_processing_metrics(self, results_df: pd.DataFrame, total_time: float) -> Dict:
        """Calculate final processing metrics with detailed timing information."""
        num_transcripts = len(results_df)
        avg_time_per_transcript = total_time / num_transcripts if num_transcripts > 0 else 0
        
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
            'total_processing_time_seconds': round(total_time, 4),
            'avg_processing_time_per_transcript_seconds': round(avg_time_per_transcript, 4),
            'estimated_time_for_1m_transcripts': time_1m_estimate,
            'pii_types_distribution': dict(self.stats['pii_types']),
            'custom_detections_distribution': dict(self.stats['custom_detections'])
        }
    
    def _log_mlflow_metrics(self, metrics: Dict, results_df: pd.DataFrame):
        """Log metrics to MLflow."""
        if not self.enable_mlflow:
            return
            
        try:
            # Log scalar metrics
            mlflow.log_metrics({
                'total_transcripts': metrics['total_transcripts'],
                'total_pii_detected': metrics['total_pii_detected'],
                'avg_pii_per_transcript': metrics['avg_pii_per_transcript'],
                'total_processing_time_seconds': metrics['total_processing_time_seconds'],
                'avg_processing_time_per_transcript_seconds': metrics['avg_processing_time_per_transcript_seconds']
            })
            
            # Log PII types distribution as JSON
            mlflow.log_dict(metrics['pii_types_distribution'], "pii_types_distribution.json")
            
            # Log custom detections distribution as JSON
            mlflow.log_dict(metrics['custom_detections_distribution'], "custom_detections_distribution.json")
            
            # Log sample results
            sample_results = results_df.head(5)[['call_id', 'pii_count', 'processing_time']].to_dict('records')
            mlflow.log_dict(sample_results, "sample_results.json")
            
            print("✅ MLflow metrics logged successfully")
            
        except Exception as e:
            print(f"⚠️ MLflow logging failed: {e}")


def test_presidio_installation():
    """Test function to verify Presidio installation and basic functionality."""
    print("🧪 Testing Presidio installation...")
    
    try:
        framework = PurePresidioFramework(enable_mlflow=False)
        
        # Test with a sample transcript
        test_text = """
        Agent: Hi, this is Sarah from Bricks Health Insurance.
        Agent: Could you confirm your full name, please?
        Customer: John Smith.
        Agent: And your email address, please?
        Customer: john.smith@example.com.
        Agent: May I have your mobile number?
        Customer: 0412 345 678.
        """
        
        result = framework.process_transcript(test_text)
        
        print("✅ Presidio installation test successful!")
        print(f"   Original text length: {len(test_text)}")
        print(f"   Anonymized text length: {len(result['anonymized_text'])}")
        print(f"   PII detections: {result['pii_count']}")
        print(f"   Processing time: {result['processing_time']:.4f} seconds")
        print(f"\n📝 Sample anonymized text:\n{result['anonymized_text'][:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Presidio installation test failed: {e}")
        return False


if __name__ == "__main__":
    # Test installation first
    if test_presidio_installation():
        print("\n" + "="*50)
        print("🎯 Ready to process full dataset!")
        print("Run: framework.process_dataset('.data/synthetic_call_transcripts.csv')")
    else:
        print("\n❌ Please fix installation issues before proceeding.")
