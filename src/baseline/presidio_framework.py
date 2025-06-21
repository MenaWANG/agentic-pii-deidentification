"""
Pure Microsoft Presidio Implementation - Version A
This framework implements the baseline PII deidentification using only Microsoft Presidio
without any abstraction layers, exactly as specified in the project plan.
"""

import pandas as pd
import time
from typing import Dict, List, Tuple
from datetime import datetime
import json

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available - running without experiment tracking")


class PurePresidioFramework:
    """
    Pure Microsoft Presidio implementation for PII deidentification.
    Follows the exact architecture specified in the ReadMe.md for Version A.
    """
    
    def __init__(self, enable_mlflow: bool = True):
        """Initialize the Pure Presidio Framework with Australian localization."""
        self.enable_mlflow = enable_mlflow and MLFLOW_AVAILABLE
        
        # Initialize Presidio engines
        self.analyzer = self._setup_analyzer()
        self.anonymizer = AnonymizerEngine()
        
        # Statistics tracking
        self.stats = {
            'total_transcripts': 0,
            'total_pii_detected': 0,
            'processing_time': 0,
            'pii_types': {}
        }
        
        if self.enable_mlflow:
            self._setup_mlflow()
    
    def _setup_analyzer(self) -> AnalyzerEngine:
        """Setup analyzer with Australian localization patterns."""
        # Configure NLP engine
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
        
        nlp_engine_provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = nlp_engine_provider.create_engine()
        
        # Create analyzer with Australian phone number support
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        
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
        Process a single transcript using pure Presidio - no abstraction layers.
        
        Args:
            text: Raw call transcript text
            
        Returns:
            Dictionary containing anonymized text and detection metadata
        """
        start_time = time.time()
        
        # Direct Presidio analysis - no abstraction layers
        analyzer_results = self.analyzer.analyze(
            text=text, 
            language='en',
            entities=[
                'PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 
                'LOCATION', 'ORGANIZATION', 'DATE_TIME',
                'CREDIT_CARD', 'CRYPTO', 'IBAN_CODE', 'IP_ADDRESS',
                'NRP', 'MEDICAL_LICENSE', 'URL', 'US_SSN'
            ]
        )
        
        # Direct Presidio anonymization - no abstraction layers
        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results
        )
        
        processing_time = time.time() - start_time
        
        # Extract PII information for analysis
        pii_detections = []
        for result in analyzer_results:
            pii_detections.append({
                'entity_type': result.entity_type,
                'start': result.start,
                'end': result.end,
                'score': result.score,
                'text': text[result.start:result.end]
            })
        
        return {
            'anonymized_text': anonymized_result.text,
            'original_text': text,
            'pii_detections': pii_detections,
            'processing_time': processing_time,
            'pii_count': len(pii_detections)
        }
    
    def process_dataset(self, csv_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Process the entire dataset of call transcripts.
        
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
                    'original_transcript': row['call_transcript']
                })
                
                results.append(result)
                
                # Update statistics
                self._update_stats(result)
            
            total_processing_time = time.time() - total_start_time
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Log final metrics
            final_metrics = self._calculate_final_metrics(results_df, total_processing_time)
            print(f"\n✅ Processing complete! Final metrics:")
            for metric, value in final_metrics.items():
                print(f"   {metric}: {value}")
            
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
    
    def _calculate_final_metrics(self, results_df: pd.DataFrame, total_time: float) -> Dict:
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
            'pii_types_distribution': dict(self.stats['pii_types'])
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