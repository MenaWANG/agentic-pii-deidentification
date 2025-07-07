"""
Demonstration of the user's proposed workflow:
a) Original transcript → b) Normalized transcript → c) Cleaned transcript
With single canonical ground truth columns.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.text_normaliser import TextNormaliser
from evaluation.metrics import PIIEvaluator
import pandas as pd


class ProposedWorkflowDemo:
    """Demonstrates the user's proposed three-stage workflow."""
    
    def __init__(self):
        self.normalizer = TextNormaliser()
        self.evaluator = PIIEvaluator(matching_mode='business')
    
    def process_transcript(self, original_transcript: str, ground_truth: dict):
        """
        Process a single transcript through the three stages.
        
        Args:
            original_transcript: Stage (a) - raw transcript with variations
            ground_truth: Dictionary with canonical PII values
            
        Returns:
            Dictionary with all three stages and evaluation results
        """
        
        # Stage A: Original transcript (input)
        stage_a = original_transcript
        
        # Stage B: Normalized transcript (after TextNormaliser)
        stage_b = self.normalizer.normalize_text(stage_a)
        
        # Stage C: Cleaned transcript (after PII removal - simulated)
        # In real pipeline, this would come from Presidio/Agentic framework
        stage_c = self.simulate_pii_removal(stage_b, ground_truth)
        
        # Evaluation: Compare ground truth with what was found in stage B
        evaluation_results = self.evaluate_detection(stage_b, ground_truth)
        
        return {
            'stage_a_original': stage_a,
            'stage_b_normalized': stage_b,
            'stage_c_cleaned': stage_c,
            'ground_truth': ground_truth,
            'evaluation': evaluation_results
        }
    
    def simulate_pii_removal(self, normalized_text: str, ground_truth: dict):
        """Simulate PII removal for demonstration."""
        cleaned = normalized_text
        
        # Replace known PII with placeholders
        if ground_truth.get('member_first_name'):
            cleaned = cleaned.replace(ground_truth['member_first_name'], '[PERSON]')
        if ground_truth.get('member_mobile'):
            cleaned = cleaned.replace(ground_truth['member_mobile'], '[PHONE]')
        if ground_truth.get('member_email'):
            cleaned = cleaned.replace(ground_truth['member_email'], '[EMAIL]')
        
        return cleaned
    
    def evaluate_detection(self, normalized_text: str, ground_truth: dict):
        """Evaluate PII detection using existing evaluation functions."""
        results = {}
        
        for pii_type, pii_value in ground_truth.items():
            if pii_value:
                # Use existing _find_pii_positions method
                positions = self.evaluator._find_pii_positions(
                    pii_value, normalized_text, pii_type
                )
                results[pii_type] = {
                    'value': pii_value,
                    'found_positions': positions,
                    'detected': len(positions) > 0
                }
        
        return results


def demonstrate_workflow():
    """Run the complete workflow demonstration."""
    
    print("🛡️ Your Proposed Workflow - Three-Stage Processing")
    print("=" * 60)
    
    # Test data with variations
    test_cases = [
        {
            'call_id': 'test_001',
            'original_transcript': "Hi c h l o e, your number is zero four one two three four five six",
            'ground_truth': {
                'member_first_name': 'chloe',  # Canonical form
                'member_mobile': '041234556'   # Canonical form
            }
        },
        {
            'call_id': 'test_002', 
            'original_transcript': "Hello j o h n at company dot com, mobile zero four eight seven",
            'ground_truth': {
                'member_first_name': 'john',
                'member_email': 'john@company.com',
                'member_mobile': '0487'
            }
        }
    ]
    
    demo = ProposedWorkflowDemo()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {test_case['call_id']}")
        print("-" * 40)
        
        result = demo.process_transcript(
            test_case['original_transcript'],
            test_case['ground_truth']
        )
        
        # Display the three stages
        print(f"(a) Original:    {result['stage_a_original']}")
        print(f"(b) Normalized:  {result['stage_b_normalized']}")  
        print(f"(c) Cleaned:     {result['stage_c_cleaned']}")
        
        print(f"\n🎯 Ground Truth: {result['ground_truth']}")
        
        print(f"\n📊 Evaluation Results:")
        for pii_type, eval_result in result['evaluation'].items():
            status = "✅ FOUND" if eval_result['detected'] else "❌ MISSED"
            print(f"  {pii_type}: '{eval_result['value']}' → {status}")
            if eval_result['found_positions']:
                for start, end in eval_result['found_positions']:
                    found_text = result['stage_b_normalized'][start:end]
                    print(f"    Found: '{found_text}' at position {start}-{end}")


def show_benefits():
    """Show the benefits of this approach."""
    
    print("\n" + "=" * 60)
    print("🌟 BENEFITS OF YOUR PROPOSED APPROACH")
    print("=" * 60)
    
    benefits = [
        "✅ Single canonical ground truth columns",
        "✅ Clear three-stage pipeline (a → b → c)",
        "✅ All existing functions work without modification",
        "✅ TextNormaliser handles variations automatically",
        "✅ PIIEvaluator works on normalized text (stage b)",
        "✅ Easy to debug each transformation step",
        "✅ Framework-compatible design",
        "✅ No sparse data or multiple variant columns",
        "✅ Scales well to new PII variations"
    ]
    
    for benefit in benefits:
        print(benefit)
    
    print("\n💡 This approach perfectly leverages your existing architecture!")


def show_implementation_steps():
    """Show how to implement this in the actual framework."""
    
    print("\n" + "=" * 60)
    print("🔧 IMPLEMENTATION IN YOUR FRAMEWORK")
    print("=" * 60)
    
    steps = [
        "1. Keep current ground truth structure (canonical values only)",
        "2. In your pipeline, save three versions:",
        "   - original_transcript (stage a)", 
        "   - normalized_transcript (stage b = TextNormaliser output)",
        "   - cleaned_transcript (stage c = framework output)",
        "3. Run evaluation on stage b (normalized) vs ground truth",
        "4. Use stage c for final output/production use",
        "5. All existing evaluation code works perfectly!"
    ]
    
    for step in steps:
        print(step)


if __name__ == "__main__":
    demonstrate_workflow()
    show_benefits() 
    show_implementation_steps() 