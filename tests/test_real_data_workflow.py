"""
Demonstration using the actual synthetic_call_transcripts_voice_to_texts.csv data
to show how the three-stage workflow handles real PII variations.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from utils.text_normaliser import TextNormaliser
from evaluation.metrics import PIIEvaluator


class RealDataWorkflowDemo:
    """Demonstrates the three-stage workflow with real synthetic data."""
    
    def __init__(self):
        self.normalizer = TextNormaliser()
        self.evaluator = PIIEvaluator(matching_mode='business')
    
    def load_and_analyze_data(self):
        """Load the real data and analyze its structure."""
        df = pd.read_csv('../.data/synthetic_call_transcripts_voice_to_texts.csv')
        
        print("🔍 ACTUAL DATA STRUCTURE ANALYSIS")
        print("=" * 50)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Show the duplicate column issue
        print(f"\n📊 Column Issues Found:")
        column_counts = {}
        for col in df.columns:
            column_counts[col] = column_counts.get(col, 0) + 1
        
        for col, count in column_counts.items():
            if count > 1:
                print(f"  ❌ '{col}' appears {count} times (duplicate columns)")
        
        return df
    
    def demonstrate_current_structure_problems(self, df):
        """Show problems with current multi-column structure."""
        print(f"\n❌ CURRENT STRUCTURE PROBLEMS")
        print("=" * 50)
        
        # Show example of row 2 which has variations
        row2 = df.iloc[1]  # Row 2 (0-indexed)
        print(f"Example from call_id {row2['call_id']}:")
        print(f"  member_first_name columns: {row2['member_first_name']}")
        print(f"  member_mobile columns: {row2['member_mobile']}")
        print(f"  member_email columns: {row2['member_email']}")
        
        problems = [
            "• Duplicate column names make data access confusing",
            "• Hard to know which column contains which variation",
            "• Complex evaluation logic needed",
            "• Doesn't scale to new variation types"
        ]
        
        for problem in problems:
            print(problem)
    
    def convert_to_canonical_structure(self, df):
        """Convert the multi-column data to clean canonical structure."""
        print(f"\n✅ CONVERTING TO CANONICAL STRUCTURE")
        print("=" * 50)
        
        # Clean up the dataframe by keeping only canonical columns
        # Note: The CSV has duplicate column names, so pandas will auto-suffix them
        
        # Let's examine what pandas actually loaded
        actual_columns = df.columns.tolist()
        print(f"Pandas loaded columns: {actual_columns}")
        
        # Create clean structure with canonical values only
        clean_df = pd.DataFrame({
            'call_id': df['call_id'],
            'consultant_first_name': df['consultant_first_name'],
            'member_number': df['member_number'].astype(str),  # Take first occurrence
            'member_first_name': df.iloc[:, 2],  # First member_first_name column
            'member_full_name': df['member_full_name'],
            'member_mobile': df.iloc[:, 5],  # First member_mobile column  
            'member_email': df.iloc[:, 7],  # First member_email column
            'member_address': df['member_address'],
            'call_transcript': df['call_transcript']
        })
        
        print(f"✅ Clean structure created with canonical values only")
        print(f"New shape: {clean_df.shape}")
        
        return clean_df
    
    def demonstrate_three_stage_workflow(self, clean_df):
        """Show the three-stage workflow with real data."""
        print(f"\n🛡️ THREE-STAGE WORKFLOW DEMONSTRATION")
        print("=" * 50)
        
        # Take the first few rows for demonstration
        for idx in [1]:  # Row 2 which has interesting variations
            row = clean_df.iloc[idx]
            call_id = row['call_id']
            
            print(f"\n📝 Processing call_id: {call_id}")
            print("-" * 30)
            
            # Stage A: Original transcript
            stage_a = row['call_transcript']
            
            # Stage B: Normalized transcript  
            stage_b = self.normalizer.normalize_text(stage_a)
            
            # Stage C: Simulate cleaned transcript (PII removed)
            stage_c = self.simulate_pii_cleaning(stage_b, row)
            
            # Ground truth (canonical values)
            ground_truth = {
                'member_first_name': row['member_first_name'],
                'member_mobile': row['member_mobile'],
                'member_email': row['member_email'],
                'member_number': str(row['member_number'])
            }
            
            print(f"🎯 Ground Truth (canonical): {ground_truth}")
            print(f"\n(a) Original excerpt:")
            print(f"    {stage_a[:200]}...")
            print(f"\n(b) Normalized excerpt:")  
            print(f"    {stage_b[:200]}...")
            print(f"\n(c) Cleaned excerpt:")
            print(f"    {stage_c[:200]}...")
            
            # Evaluation: Check what gets found in stage B
            self.evaluate_stage_b(stage_b, ground_truth)
    
    def simulate_pii_cleaning(self, normalized_text, row):
        """Simulate PII removal for demonstration."""
        cleaned = normalized_text
        
        # Replace known PII with placeholders
        replacements = [
            (row['member_first_name'], '[PERSON]'),
            (row['member_full_name'], '[PERSON]'),  
            (row['member_mobile'], '[PHONE]'),
            (row['member_email'], '[EMAIL]'),
            (str(row['member_number']), '[MEMBER_ID]'),
            (row['member_address'], '[ADDRESS]')
        ]
        
        for original, placeholder in replacements:
            if pd.notna(original) and str(original).strip():
                cleaned = cleaned.replace(str(original), placeholder)
        
        return cleaned
    
    def evaluate_stage_b(self, normalized_text, ground_truth):
        """Evaluate PII detection in the normalized text."""
        print(f"\n📊 EVALUATION RESULTS:")
        
        total_found = 0
        total_expected = 0
        
        for pii_type, pii_value in ground_truth.items():
            if pii_value and str(pii_value).strip():
                total_expected += 1
                
                # Use enhanced evaluation method
                positions = self.evaluator._find_pii_positions(
                    str(pii_value), normalized_text, pii_type
                )
                
                if positions:
                    total_found += 1
                    status = "✅ FOUND"
                    for start, end in positions:
                        found_text = normalized_text[start:end]
                        print(f"  {pii_type}: '{pii_value}' → {status}")
                        print(f"    Found: '{found_text}' at position {start}-{end}")
                else:
                    status = "❌ MISSED"
                    print(f"  {pii_type}: '{pii_value}' → {status}")
        
        recall = total_found / total_expected if total_expected > 0 else 0
        print(f"\n🎯 Recall: {total_found}/{total_expected} = {recall:.1%}")


def run_real_data_demo():
    """Run the complete demonstration with real data."""
    
    print("🛡️ REAL DATA WORKFLOW DEMONSTRATION")
    print("Using: synthetic_call_transcripts_voice_to_texts.csv")
    print("=" * 60)
    
    demo = RealDataWorkflowDemo()
    
    # 1. Load and analyze the current structure
    df = demo.load_and_analyze_data()
    
    # 2. Show problems with current structure
    demo.demonstrate_current_structure_problems(df)
    
    # 3. Convert to canonical structure
    clean_df = demo.convert_to_canonical_structure(df)
    
    # 4. Demonstrate three-stage workflow
    demo.demonstrate_three_stage_workflow(clean_df)
    
    # 5. Show final recommendations
    print(f"\n" + "=" * 60)
    print("🌟 FINAL RECOMMENDATIONS FOR YOUR DATA")
    print("=" * 60)
    
    recommendations = [
        "✅ Convert to single canonical columns (member_first_name: 'Chloe')",
        "✅ Keep raw transcripts with all variations ('C h l o e', 'zero four...')",
        "✅ Use three-stage workflow: original → normalized → cleaned", 
        "✅ Run evaluation on normalized text vs canonical ground truth",
        "✅ All your existing TextNormaliser + PIIEvaluator code works perfectly!",
        "✅ Much cleaner than current duplicate column structure"
    ]
    
    for rec in recommendations:
        print(rec)
    
    print(f"\n💡 Your proposed approach is perfect for this data!")


if __name__ == "__main__":
    run_real_data_demo() 