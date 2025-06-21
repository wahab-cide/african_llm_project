#!/usr/bin/env python3
"""
Master script to run the complete African LLM enhancement pipeline.
This script automates the entire process from data enhancement to model training.
"""

import argparse
import sys
import subprocess
import time
from pathlib import Path
from typing import List, Dict

def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        elapsed = time.time() - start_time
        print(f"✅ Completed in {elapsed:.1f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed with exit code {e.returncode}")
        return False

def check_prerequisites() -> bool:
    """Check if prerequisites are met."""
    print("🔍 Checking prerequisites...")
    
    # Check if processed data exists
    processed_dir = Path("data/processed")
    if not processed_dir.exists():
        print("❌ data/processed/ directory not found")
        print("Please run the data preparation pipeline first:")
        print("python tokenization/prepare_data.py")
        return False
    
    # Check if tokenizer exists
    tokenizer_path = Path("tokenization/htf_bpe_16k.model")
    if not tokenizer_path.exists():
        print("❌ Tokenizer not found")
        print("Please train the tokenizer first:")
        print("python tokenization/train_tokenizer.py")
        return False
    
    print("✅ Prerequisites met")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Run complete African LLM enhancement pipeline"
    )
    parser.add_argument(
        "--skip-data-enhancement",
        action="store_true",
        help="Skip data enhancement step (use existing enhanced data)"
    )
    parser.add_argument(
        "--skip-tokenizer-training",
        action="store_true", 
        help="Skip language-specific tokenizer training"
    )
    parser.add_argument(
        "--skip-dataset-building",
        action="store_true",
        help="Skip enhanced dataset building"
    )
    parser.add_argument(
        "--skip-model-training",
        action="store_true",
        help="Skip model training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/enhanced.yaml",
        help="Training configuration file (default: training/configs/enhanced.yaml)"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        help="Maximum examples per file for dataset building"
    )
    
    args = parser.parse_args()
    
    print("🌍 AFRICAN LLM ENHANCEMENT PIPELINE")
    print("="*60)
    print("This script will run the complete enhancement pipeline:")
    print("1. Enhanced data processing with language tags")
    print("2. Language-specific tokenizer training (optional)")
    print("3. Enhanced dataset building")
    print("4. Enhanced model training")
    print("5. Demo testing")
    print("="*60)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Step 1: Enhanced Data Processing
    if not args.skip_data_enhancement:
        if not run_command(
            ["python", "scripts/enhance_data_pipeline.py"],
            "Step 1: Enhanced Data Processing"
        ):
            print("❌ Data enhancement failed. Stopping pipeline.")
            sys.exit(1)
    else:
        print("⏭️  Skipping data enhancement")
    
    # Step 2: Language-Specific Tokenizer Training (Optional)
    if not args.skip_tokenizer_training:
        if not run_command(
            ["python", "scripts/train_language_specific_tokenizers.py", "--evaluate"],
            "Step 2: Language-Specific Tokenizer Training"
        ):
            print("⚠️  Tokenizer training failed, but continuing with shared tokenizer")
    else:
        print("⏭️  Skipping tokenizer training")
    
    # Step 3: Enhanced Dataset Building
    if not args.skip_dataset_building:
        cmd = ["python", "scripts/build_enhanced_dataset.py", "--balance_content"]
        if args.max_examples:
            cmd.extend(["--max_examples_per_file", str(args.max_examples)])
        
        if not run_command(
            cmd,
            "Step 3: Enhanced Dataset Building"
        ):
            print("❌ Dataset building failed. Stopping pipeline.")
            sys.exit(1)
    else:
        print("⏭️  Skipping dataset building")
    
    # Step 4: Enhanced Model Training
    if not args.skip_model_training:
        if not run_command(
            ["python", "training/scripts/train.py", "--config", args.config],
            "Step 4: Enhanced Model Training"
        ):
            print("❌ Model training failed. Stopping pipeline.")
            sys.exit(1)
    else:
        print("⏭️  Skipping model training")
    
    # Step 5: Demo Testing
    print(f"\n{'='*60}")
    print("🎯 Step 5: Demo Testing")
    print(f"{'='*60}")
    
    # Check if model exists
    model_path = Path("outputs/models/enhanced-v1/final")
    if model_path.exists():
        print("✅ Enhanced model found!")
        print("\nYou can now test the enhanced model:")
        print("python deployment/enhanced_demo.py")
        print("\nOr run batch tests:")
        print("python deployment/enhanced_demo.py --language sw --content_type dialogue --prompts 'Hello' 'How are you?'")
    else:
        print("⚠️  Enhanced model not found. You may need to train it first.")
        print("python training/scripts/train.py --config training/configs/enhanced.yaml")
    
    print(f"\n{'='*60}")
    print("🎉 ENHANCEMENT PIPELINE COMPLETED!")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("1. Test the enhanced model with the demo")
    print("2. Evaluate generation quality")
    print("3. Fine-tune parameters if needed")
    print("4. Deploy the model")
    
    print("\nFor more information, see:")
    print("- docs/ENHANCEMENT_GUIDE.md")
    print("- docs/TRAINING.md")
    print("- README.md")

if __name__ == "__main__":
    main() 