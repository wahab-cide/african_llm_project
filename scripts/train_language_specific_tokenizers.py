#!/usr/bin/env python3
"""
Train language-specific SentencePiece tokenizers for better performance
on individual African languages.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict
import sentencepiece as spm
import tempfile
import shutil
from tqdm import tqdm

# Language configurations
LANG_CONFIGS = {
    "am": {
        "name": "Amharic",
        "vocab_size": 8000,
        "character_coverage": 0.9995,
        "model_type": "bpe"
    },
    "ff": {
        "name": "Fulani", 
        "vocab_size": 6000,
        "character_coverage": 0.9995,
        "model_type": "bpe"
    },
    "ha": {
        "name": "Hausa",
        "vocab_size": 8000,
        "character_coverage": 0.9995,
        "model_type": "bpe"
    },
    "so": {
        "name": "Somali",
        "vocab_size": 8000,
        "character_coverage": 0.9995,
        "model_type": "bpe"
    },
    "sw": {
        "name": "Swahili",
        "vocab_size": 8000,
        "character_coverage": 0.9995,
        "model_type": "bpe"
    },
    "yo": {
        "name": "Yoruba",
        "vocab_size": 8000,
        "character_coverage": 0.9995,
        "model_type": "bpe"
    }
}

def find_language_files(data_dir: Path, lang_code: str) -> List[Path]:
    """Find all files for a specific language."""
    files = []
    
    # Look for language-specific files
    patterns = [
        f"*{lang_code}*.txt",
        f"{lang_code}_*.txt", 
        f"*_{lang_code}.txt",
        f"{lang_code}.txt"
    ]
    
    for pattern in patterns:
        files.extend(data_dir.glob(pattern))
    
    # Also check subdirectories
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            for pattern in patterns:
                files.extend(subdir.glob(pattern))
    
    return sorted(list(set(files)))

def stream_files_to_temp(files: List[Path], temp_file: Path) -> int:
    """Stream all files line by line to a temporary file for training."""
    total_lines = 0
    
    with temp_file.open('w', encoding='utf-8') as out_f:
        for file_path in tqdm(files, desc="Reading files"):
            try:
                with file_path.open('r', encoding='utf-8', errors='replace') as in_f:
                    for line in in_f:
                        line = line.strip()
                        if line and len(line) > 10:  # Skip very short lines
                            out_f.write(line + '\n')
                            total_lines += 1
            except Exception as e:
                print(f"Warning: Error reading {file_path}: {e}")
                continue
    
    return total_lines

def train_language_tokenizer(
    input_file: Path,
    output_prefix: str,
    config: Dict
) -> None:
    """Train a SentencePiece tokenizer for a specific language."""
    
    # Prepare training arguments
    train_args = [
        f"--input={input_file}",
        f"--model_prefix={output_prefix}",
        f"--model_type={config['model_type']}",
        f"--vocab_size={config['vocab_size']}",
        f"--character_coverage={config['character_coverage']}",
        "--user_defined_symbols=<bos>,<eos>,<pad>,<sep>",
        "--unk_id=0",
        "--bos_id=1", 
        "--eos_id=2",
        "--pad_id=3",
        "--hard_vocab_limit=false",
        "--split_by_unicode_script=true",
        "--split_by_number=true",
        "--split_by_whitespace=true",
        "--treat_whitespace_as_suffix=false",
        "--allow_whitespace_only_pieces=true",
        "--split_digits=true",
        "--byte_fallback=true",
        "--unk_piece=<unk>",
        "--bos_piece=<bos>",
        "--eos_piece=<eos>",
        "--pad_piece=<pad>",
        "--train_extremely_large_corpus=false",
        "--shuffle_input_sentence=true",
        "--input_sentence_size=10000000",
        "--max_sentencepiece_length=16",
        "--minloglevel=1"
    ]
    
    print(f"Training {config['name']} tokenizer...")
    print(f"  Vocabulary size: {config['vocab_size']}")
    print(f"  Model type: {config['model_type']}")
    print(f"  Character coverage: {config['character_coverage']}")
    
    # Train the tokenizer
    spm.SentencePieceTrainer.train(" ".join(train_args))
    
    print(f"✅ {config['name']} tokenizer trained successfully!")

def evaluate_tokenizer(model_path: Path, test_file: Path) -> Dict:
    """Evaluate tokenizer performance on test data."""
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    
    total_tokens = 0
    total_chars = 0
    compression_ratio = 0
    
    with test_file.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                tokens = sp.encode_as_ids(line)
                total_tokens += len(tokens)
                total_chars += len(line)
    
    if total_chars > 0:
        compression_ratio = total_tokens / total_chars
    
    return {
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "compression_ratio": compression_ratio,
        "vocab_size": sp.get_piece_size()
    }

def main():
    parser = argparse.ArgumentParser(
        description="Train language-specific SentencePiece tokenizers"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/enhanced",
        help="Directory containing enhanced language data (default: data/enhanced)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        default="tokenization/language_specific",
        help="Output directory for language-specific tokenizers (default: tokenization/language_specific)"
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=list(LANG_CONFIGS.keys()),
        help="Languages to train tokenizers for (default: all)"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate tokenizer performance after training"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Validate inputs
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train tokenizers for each language
    results = {}
    
    for lang_code in args.languages:
        if lang_code not in LANG_CONFIGS:
            print(f"Warning: Unknown language code {lang_code}, skipping")
            continue
        
        config = LANG_CONFIGS[lang_code]
        print(f"\n{'='*50}")
        print(f"Processing {config['name']} ({lang_code})")
        print(f"{'='*50}")
        
        # Find language files
        lang_files = find_language_files(data_dir, lang_code)
        
        if not lang_files:
            print(f"Warning: No files found for {lang_code}, skipping")
            continue
        
        print(f"Found {len(lang_files)} files:")
        for f in lang_files:
            print(f"  - {f}")
        
        # Create temporary file for training
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_f:
            temp_file = Path(temp_f.name)
        
        try:
            # Stream files to temp
            total_lines = stream_files_to_temp(lang_files, temp_file)
            print(f"Total lines for training: {total_lines:,}")
            
            if total_lines < 1000:
                print(f"Warning: Very few lines ({total_lines}) for {lang_code}")
            
            # Train tokenizer
            output_prefix = output_dir / f"{lang_code}_tokenizer"
            train_language_tokenizer(temp_file, str(output_prefix), config)
            
            # Evaluate if requested
            if args.evaluate:
                print(f"Evaluating {lang_code} tokenizer...")
                eval_results = evaluate_tokenizer(
                    output_prefix.with_suffix('.model'),
                    lang_files[0]  # Use first file for evaluation
                )
                results[lang_code] = eval_results
                
                print(f"  Vocabulary size: {eval_results['vocab_size']:,}")
                print(f"  Compression ratio: {eval_results['compression_ratio']:.3f}")
                print(f"  Total tokens: {eval_results['total_tokens']:,}")
                print(f"  Total characters: {eval_results['total_chars']:,}")
        
        finally:
            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()
    
    # Print summary
    print(f"\n{'='*50}")
    print("TRAINING SUMMARY")
    print(f"{'='*50}")
    
    for lang_code in args.languages:
        if lang_code in LANG_CONFIGS:
            config = LANG_CONFIGS[lang_code]
            model_path = output_dir / f"{lang_code}_tokenizer.model"
            
            if model_path.exists():
                print(f"✅ {config['name']} ({lang_code}): {model_path}")
                if lang_code in results:
                    ratio = results[lang_code]['compression_ratio']
                    print(f"   Compression ratio: {ratio:.3f}")
            else:
                print(f"❌ {config['name']} ({lang_code}): Failed")
    
    print(f"\nTokenizers saved to: {output_dir}")

if __name__ == "__main__":
    main() 