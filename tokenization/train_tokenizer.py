"""
Train a shared SentencePiece tokenizer for African language corpus.
"""

import argparse
import os
import sys
from pathlib import Path
import sentencepiece as spm


def find_input_files(input_glob: str) -> list[Path]:
    """Find all .txt files matching the input glob pattern, relative to project root."""
    # Always resolve relative to the script's parent directory (project root)
    project_root = Path(__file__).parent.parent.resolve()
    files = list(project_root.glob(input_glob))
    txt_files = [f for f in files if f.suffix == '.txt' and f.is_file()]
    return txt_files


def stream_files_to_temp(files: list[Path], temp_file: Path) -> int:
    """Stream all files line by line to a temporary file for training."""
    total_lines = 0
    
    with temp_file.open('w', encoding='utf-8') as out_f:
        for file_path in files:
            print(f"Reading {file_path}...")
            try:
                with file_path.open('r', encoding='utf-8', errors='replace') as in_f:
                    for line in in_f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            out_f.write(line + '\n')
                            total_lines += 1
            except Exception as e:
                print(f"Warning: Error reading {file_path}: {e}")
                continue
    
    return total_lines


def train_tokenizer(
    input_file: Path,
    output_prefix: str,
    vocab_size: int,
    model_type: str = "bpe",
    character_coverage: float = 1.0
) -> None:
    """Train the SentencePiece tokenizer."""
    
    # Prepare training arguments
    train_args = [
        f"--input={input_file}",
        f"--model_prefix={output_prefix}",
        f"--model_type={model_type}",
        f"--vocab_size={vocab_size}",
        f"--character_coverage={character_coverage}",
        "--user_defined_symbols=<bos>,<eos>,<pad>",
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
    
    print(f"Training tokenizer with {vocab_size} vocabulary size...")
    print(f"Model type: {model_type}")
    print(f"Character coverage: {character_coverage}")
    
    # Train the tokenizer
    spm.SentencePieceTrainer.train(" ".join(train_args))


def print_vocab_preview(vocab_file: Path, num_tokens: int = 10) -> None:
    """Print a preview of the vocabulary."""
    print(f"\nVocabulary preview (first {num_tokens} tokens):")
    print("-" * 50)
    
    try:
        with vocab_file.open('r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_tokens:
                    break
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    token, score = parts[0], parts[1]
                    print(f"{i:2d}: {token!r} (score: {score})")
                else:
                    print(f"{i:2d}: {line.strip()}")
    except Exception as e:
        print(f"Warning: Could not read vocabulary file: {e}")


def get_file_size(file_path: Path) -> str:
    """Get human-readable file size."""
    try:
        size_bytes = file_path.stat().st_size
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
    except:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Train a shared SentencePiece tokenizer for African language corpus"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=16003,
        help="Vocabulary size (default: 16003)"
    )
    parser.add_argument(
        "--input_glob",
        type=str,
        default="data/processed/*.txt",
        help="Input file glob pattern (default: data/processed/*.txt)"
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="tokenization/htf_bpe_16k",
        help="Output prefix for model and vocab files (default: tokenization/htf_bpe_16k)"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="bpe",
        choices=["bpe", "unigram", "char", "word"],
        help="Model type (default: bpe)"
    )
    parser.add_argument(
        "--character_coverage",
        type=float,
        default=1.0,
        help="Character coverage (default: 1.0)"
    )
    
    args = parser.parse_args()
    
   
    output_dir = Path(args.out_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
  
    print(f"Looking for files matching: {args.input_glob}")
    input_files = find_input_files(args.input_glob)
    
    if not input_files:
        print(f"Error: No .txt files found matching pattern '{args.input_glob}'")
        print("Please ensure you have processed text files in the data/processed/ directory.")
        sys.exit(1)
    
    print(f"Found {len(input_files)} input files:")
    for f in input_files:
        print(f"  - {f}")
    
 
    temp_file = Path("temp_training_data.txt")
    
    try:
        
        print(f"\nStreaming {len(input_files)} files to temporary training file...")
        total_lines = stream_files_to_temp(input_files, temp_file)
        
        if total_lines == 0:
            print("Error: No valid text lines found in input files")
            sys.exit(1)
        
        print(f"Total lines for training: {total_lines:,}")
        
      
        train_tokenizer(
            input_file=temp_file,
            output_prefix=args.out_prefix,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            character_coverage=args.character_coverage
        )
        
      
        model_file = Path(f"{args.out_prefix}.model")
        vocab_file = Path(f"{args.out_prefix}.vocab")
        
        if not model_file.exists() or not vocab_file.exists():
            print("Error: Training failed - output files not created")
            sys.exit(1)
        
      
        print(f"\n Tokenizer training completed successfully!")
        print(f"Model file: {model_file} ({get_file_size(model_file)})")
        print(f"Vocab file: {vocab_file} ({get_file_size(vocab_file)})")
        
       
        print_vocab_preview(vocab_file)
        
        print(f"\nTokenizer is ready for use!")
        print(f"Model: {model_file}")
        print(f"Vocabulary: {vocab_file}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)
    
    finally:
       
        if temp_file.exists():
            temp_file.unlink()
            print(f"Cleaned up temporary file: {temp_file}")


if __name__ == "__main__":
    main() 