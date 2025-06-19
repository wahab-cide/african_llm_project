#!/usr/bin/env python3
"""
Build tokenized dataset from processed text files using SentencePiece tokenizer.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
import multiprocessing as mp
from datasets import Dataset, Features, Value, Sequence
import sentencepiece as spm
import tqdm


def find_processed_files(data_dir: Path) -> List[Path]:
    """Find all .txt files in the processed data directory."""
    files = list(data_dir.glob("*.txt"))
    txt_files = [f for f in files if f.suffix == '.txt' and f.is_file()]
    return sorted(txt_files)


def load_tokenizer(model_path: Path) -> spm.SentencePieceProcessor:
    """Load the trained SentencePiece tokenizer."""
    try:
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(str(model_path))
        print(f"Loaded tokenizer from {model_path}")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)


def process_file_chunk(args: tuple) -> List[Dict[str, Any]]:
    """Process a chunk of lines from a file."""
    file_path, tokenizer, max_length, max_examples_per_file = args
    results = []
    
    try:
        with file_path.open('r', encoding='utf-8', errors='replace') as f:
            for line_num, line in enumerate(f):
                # Limit examples per file if specified
                if max_examples_per_file and line_num >= max_examples_per_file:
                    break
                    
                line = line.strip()
                if not line:
                    continue
                
                # Tokenize the line
                input_ids = tokenizer.encode(line)
                
                if len(input_ids) > max_length:
                    input_ids = input_ids[:max_length]
                
                if len(input_ids) < 10:
                    continue
                
                results.append({
                    "text": line,
                    "input_ids": input_ids,
                    "length": len(input_ids),
                    "source_file": file_path.name
                })
                
    except Exception as e:
        print(f"Warning: Error processing {file_path}: {e}")
    
    return results


def build_tokenized_dataset(
    data_dir: Path,
    tokenizer_path: Path,
    output_dir: Path,
    max_length: int = 256,
    num_proc: int = 4,
    max_examples_per_file: int = None
) -> None:
    """Build and save the tokenized dataset."""
    
    # Find input files
    input_files = find_processed_files(data_dir)
    if not input_files:
        print(f"Error: No .txt files found in {data_dir}")
        sys.exit(1)
    
    print(f"Found {len(input_files)} input files:")
    for f in input_files:
        print(f"  - {f.name}")
    

    tokenizer = load_tokenizer(tokenizer_path)
    

    output_dir.mkdir(parents=True, exist_ok=True)
    

    print(f"\nProcessing files with {num_proc} workers...")
    if max_examples_per_file:
        print(f"Limiting to {max_examples_per_file} examples per file")
    
  
    chunk_args = [(f, tokenizer, max_length, max_examples_per_file) for f in input_files]
    
 
    all_results = []
    with mp.Pool(num_proc) as pool:
        for result in tqdm.tqdm(
            pool.imap(process_file_chunk, chunk_args),
            total=len(input_files),
            desc="Processing files"
        ):
            all_results.extend(result)
    
    if not all_results:
        print("Error: No valid tokenized examples found")
        sys.exit(1)
    
    # Create dataset
    print(f"\nCreating dataset with {len(all_results):,} examples...")
    
    # Define features
    features = Features({
        "text": Value("string"),
        "input_ids": Sequence(Value("int32")),
        "length": Value("int32"),
        "source_file": Value("string")
    })
    
    dataset = Dataset.from_list(all_results, features=features)
    
    # Save dataset
    print(f"Saving dataset to {output_dir}...")
    dataset.save_to_disk(str(output_dir))
    
  
    avg_length = sum(ex["length"] for ex in all_results) / len(all_results)
    max_len = max(ex["length"] for ex in all_results)
    min_len = min(ex["length"] for ex in all_results)
    
    print(f"\n✅ Tokenized dataset created successfully!")
    print(f"Total examples: {len(all_results):,}")
    print(f"Average sequence length: {avg_length:.1f}")
    print(f"Min/Max sequence length: {min_len}/{max_len}")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Build tokenized dataset from processed text files"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing processed .txt files (default: data/processed)"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="tokenization/htf_bpe_16k.model",
        help="Path to SentencePiece tokenizer model (default: tokenization/htf_bpe_16k.model)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/tokenized_dataset",
        help="Output directory for tokenized dataset (default: data/processed/tokenized_dataset)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length (default: 256)"
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=4,
        help="Number of processes for parallel processing (default: 4)"
    )
    parser.add_argument(
        "--max_examples_per_file",
        type=int,
        help="Maximum number of examples per file (optional)"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    data_dir = Path(args.data_dir)
    tokenizer_path = Path(args.tokenizer_path)
    output_dir = Path(args.output_dir)
    
    # Validate inputs
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        sys.exit(1)
    
    if not tokenizer_path.exists():
        print(f"Error: Tokenizer file {tokenizer_path} does not exist")
        sys.exit(1)
    
    # Build dataset
    build_tokenized_dataset(
        data_dir=data_dir,
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
        max_length=args.max_length,
        num_proc=args.num_proc,
        max_examples_per_file=args.max_examples_per_file
    )


if __name__ == "__main__":
    main() 