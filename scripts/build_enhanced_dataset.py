#!/usr/bin/env python3
"""
Build enhanced tokenized dataset from enhanced text files with language tags
and content type tags using SentencePiece tokenizer.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import multiprocessing as mp
from datasets import Dataset, Features, Value, Sequence
import sentencepiece as spm
import tqdm
import re
import random

def load_tokenizer(tokenizer_path: Path) -> spm.SentencePieceProcessor:
    """Load SentencePiece tokenizer."""
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    
    sp = spm.SentencePieceProcessor()
    sp.load(str(tokenizer_path))
    return sp

def find_enhanced_files(data_dir: Path) -> List[Path]:
    """Find all enhanced language files."""
    files = list(data_dir.glob("*_enhanced.txt"))
    return sorted(files)

def extract_language_and_content(text: str) -> tuple[str, str, str]:
    """Extract language code, content type, and clean text from tagged text."""
    # Pattern: <lang> <content_type> text
    pattern = r'<([a-z]{2})>\s*(<[^>]+>)\s*(.*)'
    match = re.match(pattern, text)
    
    if match:
        lang_code = match.group(1)
        content_type = match.group(2)
        clean_text = match.group(3)
        return lang_code, content_type, clean_text
    else:
        # Fallback: try to extract just language
        lang_match = re.search(r'<([a-z]{2})>', text)
        if lang_match:
            return lang_match.group(1), "<general>", text
        else:
            return "unknown", "<general>", text

def process_file_chunk_enhanced(args: tuple) -> List[Dict[str, Any]]:
    """Process a chunk of lines from an enhanced file."""
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
                
                # Extract language and content type
                lang_code, content_type, clean_text = extract_language_and_content(line)
                
                # Tokenize the clean text
                input_ids = tokenizer.encode(clean_text)
                
                if len(input_ids) > max_length:
                    input_ids = input_ids[:max_length]
                
                if len(input_ids) < 10:
                    continue
                
                results.append({
                    "text": line,  # Keep original tagged text
                    "clean_text": clean_text,
                    "input_ids": input_ids,
                    "length": len(input_ids),
                    "language": lang_code,
                    "content_type": content_type,
                    "source_file": file_path.name
                })
                
    except Exception as e:
        print(f"Warning: Error processing {file_path}: {e}")
    
    return results

def balance_content_types(examples: List[Dict[str, Any]], target_distribution: Dict[str, float]) -> List[Dict[str, Any]]:
    """Balance examples by content type according to target distribution."""
    # Group by content type
    content_groups = {}
    for ex in examples:
        content_type = ex["content_type"]
        if content_type not in content_groups:
            content_groups[content_type] = []
        content_groups[content_type].append(ex)
    
    # Calculate target counts
    total_examples = len(examples)
    target_counts = {}
    for content_type, ratio in target_distribution.items():
        target_counts[f"<{content_type}>"] = int(total_examples * ratio)
    
    # Balance each content type
    balanced_examples = []
    for content_type, target_count in target_counts.items():
        if content_type in content_groups:
            group_examples = content_groups[content_type]
            if len(group_examples) > target_count:
                # Sample randomly if we have too many
                balanced_examples.extend(random.sample(group_examples, target_count))
            else:
                # Use all if we have too few
                balanced_examples.extend(group_examples)
        else:
            print(f"Warning: No examples found for content type {content_type}")
    
    # Add any remaining examples from content types not in target distribution
    for content_type, group_examples in content_groups.items():
        if content_type not in target_counts:
            balanced_examples.extend(group_examples)
    
    return balanced_examples

def build_enhanced_dataset(
    data_dir: Path,
    tokenizer_path: Path,
    output_dir: Path,
    max_length: int = 512,
    num_proc: int = 4,
    max_examples_per_file: int = None,
    balance_content: bool = True,
    target_distribution: Dict[str, float] = None
) -> None:
    """Build and save the enhanced tokenized dataset."""
    
    # Find input files
    input_files = find_enhanced_files(data_dir)
    if not input_files:
        print(f"Error: No enhanced files found in {data_dir}")
        print("Please run the enhanced data pipeline first:")
        print("python scripts/enhance_data_pipeline.py")
        sys.exit(1)
    
    print(f"Found {len(input_files)} enhanced files:")
    for f in input_files:
        print(f"  - {f.name}")
    
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)
    print(f"Loaded tokenizer with vocabulary size: {tokenizer.get_piece_size()}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process files
    print(f"\nProcessing files with {num_proc} workers...")
    if max_examples_per_file:
        print(f"Limiting to {max_examples_per_file} examples per file")
    
    # Prepare arguments for parallel processing
    chunk_args = [(f, tokenizer, max_length, max_examples_per_file) for f in input_files]
    
    # Process files in parallel
    all_results = []
    with mp.Pool(num_proc) as pool:
        for result in tqdm.tqdm(
            pool.imap(process_file_chunk_enhanced, chunk_args),
            total=len(input_files),
            desc="Processing files"
        ):
            all_results.extend(result)
    
    if not all_results:
        print("Error: No valid tokenized examples found")
        sys.exit(1)
    
    # Balance content types if requested
    if balance_content and target_distribution:
        print(f"\nBalancing content types...")
        all_results = balance_content_types(all_results, target_distribution)
    
    # Create dataset
    print(f"\nCreating dataset with {len(all_results):,} examples...")
    
    # Define features
    features = Features({
        "text": Value("string"),
        "clean_text": Value("string"),
        "input_ids": Sequence(Value("int32")),
        "length": Value("int32"),
        "language": Value("string"),
        "content_type": Value("string"),
        "source_file": Value("string")
    })
    
    dataset = Dataset.from_list(all_results, features=features)
    
    # Save dataset
    print(f"Saving dataset to {output_dir}...")
    dataset.save_to_disk(str(output_dir))
    
    # Print statistics
    print(f"\n✅ Enhanced tokenized dataset created successfully!")
    print(f"Total examples: {len(all_results):,}")
    
    # Language distribution
    lang_counts = {}
    for ex in all_results:
        lang = ex["language"]
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    
    print(f"\nLanguage distribution:")
    for lang, count in sorted(lang_counts.items()):
        print(f"  {lang}: {count:,}")
    
    # Content type distribution
    content_counts = {}
    for ex in all_results:
        content = ex["content_type"]
        content_counts[content] = content_counts.get(content, 0) + 1
    
    print(f"\nContent type distribution:")
    for content, count in sorted(content_counts.items()):
        print(f"  {content}: {count:,}")
    
    # Length statistics
    avg_length = sum(ex["length"] for ex in all_results) / len(all_results)
    max_len = max(ex["length"] for ex in all_results)
    min_len = min(ex["length"] for ex in all_results)
    
    print(f"\nLength statistics:")
    print(f"  Average sequence length: {avg_length:.1f}")
    print(f"  Min/Max sequence length: {min_len}/{max_len}")
    print(f"Output directory: {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Build enhanced tokenized dataset from enhanced text files"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/enhanced",
        help="Directory containing enhanced language files (default: data/enhanced)"
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
        default="data/enhanced/tokenized_dataset",
        help="Output directory for tokenized dataset (default: data/enhanced/tokenized_dataset)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
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
    parser.add_argument(
        "--balance_content",
        action="store_true",
        help="Balance content types according to target distribution"
    )
    parser.add_argument(
        "--target_distribution",
        type=str,
        default='{"dialogue": 0.15, "children": 0.10, "fiction": 0.10, "news": 0.15, "academic": 0.10, "general": 0.40}',
        help="Target content type distribution as JSON string"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    data_dir = Path(args.data_dir)
    tokenizer_path = Path(args.tokenizer_path)
    output_dir = Path(args.output_dir)
    
    # Parse target distribution
    import json
    target_distribution = json.loads(args.target_distribution) if args.balance_content else None
    
    # Validate inputs
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        print("Please run the enhanced data pipeline first:")
        print("python scripts/enhance_data_pipeline.py")
        sys.exit(1)
    
    if not tokenizer_path.exists():
        print(f"Error: Tokenizer file {tokenizer_path} does not exist")
        sys.exit(1)
    
    # Build dataset
    build_enhanced_dataset(
        data_dir=data_dir,
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
        max_length=args.max_length,
        num_proc=args.num_proc,
        max_examples_per_file=args.max_examples_per_file,
        balance_content=args.balance_content,
        target_distribution=target_distribution
    )

if __name__ == "__main__":
    main() 