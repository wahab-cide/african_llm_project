from pathlib import Path
import re, unicodedata, ftfy, tqdm
import os

_RE_QUOTES = re.compile(r"[\"""]")
_RE_WS = re.compile(r"\s+")

def clean_text(text: str) -> str | None:
    """Clean text without language detection for better performance"""
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFC", text)
    text = _RE_QUOTES.sub(r"", text)
    text = _RE_WS.sub(" ", text).strip()
    
    if len(text) < 10:
        return None
    
    # Remove lines with >=30% digits
    digit_ratio = sum(1 for c in text if c.isdigit()) / len(text)
    if digit_ratio >= 0.3:
        return None
    
    # Remove lines with 'ም.' or '.com' (repeated or present)
    if 'ም.' in text or '.com' in text:
        return None
    
    # Skip lines that are mostly numbers or symbols
    alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
    if alpha_ratio < 0.3:
        return None
    
    return text if text else None

def process_files(in_glob: str, out_file: Path, lang_code: str) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    
    # Get the project root directory (parent of tokenization folder)
    project_root = Path(__file__).parent.parent
    
    # Find all matching files first
    matching_files = list(project_root.glob(in_glob))
    print(f"Found {len(matching_files)} files for {lang_code}")
    
    if not matching_files:
        print(f"No files found for {lang_code} with pattern: {in_glob}")
        return
    
    total_lines = 0
    processed_lines = 0
    
    with out_file.open("w", encoding="utf-8") as f_out:
        for fp in tqdm.tqdm(matching_files, desc=f"Processing {lang_code} files"):
            try:
                with fp.open("r", encoding="utf-8", errors="replace") as f_in:
                    for line_num, line in enumerate(f_in, 1):
                        total_lines += 1
                        line = line.strip()
                        if not line or line in seen:
                            continue
                        
                        cleaned = clean_text(line)
                        if cleaned:
                            seen.add(cleaned)
                            f_out.write(cleaned + "\n")
                            processed_lines += 1
                        
                        # Progress update every 1000 lines
                        if total_lines % 1000 == 0:
                            print(f"  {lang_code}: {total_lines} lines read, {processed_lines} lines written")
                            
            except Exception as e:
                print(f"Error processing {fp}: {e}")
                continue
    
    print(f"{lang_code}: Processed {total_lines} total lines, wrote {processed_lines} unique cleaned lines")

if __name__ == "__main__":
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Process each language
    languages = [
        ("data/raw/hausa/**/*.txt", "data/processed/hausa.txt", "ha"),
        ("data/raw/swahili/**/*.txt", "data/processed/swahili.txt", "sw"),
        ("data/raw/fulani/**/*.txt", "data/processed/fulani.txt", "ff"),
        ("data/raw/amharic/**/*.txt", "data/processed/amharic.txt", "am"),
        ("data/raw/somali/**/*.txt", "data/processed/somali.txt", "so"),
        ("data/raw/yoruba/**/*.txt", "data/processed/yoruba.txt", "yo"),
    ]
    
    token_counts = {}
    for in_glob, out_path, lang_code in languages:
        print(f"\nProcessing {lang_code}...")
        process_files(in_glob, project_root / out_path, lang_code)
        # Count tokens in output file
        out_file = project_root / out_path
        if out_file.exists():
            with out_file.open("r", encoding="utf-8") as f:
                tokens = sum(len(line.split()) for line in f)
            token_counts[lang_code] = tokens
            print(f"{lang_code}: {tokens:,} tokens")
        else:
            token_counts[lang_code] = 0
    
    # Print summary and check balance
    print("\nToken counts per language:")
    for lang, count in token_counts.items():
        print(f"  {lang}: {count:,}")
    if token_counts:
        min_tokens = min(token_counts.values())
        for lang, count in token_counts.items():
            if min_tokens > 0 and count > 3 * min_tokens:
                print(f"Warning: {lang} has more than 3x the tokens of the smallest language. Stopping.")
                break
                  
                  