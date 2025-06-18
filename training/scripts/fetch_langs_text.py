#!/usr/bin/env python3
"""
Download Somali and Fulani language corpora for language model training.

This script downloads text from multiple sources and saves them
under data/raw/somali/ and data/raw/fulani/ with proper directory structure.

Usage:
    python fetch_langs_text.py                    # Download all with default limits
    python fetch_langs_text.py --max-lines 100000 # Limit each corpus to 100k lines
    python fetch_langs_text.py --skip-existing    # Skip files that already exist
"""

import sys
import argparse
from pathlib import Path
from datasets import load_dataset, disable_caching

# Disable caching to avoid large ~/.cache folders
disable_caching()

# Configuration
LANGUAGES = {
    "somali": {
        "code": "so",
        "base_dir": Path("data/raw/somali"),
        "name": "Somali"
    },
    "fulani": {
        "code": "ff", 
        "base_dir": Path("data/raw/fulani"),
        "name": "Fulani"
    }
}


def get_file_stats(file_path: Path) -> tuple[int, str]:
    """Get line count and human-readable file size."""
    if not file_path.exists():
        return 0, "0 B"
    
    # Count lines
    with file_path.open("r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)
    
    # Get file size
    size_bytes = file_path.stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            size_str = f"{size_bytes:.0f} {unit}"
            break
        size_bytes /= 1024
    else:
        size_str = f"{size_bytes:.1f} TB"
    
    return line_count, size_str


def download_cc100(lang_code: str, base_dir: Path, lang_name: str, max_lines: int = None) -> bool:
    """Download CC-100 corpus for the given language."""
    output_file = base_dir / "cc100" / f"{lang_code}.txt"
    
    # Skip if file already exists
    if output_file.exists():
        lines, size = get_file_stats(output_file)
        print(f"⏭️  CC-100 {lang_name} already exists: {output_file} ({size}, {lines:,} lines)")
        return True
    
    try:
        # Create directory
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load dataset with streaming
        print(f"⬇️  Downloading CC-100 {lang_name}...")
        dataset = load_dataset("cc100", lang_code, split="train", streaming=True, trust_remote_code=True)
        
        # Write to file
        line_count = 0
        with output_file.open("w", encoding="utf-8") as f:
            for row in dataset:
                if row.get("text"):
                    f.write(row["text"].strip() + "\n")
                    line_count += 1
                
                # Check if we've reached the limit
                if max_lines and line_count >= max_lines:
                    print(f"   Reached limit of {max_lines:,} lines, stopping...")
                    break
                
                # Print progress every 10,000 lines
                if line_count % 10000 == 0:
                    print(f"   Processed {line_count:,} lines...")
        
        # Get final stats and print success
        _, size = get_file_stats(output_file)
        print(f"✅ CC-100 {lang_name} saved to {output_file} ({size}, {line_count:,} lines)")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download CC-100 {lang_name}: {e}")
        return False


def download_jw300(lang_code: str, base_dir: Path, lang_name: str, max_lines: int = None) -> bool:
    """Download JW300 corpus for the given language (language side only)."""
    output_file = base_dir / "extra" / f"jw300_{lang_code}.txt"
    
    # Skip if file already exists
    if output_file.exists():
        lines, size = get_file_stats(output_file)
        print(f"⏭️  JW300 {lang_name} already exists: {output_file} ({size}, {lines:,} lines)")
        return True
    
    try:
        # Create directory
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load dataset - try both directions
        print(f"⬇️  Downloading JW300 {lang_name}...")
        try:
            # Try en-{lang} first
            dataset = load_dataset("opus100", f"en-{lang_code}", split="train")
        except:
            # Fallback to {lang}-en
            dataset = load_dataset("opus100", f"{lang_code}-en", split="train")
        
        # Write to file
        line_count = 0
        with output_file.open("w", encoding="utf-8") as f:
            for row in dataset:
                if row.get("translation") and row["translation"].get(lang_code):
                    f.write(row["translation"][lang_code].strip() + "\n")
                    line_count += 1
                
                # Check if we've reached the limit
                if max_lines and line_count >= max_lines:
                    print(f"   Reached limit of {max_lines:,} lines, stopping...")
                    break
        
        # Get final stats and print success
        _, size = get_file_stats(output_file)
        print(f"✅ JW300 {lang_name} saved to {output_file} ({size}, {line_count:,} lines)")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download JW300 {lang_name}: {e}")
        return False


def download_oscar(lang_code: str, base_dir: Path, lang_name: str, max_lines: int = None) -> bool:
    """Download OSCAR corpus for the given language if available."""
    output_file = base_dir / "oscar" / f"oscar_{lang_code}.txt"
    
    # Skip if file already exists
    if output_file.exists():
        lines, size = get_file_stats(output_file)
        print(f"⏭️  OSCAR {lang_name} already exists: {output_file} ({size}, {lines:,} lines)")
        return True
    
    try:
        # Create directory
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load dataset with streaming
        print(f"⬇️  Downloading OSCAR {lang_name}...")
        config = f"unshuffled_deduplicated_{lang_code}"
        dataset = load_dataset("oscar", config, split="train", streaming=True)
        
        # Write to file
        line_count = 0
        with output_file.open("w", encoding="utf-8") as f:
            for row in dataset:
                if row.get("text"):
                    f.write(row["text"].strip() + "\n")
                    line_count += 1
                
                # Check if we've reached the limit
                if max_lines and line_count >= max_lines:
                    print(f"   Reached limit of {max_lines:,} lines, stopping...")
                    break
                
                # Print progress every 10,000 lines
                if line_count % 10000 == 0:
                    print(f"   Processed {line_count:,} lines...")
        
        # Get final stats and print success
        _, size = get_file_stats(output_file)
        print(f"✅ OSCAR {lang_name} saved to {output_file} ({size}, {line_count:,} lines)")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download OSCAR {lang_name}: {e}")
        return False


def download_alffa(base_dir: Path, max_lines: int = None) -> bool:
    """Download ALFFA speech corpus for Fulani."""
    output_file = base_dir / "extra" / "ALFFA_ff.txt"
    
    # Skip if file already exists
    if output_file.exists():
        lines, size = get_file_stats(output_file)
        print(f"⏭️  ALFFA Fulani already exists: {output_file} ({size}, {lines:,} lines)")
        return True
    
    try:
        # Create directory
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        print("⬇️  Downloading ALFFA Fulani...")
        dataset = load_dataset("afrispeech/alffa", "ff", split="train")
        
        # Write to file
        line_count = 0
        with output_file.open("w", encoding="utf-8") as f:
            for row in dataset:
                if row.get("sentence"):
                    f.write(row["sentence"].strip() + "\n")
                    line_count += 1
                
                # Check if we've reached the limit
                if max_lines and line_count >= max_lines:
                    print(f"   Reached limit of {max_lines:,} lines, stopping...")
                    break
        
        # Get final stats and print success
        _, size = get_file_stats(output_file)
        print(f"✅ ALFFA Fulani saved to {output_file} ({size}, {line_count:,} lines)")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download ALFFA Fulani: {e}")
        return False


def download_language(lang_name: str, lang_info: dict, max_lines: int = None) -> bool:
    """Download all available corpora for a specific language."""
    lang_code = lang_info["code"]
    base_dir = lang_info["base_dir"]
    display_name = lang_info["name"]
    
    print(f"\n🌍 Downloading {display_name} ({lang_code})...")
    print("=" * 50)
    
    success = True
    
    # Try CC-100
    if not download_cc100(lang_code, base_dir, display_name, max_lines):
        success = False
    
    # Try JW300
    if not download_jw300(lang_code, base_dir, display_name, max_lines):
        success = False
    
    # Try OSCAR
    if not download_oscar(lang_code, base_dir, display_name, max_lines):
        print(f"   (OSCAR {display_name} not available - this may be expected)")
    
    # Special handling for Fulani (ALFFA)
    if lang_code == "ff":
        if not download_alffa(base_dir, max_lines):
            print("   (ALFFA Fulani may not be available)")
    
    return success


def main():
    """Main function to download all language corpora."""
    parser = argparse.ArgumentParser(
        description="Download Somali and Fulani language corpora",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fetch_langs_text.py                    # Download all with default limits
  python fetch_langs_text.py --max-lines 100000 # Limit each corpus to 100k lines
  python fetch_langs_text.py --skip-existing    # Skip downloading if files already exist
        """
    )
    
    parser.add_argument(
        "--max-lines",
        type=int,
        help="Maximum number of lines to download per corpus (default: no limit)"
    )
    
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip downloading if files already exist"
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting Somali and Fulani corpus download...")
    if args.max_lines:
        print(f"📊 Limiting each corpus to {args.max_lines:,} lines")
    if args.skip_existing:
        print("⏭️  Skipping existing files")
    print("=" * 60)
    
    # Download all available corpora for each language
    all_success = True
    
    for lang_name, lang_info in LANGUAGES.items():
        if not download_language(lang_name, lang_info, args.max_lines):
            all_success = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_success:
        print("✅ All available downloads completed successfully!")
        for lang_info in LANGUAGES.values():
            print(f"📁 {lang_info['name']} files saved under: {lang_info['base_dir']}")
    else:
        print("⚠️  Some downloads failed.")
        print("💡 CC-100 and JW300 should work for Somali (so).")
        print("💡 CC-100 should work for Fulani (ff).")
        print("📋 Available African languages in CC-100: af, am, ar, as, ff, ha, ig, lg, ln, mg, mr, ms, my, ne, ns, om, or, pa, ps, qu, sc, sd, so, ss, sw, tn, wo, xh, yo, zu")
        print("📋 Available African languages in JW300: af-en, am-en, ar-en, as-en, dz-en, en-ha, en-ig, en-mg, en-mr, en-ms, en-ne, en-rw, en-sw, en-ta, en-wo, en-xh, en-yo, en-zu")
        sys.exit(1)


if __name__ == "__main__":
    main() 