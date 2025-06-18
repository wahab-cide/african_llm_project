"""
Download Hausa language corpora for language model training.

This script downloads Hausa text from multiple sources and saves them
under data/raw/hausa/ with proper directory structure.
"""

import sys
from pathlib import Path
from datasets import load_dataset, disable_caching

# Disable caching to avoid large ~/.cache folders
disable_caching()

# Configuration
LANG_CODE = "ha"
BASE_DIR = Path("data/raw/hausa")


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


def download_cc100() -> bool:
    """Download CC-100 Hausa corpus."""
    output_file = BASE_DIR / "cc100" / f"{LANG_CODE}.txt"
    
    # Skip if file already exists
    if output_file.exists():
        lines, size = get_file_stats(output_file)
        print(f"  CC-100 Hausa already exists: {output_file} ({size}, {lines:,} lines)")
        return True
    
    try:
        # Create directory
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load dataset with streaming
        print(" Downloading CC-100 Hausa...")
        dataset = load_dataset("cc100", LANG_CODE, split="train", streaming=True, trust_remote_code=True)
        
        # Write to file
        line_count = 0
        with output_file.open("w", encoding="utf-8") as f:
            for row in dataset:
                if row.get("text"):
                    f.write(row["text"].strip() + "\n")
                    line_count += 1
                
                # Print progress every 10,000 lines
                if line_count % 10000 == 0:
                    print(f"   Processed {line_count:,} lines...")
        
        # Get final stats and print success
        _, size = get_file_stats(output_file)
        print(f" CC-100 Hausa saved to {output_file} ({size}, {line_count:,} lines)")
        return True
        
    except Exception as e:
        print(f" Failed to download CC-100 Hausa: {e}")
        return False


def download_jw300() -> bool:
    """Download JW300 Hausa-English corpus (Hausa side only)."""
    output_file = BASE_DIR / "extra" / f"jw300_{LANG_CODE}.txt"
    
    # Skip if file already exists
    if output_file.exists():
        lines, size = get_file_stats(output_file)
        print(f" JW300 Hausa already exists: {output_file} ({size}, {lines:,} lines)")
        return True
    
    try:
        # Create directory
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load dataset - try both directions since we know en-ha exists
        print("  Downloading JW300 Hausa...")
        try:
            # Try en-ha first (this should work based on previous tests)
            dataset = load_dataset("opus100", f"en-{LANG_CODE}", split="train")
        except:
            # Fallback to ha-en
            dataset = load_dataset("opus100", f"{LANG_CODE}-en", split="train")
        
        # Write to file
        line_count = 0
        with output_file.open("w", encoding="utf-8") as f:
            for row in dataset:
                if row.get("translation") and row["translation"].get(LANG_CODE):
                    f.write(row["translation"][LANG_CODE].strip() + "\n")
                    line_count += 1
        
        # Get final stats and print success
        _, size = get_file_stats(output_file)
        print(f" JW300 Hausa saved to {output_file} ({size}, {line_count:,} lines)")
        return True
        
    except Exception as e:
        print(f"Failed to download JW300 Hausa: {e}")
        return False


def download_global_voices() -> bool:
    """Download Global Voices Hausa-English corpus (Hausa side only)."""
    output_file = BASE_DIR / "extra" / f"gv_{LANG_CODE}.txt"
    
    # Skip if file already exists
    if output_file.exists():
        lines, size = get_file_stats(output_file)
        print(f" Global Voices Hausa already exists: {output_file} ({size}, {lines:,} lines)")
        return True
    
    try:
        # Create directory
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        print(" Downloading Global Voices Hausa...")
        dataset = load_dataset("globalvoices", f"{LANG_CODE}-en", split="train")
        
        # Write to file
        line_count = 0
        with output_file.open("w", encoding="utf-8") as f:
            for row in dataset:
                if row.get("translation") and row["translation"].get(LANG_CODE):
                    f.write(row["translation"][LANG_CODE].strip() + "\n")
                    line_count += 1
        
        # Get final stats and print success
        _, size = get_file_stats(output_file)
        print(f"Global Voices Hausa saved to {output_file} ({size}, {line_count:,} lines)")
        return True
        
    except Exception as e:
        print(f"Failed to download Global Voices Hausa: {e}")
        return False


def download_oscar() -> bool:
    """Download OSCAR Hausa corpus if available."""
    output_file = BASE_DIR / "oscar" / f"oscar_{LANG_CODE}.txt"
    
    # Skip if file already exists
    if output_file.exists():
        lines, size = get_file_stats(output_file)
        print(f" OSCAR Hausa already exists: {output_file} ({size}, {lines:,} lines)")
        return True
    
    try:
        # Create directory
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load dataset with streaming
        print("Downloading OSCAR Hausa...")
        config = f"unshuffled_deduplicated_{LANG_CODE}"
        dataset = load_dataset("oscar", config, split="train", streaming=True)
        
        # Write to file
        line_count = 0
        with output_file.open("w", encoding="utf-8") as f:
            for row in dataset:
                if row.get("text"):
                    f.write(row["text"].strip() + "\n")
                    line_count += 1
                
                # Print progress every 10,000 lines
                if line_count % 10000 == 0:
                    print(f"   Processed {line_count:,} lines...")
        
        # Get final stats and print success
        _, size = get_file_stats(output_file)
        print(f"OSCAR Hausa saved to {output_file} ({size}, {line_count:,} lines)")
        return True
        
    except Exception as e:
        print(f"Failed to download OSCAR Hausa: {e}")
        return False


def main():
    """Main function to download all Hausa corpora."""
    print(" Starting Hausa corpus download...")
    print("=" * 50)
    
    # Download all available corpora
    success = True
    
    # Try CC-100 (should work - ha is available)
    if not download_cc100():
        success = False
    
    # Try JW300 (should work - en-ha is available)
    if not download_jw300():
        success = False
    
    # Try OSCAR (likely won't work - ha not in OSCAR)
    if not download_oscar():
        print("   (OSCAR Hausa not available - this is expected)")
    
    # Try Global Voices (may or may not work)
    if not download_global_voices():
        print("   (Global Voices Hausa may not be available)")
    
    # Summary
    print("\n" + "=" * 50)
    if success:
        print("All available downloads completed successfully!")
        print(f" Files saved under: {BASE_DIR}")
    else:
        print("  Some downloads failed.")
        print(" CC-100 and JW300 should work for Hausa.")
        print(" Available African languages in CC-100: af, am, ar, as, ff, ha, ig, lg, ln, mg, mr, ms, my, ne, ns, om, or, pa, ps, qu, sc, sd, so, ss, sw, tn, wo, xh, yo, zu")
        print(" Available African languages in JW300: af-en, am-en, ar-en, as-en, dz-en, en-ha, en-ig, en-mg, en-mr, en-ms, en-ne, en-rw, en-sw, en-ta, en-wo, en-xh, en-yo, en-zu")
        sys.exit(1)


if __name__ == "__main__":
    main() 