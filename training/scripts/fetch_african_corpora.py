
import argparse
import sys
from pathlib import Path
from datasets import load_dataset, disable_caching


disable_caching()

# Language configurations
LANGS = {
    "ha": {"name": "Hausa", "oscar": True, "extra": "jw300"},
    "ak": {"name": "Twi/Akan", "oscar": True, "extra": "jw300"},
    "ff": {"name": "Fulani", "oscar": True, "extra": "alffa"}
}

RAW_DIR = Path("data/raw")


def get_file_size(file_path: Path) -> str:
    """Get human-readable file size."""
    if not file_path.exists():
        return "0 B"
    
    size = file_path.stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def download_oscar(lang: str) -> bool:
    """Download OSCAR corpus for the given language."""
    output_file = RAW_DIR / lang / "oscar" / f"oscar_{lang}.txt"
    
 
    if output_file.exists():
        size = get_file_size(output_file)
        print(f"OSCAR {lang} already exists: {output_file} ({size})")
        return True
    
    try:
   
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
     
        config = f"unshuffled_deduplicated_{lang}"
        print(f"Downloading OSCAR {lang}...")
        
        dataset = load_dataset("oscar", config, split="train", streaming=True)
        
       
        with output_file.open("w", encoding="utf-8") as f:
            for row in dataset:
                if row.get("text"):
                    f.write(row["text"].strip() + "\n")
        
        size = get_file_size(output_file)
        print(f"Saved OSCAR {lang} to {output_file} ({size})")
        return True
        
    except Exception as e:
        print(f"Failed to download OSCAR {lang}: {e}")
        return False


def download_jw300(lang: str) -> bool:
    """Download JW300 corpus for Hausa or Twi."""
    output_file = RAW_DIR / lang / "extra" / f"jw300_{lang}.txt"
    
   
    if output_file.exists():
        size = get_file_size(output_file)
        print(f"JW300 {lang} already exists: {output_file} ({size})")
        return True
    
    try:
     
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        
        config = f"en-{lang}"
        print(f" Downloading JW300 {lang}...")
        
        dataset = load_dataset("opus100", config, split="train")
        
   
        with output_file.open("w", encoding="utf-8") as f:
            for row in dataset:
                if row.get("translation") and row["translation"].get(lang):
                    f.write(row["translation"][lang].strip() + "\n")
        
        size = get_file_size(output_file)
        print(f"Saved JW300 {lang} to {output_file} ({size})")
        return True
        
    except Exception as e:
        print(f"Failed to download JW300 {lang}: {e}")
        return False


def download_alffa() -> bool:
    """Download ALFFA corpus for Fulani."""
    output_file = RAW_DIR / "ff" / "extra" / "ALFFA_ff.txt"
    

    if output_file.exists():
        size = get_file_size(output_file)
        print(f"ALFFA ff already exists: {output_file} ({size})")
        return True
    
    try:
     
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
 
        print("Downloading ALFFA ff...")
        
        dataset = load_dataset("afrispeech/alffa", "ff", split="train")
        
     
        with output_file.open("w", encoding="utf-8") as f:
            for row in dataset:
                if row.get("sentence"):
                    f.write(row["sentence"].strip() + "\n")
        
        size = get_file_size(output_file)
        print(f"Saved ALFFA ff to {output_file} ({size})")
        return True
        
    except Exception as e:
        print(f"Failed to download ALFFA ff: {e}")
        return False


def download_language(lang: str) -> bool:
    """Download all corpora for a specific language."""
    if lang not in LANGS:
        print(f" Unknown language: {lang}")
        return False
    
    lang_info = LANGS[lang]
    print(f"\n Downloading {lang_info['name']} ({lang})...")
    print("=" * 50)
    
    success = True
    
    # Download OSCAR if available
    if lang_info["oscar"]:
        if not download_oscar(lang):
            success = False
    
    # Download extra corpus
    if lang_info["extra"] == "jw300":
        if not download_jw300(lang):
            success = False
    elif lang_info["extra"] == "alffa":
        if not download_alffa():
            success = False
    
    return success


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download African language corpora",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fetch_african_corpora.py          # Download all languages
  python fetch_african_corpora.py --lang ha  # Download only Hausa
  python fetch_african_corpora.py --lang ak  # Download only Twi/Akan
  python fetch_african_corpora.py --lang ff  # Download only Fulani
        """
    )
    
    parser.add_argument(
        "--lang",
        choices=["ha", "ak", "ff"],
        help="Download specific language only"
    )
    
    args = parser.parse_args()
    
   
    if args.lang:
        languages = [args.lang]
    else:
        languages = list(LANGS.keys())
    
    print("🚀 Starting African language corpus download...")
    

    all_success = True
    for lang in languages:
        if not download_language(lang):
            all_success = False
    

    print("\n" + "=" * 50)
    if all_success:
        print("All downloads completed successfully!")
    else:
        print("Some downloads failed. Check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main() 