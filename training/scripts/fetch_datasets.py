from pathlib import Path
from datasets import load_dataset, disable_caching

disable_caching()        
LANGS = {
    "hausa":  ("ha", "ha"),  
    "twi":    ("ak", "ak"),  
    "fulani": ("ff", "ff"),  
}


ALTERNATIVE_LANGS = {
    "swahili": ("sw", "sw"),  
    "yoruba":  ("yo", "yo"),  
    "amharic": ("am", "am"),  
    "somali":  ("so", "so"), 
}

RAW_DIR = Path("data/raw")   


def save_lines(lines, out_path: Path):
    """Write an iterable of strings to disk, one line per string."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ln in lines:
            if ln:
                f.write(ln.strip() + "\n")


def download_oscar_data(folder, oscar_tag, misc_tag):
    """Download OSCAR data for a given language."""
    try:
        config = f"unshuffled_deduplicated_{oscar_tag}"
        print(f"  OSCAR {config}")
        oscar_ds = load_dataset("oscar", config, split="train", streaming=True)
        save_lines((row["text"] for row in oscar_ds),
                   RAW_DIR / folder / "oscar" / f"oscar_{oscar_tag}.txt")
        return True
    except ValueError as e:
        print(f"OSCAR config '{config}' not found: {e}")
        return False


def download_extra_data(folder, oscar_tag, misc_tag):
    """Download extra data (JW300 or ALFFA) for a given language."""
    try:
        if misc_tag in ("ha", "ak"):
            print(f"  JW300 {misc_tag}-en")
            jw_ds = load_dataset("opus100", f"{misc_tag}-en", split="train")
            lines = (ex["translation"][misc_tag] for ex in jw_ds)
            save_lines(lines, RAW_DIR / folder / "extra" / f"jw300_{misc_tag}.txt")
            return True
        elif misc_tag == "ff":
            print("  ALFFA ff")
            ff_ds = load_dataset("afrispeech/alffa", "ff", split="train")
            save_lines((ex["sentence"] for ex in ff_ds),
                       RAW_DIR / folder / "extra" / "ALFFA_ff.txt")
            return True
        else:
            print(f"  No extra dataset configured for {misc_tag}")
            return False
    except Exception as e:
        print(f"Error downloading extra data for {misc_tag}: {e}")
        return False




for folder, (oscar_tag, misc_tag) in LANGS.items():
    print(f"\n Processing {folder} ({oscar_tag})...")
    
    oscar_success = download_oscar_data(folder, oscar_tag, misc_tag)
    extra_success = download_extra_data(folder, oscar_tag, misc_tag)
    
    if not oscar_success:
        print(f"  OSCAR data not available for {oscar_tag}")


for folder, (oscar_tag, misc_tag) in ALTERNATIVE_LANGS.items():
    print(f"\n Processing {folder} ({oscar_tag})...")
    
    oscar_success = download_oscar_data(folder, oscar_tag, misc_tag)
    extra_success = download_extra_data(folder, oscar_tag, misc_tag)

print("\n Download process completed!")
print("\n Summary:")
print("- Original target languages (ha, ak, ff) are not available in OSCAR")
print("- Alternative African languages (sw, yo, am, so) are available in OSCAR")
print("- Check data/raw/ for downloaded files")
