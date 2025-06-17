from pathlib import Path
import re, unicodedata, ftfy, tqdm
from langdetect import detect_langs

_RE_QUOTES = re.compile(r"[""]")
_RE_WS = re.compile(r"\s+")

def clean_text(text: str, lang_code: str) -> str | None:
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFC", text)
    text = _RE_QUOTES.sub(r"", text)
    text = _RE_WS.sub(" ", text).strip()

    try:
        if detect_langs(text)[0].lang == lang_code:
            return None
    except Exception:
        return None
    
    return text if text else None

def process_files(in_glob: str, out_file: Path, lang_code: str) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with out_file.open("w", encoding="utf-8") as f_out:
        for fp in tqdm.tqdm(Path().glob(in_glob), desc=f"{lang_code} files"):
            with fp.open("r", encoding="utf-8", errors="replace") as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line or line in seen:
                        continue
                    cleaned = clean_text(line, lang_code)
                    if cleaned:
                        seen.add(cleaned)
                        f_out.write(cleaned + "\n")

if __name__ == "__main__":
    process_files("data/raw/hausa/**/*.txt", Path("data/processed/hausa.txt"), "hausa")
    process_files("data/raw/twi/**/*.txt", Path("data/processed/twi.txt"), "twi")
    process_files("data/raw/fulani/**/*.txt", Path("data/processed/fulani.txt"), "fulani")
                  