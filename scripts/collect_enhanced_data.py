#!/usr/bin/env python3
"""
Enhanced data collection script for African LLM with focus on high-quality,
diverse sources including dialogue, fiction, children's stories, and news.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import requests
import json
import time
import random
from datasets import load_dataset, disable_caching
from tqdm import tqdm
import re
import unicodedata
import ftfy

# Disable caching to avoid large ~/.cache folders
disable_caching()

# Language configurations with multiple data sources
LANGUAGE_SOURCES = {
    "hausa": {
        "name": "Hausa",
        "code": "ha",
        "sources": [
            {
                "name": "OSCAR",
                "config": "unshuffled_deduplicated_ha",
                "type": "web_text"
            },
            {
                "name": "CC-100",
                "config": "ha",
                "type": "web_text"
            },
            {
                "name": "JW300",
                "config": "ha-en",
                "type": "parallel"
            },
            {
                "name": "Global Voices",
                "config": "ha-en",
                "type": "parallel"
            },
            {
                "name": "Masakhane",
                "config": "ha",
                "type": "news"
            }
        ]
    },
    "swahili": {
        "name": "Swahili",
        "code": "sw",
        "sources": [
            {
                "name": "OSCAR",
                "config": "unshuffled_deduplicated_sw",
                "type": "web_text"
            },
            {
                "name": "CC-100",
                "config": "sw",
                "type": "web_text"
            },
            {
                "name": "JW300",
                "config": "sw-en",
                "type": "parallel"
            },
            {
                "name": "Global Voices",
                "config": "sw-en",
                "type": "parallel"
            },
            {
                "name": "Masakhane",
                "config": "sw",
                "type": "news"
            }
        ]
    },
    "yoruba": {
        "name": "Yoruba",
        "code": "yo",
        "sources": [
            {
                "name": "OSCAR",
                "config": "unshuffled_deduplicated_yo",
                "type": "web_text"
            },
            {
                "name": "CC-100",
                "config": "yo",
                "type": "web_text"
            },
            {
                "name": "JW300",
                "config": "yo-en",
                "type": "parallel"
            },
            {
                "name": "Global Voices",
                "config": "yo-en",
                "type": "parallel"
            }
        ]
    },
    "amharic": {
        "name": "Amharic",
        "code": "am",
        "sources": [
            {
                "name": "OSCAR",
                "config": "unshuffled_deduplicated_am",
                "type": "web_text"
            },
            {
                "name": "CC-100",
                "config": "am",
                "type": "web_text"
            },
            {
                "name": "JW300",
                "config": "am-en",
                "type": "parallel"
            },
            {
                "name": "Global Voices",
                "config": "am-en",
                "type": "parallel"
            }
        ]
    },
    "somali": {
        "name": "Somali",
        "code": "so",
        "sources": [
            {
                "name": "OSCAR",
                "config": "unshuffled_deduplicated_so",
                "type": "web_text"
            },
            {
                "name": "CC-100",
                "config": "so",
                "type": "web_text"
            },
            {
                "name": "JW300",
                "config": "so-en",
                "type": "parallel"
            },
            {
                "name": "Global Voices",
                "config": "so-en",
                "type": "parallel"
            }
        ]
    },
    "fulani": {
        "name": "Fulani",
        "code": "ff",
        "sources": [
            {
                "name": "OSCAR",
                "config": "unshuffled_deduplicated_ff",
                "type": "web_text"
            },
            {
                "name": "ALFFA",
                "config": "ff",
                "type": "speech_text"
            },
            {
                "name": "JW300",
                "config": "ff-en",
                "type": "parallel"
            }
        ]
    }
}

# High-quality synthetic data for each language
SYNTHETIC_DATA = {
    "hausa": {
        "dialogues": [
            "Ina kwana? Lafiya lau. Yaya kake?",
            "Me kake yi? Ina aiki. Kana da yara?",
            "Ina zuwa gida. Sai an jima. Barka da rana.",
            "Kana da abinci? E, ina da shi. Ka ci?",
            "Yaya aikin gida? Yana da kyau. Ka gama?",
            "Ina za ka tafi? Ina zuwa kasuwa. Ka dawo da wuri.",
            "Kana da kuɗi? E, ina da shi. Nawa?",
            "Yaya makaranta? Yana da kyau. Ka koyi?",
            "Ina za ka ci abinci? Ina zuwa gidan abinci. Ka dawo.",
            "Kana da mota? A'a, ba ni da ita. Ka tafi da bas?"
        ],
        "stories": [
            "Wani yaro yana da kare. Kare yana son wasa. Yaro da kare suna tafiya a fili.",
            "Wata yarinya tana da bishiya. Bishiya tana da 'ya'ya. Yarinya tana cin 'ya'yan bishiya.",
            "Wani mutum yana da gida. Gida yana da kyau. Mutum yana zaune a gida.",
            "Wata mata tana da littafi. Littafi yana da kyau. Mata tana karanta littafi.",
            "Wani yaro yana da ball. Ball yana da kyau. Yaro yana wasa da ball."
        ],
        "news": [
            "Gwamnati ta sanar da sabon shirin. Shiri zai taimaka wa talakawa.",
            "Hukumar kula da harkokin makarantu ta sanar da sabon tsari.",
            "Masana kimiyya sun gano sabon magani. Magani zai taimaka wa marasa lafiya.",
            "Hukumar kula da harkokin sufuri ta sanar da sabon tsari.",
            "Masana tattalin arziki sun yi nazari kan tattalin arzikin ƙasa."
        ],
        "instructions": [
            "Don yin shayi, ka saka ruwa a tukunya. Ka dora a wuta. Ka jira har ya tafasa.",
            "Don yin abinci, ka wanke kayan abinci. Ka yanka su. Ka dora a wuta.",
            "Don yin wanka, ka saka ruwa a kwano. Ka saka sabulu. Ka wanke jikinka.",
            "Don yin aiki, ka tashi da wuri. Ka tafi wurin aiki. Ka yi aikin da kyau.",
            "Don yin karatu, ka zauna a wuri mai haske. Ka buɗe littafi. Ka karanta da hankali."
        ]
    },
    "swahili": {
        "dialogues": [
            "Habari? Nzuri. Ukoje?",
            "Unafanya nini? Ninafanya kazi. Una watoto?",
            "Ninaenda nyumbani. Kwaheri. Asante.",
            "Una chakula? Ndiyo, nina chakula. Ulikula?",
            "Jinsi gani kazi ya nyumbani? Nzuri. Umemaliza?",
            "Unaenda wapi? Ninaenda sokoni. Urudi mapema.",
            "Una pesa? Ndiyo, nina pesa. Ngapi?",
            "Jinsi gani shule? Nzuri. Umesoma?",
            "Unaenda kula wapi? Ninaenda hoteli. Urudi.",
            "Una gari? Hapana, sina gari. Unaenda na basi?"
        ],
        "stories": [
            "Mtoto mmoja alikuwa na mbwa. Mbwa alipenda kucheza. Mtoto na mbwa walikwenda uwanjani.",
            "Msichana mmoja alikuwa na mti. Mti ulikuwa na matunda. Msichana alikula matunda ya mti.",
            "Mtu mmoja alikuwa na nyumba. Nyumba ilikuwa nzuri. Mtu alikaa nyumbani.",
            "Mwanamke mmoja alikuwa na kitabu. Kitabu kilikuwa kizuri. Mwanamke alisoma kitabu.",
            "Kijana mmoja alikuwa na mpira. Mpira ulikuwa mzuri. Kijana alicheza na mpira."
        ],
        "news": [
            "Serikali imetangaza mpango mpya. Mpango utasaidia watu maskini.",
            "Wizara ya elimu imetangaza mfumo mpya.",
            "Wanasayansi wamegundua dawa mpya. Dawa itasaidia wagonjwa.",
            "Wizara ya usafiri imetangaza mfumo mpya.",
            "Wachumi wamefanya uchambuzi wa uchumi wa nchi."
        ],
        "instructions": [
            "Kufanya chai, weka maji kwenye sufuria. Weka motoni. Subiri mpaka yachemshe.",
            "Kufanya chakula, osha vyakula. Kata. Weka motoni.",
            "Kufanya oga, weka maji kwenye bakuli. Weka sabuni. Oga mwili wako.",
            "Kufanya kazi, amka mapema. Nenda mahali pa kazi. Fanya kazi vizuri.",
            "Kusoma, kaa mahali penye mwanga. Fungua kitabu. Soma kwa makini."
        ]
    },
    "yoruba": {
        "dialogues": [
            "Bawo ni? Dara. Se daadaa ni?",
            "Kini o n se? Mo n se ise. O ni omo?",
            "Mo n lo ile. O daabo. O se.",
            "O ni ounje? Beeni, mo ni ounje. O ti je?",
            "Bawo ni ise ile? Dara. O ti pari?",
            "Ibo lo n lo? Mo n lo oja. O pada ni kiakia.",
            "O ni owo? Beeni, mo ni owo. Melo?",
            "Bawo ni ile iwe? Dara. O ti ka?",
            "Ibo lo n lo je? Mo n lo ile ounje. O pada.",
            "O ni oko? Rara, ko si mi. O n lo ni basi?"
        ],
        "stories": [
            "Omo kan ni aja. Aja fẹ́ràn eré. Omo ati aja lọ sí pápá.",
            "Obinrin kan ni igi. Igi ni èso. Obinrin jẹ èso igi.",
            "Enikan ni ile. Ile dara. Enikan gbé inú ile.",
            "Obirin kan ni iwe. Iwe dara. Obirin ka iwe.",
            "Omo kan ni bọọlu. Bọọlu dara. Omo fi bọọlu eré."
        ],
        "news": [
            "Ijoba ti fi agbara jade titun. Agbara yoo ran awọn talaka lowo.",
            "Igbimọ ile iwe ti fi eto titun jade.",
            "Awọn onimo sayensi ti ri oogun titun. Oogun yoo ran awọn alaisan lowo.",
            "Igbimọ irin ajo ti fi eto titun jade.",
            "Awọn onimo owo ti se iwadi lori owo orile-ede."
        ],
        "instructions": [
            "Lati se tii, fi omi sinu ikoko. Fi ina si. Duro titi ti o gbona.",
            "Lati se ounje, wo awọn ounje. Ge won. Fi ina si.",
            "Lati se wẹ, fi omi sinu abo. Fi ose si. Wẹ ara re.",
            "Lati se ise, jide ni kiakia. Lo ibi ise. Se ise ni daradara.",
            "Lati ka, joko ibi ti o ni imole. Si iwe. Ka ni akitiyan."
        ]
    }
}

def clean_text_enhanced(text: str) -> Optional[str]:
    """Enhanced text cleaning with better quality filters."""
    # Basic cleaning
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    if len(text) < 20:  # Increased minimum length
        return None
    
    # Remove lines with too many digits
    digit_ratio = sum(1 for c in text if c.isdigit()) / len(text)
    if digit_ratio >= 0.15:  # Reduced threshold
        return None
    
    # Remove lines with problematic patterns
    bad_patterns = [
        r'ም\.', r'\.com', r'www\.', r'http', r'@', r'#',
        r'\d{4}-\d{2}-\d{2}',  # Dates
        r'\d{2}:\d{2}:\d{2}',  # Times
        r'[A-Z]{5,}',  # All caps words
        r'[^\w\s\.,!?;:()\[\]{}"\'-]',  # Too many special chars
    ]
    
    for pattern in bad_patterns:
        if re.search(pattern, text):
            return None
    
    # Check for minimum meaningful content
    alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
    if alpha_ratio < 0.5:  # Increased threshold
        return None
    
    # Remove lines that are mostly punctuation
    punct_ratio = sum(1 for c in text if c in '.,!?;:()[]{}') / len(text)
    if punct_ratio > 0.25:
        return None
    
    return text

def download_oscar_data(lang_code: str, output_file: Path) -> int:
    """Download OSCAR data for a language."""
    try:
        config = f"unshuffled_deduplicated_{lang_code}"
        print(f"  Downloading OSCAR {config}...")
        
        dataset = load_dataset("oscar", config, split="train", streaming=True)
        
        line_count = 0
        with output_file.open('w', encoding='utf-8') as f:
            for row in tqdm(dataset, desc=f"OSCAR {lang_code}"):
                if row.get("text"):
                    cleaned = clean_text_enhanced(row["text"])
                    if cleaned:
                        f.write(cleaned + '\n')
                        line_count += 1
                
                # Limit to reasonable size
                if line_count >= 50000:
                    break
        
        return line_count
    except Exception as e:
        print(f"  Failed to download OSCAR {lang_code}: {e}")
        return 0

def download_cc100_data(lang_code: str, output_file: Path) -> int:
    """Download CC-100 data for a language."""
    try:
        print(f"  Downloading CC-100 {lang_code}...")
        
        dataset = load_dataset("cc100", lang_code, split="train", streaming=True)
        
        line_count = 0
        with output_file.open('w', encoding='utf-8') as f:
            for row in tqdm(dataset, desc=f"CC-100 {lang_code}"):
                if row.get("text"):
                    cleaned = clean_text_enhanced(row["text"])
                    if cleaned:
                        f.write(cleaned + '\n')
                        line_count += 1
                
                # Limit to reasonable size
                if line_count >= 50000:
                    break
        
        return line_count
    except Exception as e:
        print(f"  Failed to download CC-100 {lang_code}: {e}")
        return 0

def download_jw300_data(lang_code: str, output_file: Path) -> int:
    """Download JW300 parallel data for a language."""
    try:
        config = f"{lang_code}-en"
        print(f"  Downloading JW300 {config}...")
        
        dataset = load_dataset("opus100", config, split="train")
        
        line_count = 0
        with output_file.open('w', encoding='utf-8') as f:
            for row in tqdm(dataset, desc=f"JW300 {lang_code}"):
                if row.get("translation") and row["translation"].get(lang_code):
                    text = row["translation"][lang_code]
                    cleaned = clean_text_enhanced(text)
                    if cleaned:
                        f.write(cleaned + '\n')
                        line_count += 1
                
                # Limit to reasonable size
                if line_count >= 10000:
                    break
        
        return line_count
    except Exception as e:
        print(f"  Failed to download JW300 {lang_code}: {e}")
        return 0

def download_global_voices_data(lang_code: str, output_file: Path) -> int:
    """Download Global Voices data for a language."""
    try:
        config = f"{lang_code}-en"
        print(f"  Downloading Global Voices {config}...")
        
        dataset = load_dataset("globalvoices", config, split="train")
        
        line_count = 0
        with output_file.open('w', encoding='utf-8') as f:
            for row in tqdm(dataset, desc=f"Global Voices {lang_code}"):
                if row.get("translation") and row["translation"].get(lang_code):
                    text = row["translation"][lang_code]
                    cleaned = clean_text_enhanced(text)
                    if cleaned:
                        f.write(cleaned + '\n')
                        line_count += 1
                
                # Limit to reasonable size
                if line_count >= 5000:
                    break
        
        return line_count
    except Exception as e:
        print(f"  Failed to download Global Voices {lang_code}: {e}")
        return 0

def download_masakhane_data(lang_code: str, output_file: Path) -> int:
    """Download Masakhane news data for a language."""
    try:
        print(f"  Downloading Masakhane {lang_code}...")
        
        dataset = load_dataset("masakhane/masakhane-news", lang_code, split="train")
        
        line_count = 0
        with output_file.open('w', encoding='utf-8') as f:
            for row in tqdm(dataset, desc=f"Masakhane {lang_code}"):
                if row.get("text"):
                    cleaned = clean_text_enhanced(row["text"])
                    if cleaned:
                        f.write(cleaned + '\n')
                        line_count += 1
                
                # Limit to reasonable size
                if line_count >= 5000:
                    break
        
        return line_count
    except Exception as e:
        print(f"  Failed to download Masakhane {lang_code}: {e}")
        return 0

def download_alffa_data(lang_code: str, output_file: Path) -> int:
    """Download ALFFA speech data for Fulani."""
    try:
        print(f"  Downloading ALFFA {lang_code}...")
        
        dataset = load_dataset("afrispeech/alffa", lang_code, split="train")
        
        line_count = 0
        with output_file.open('w', encoding='utf-8') as f:
            for row in tqdm(dataset, desc=f"ALFFA {lang_code}"):
                if row.get("sentence"):
                    cleaned = clean_text_enhanced(row["sentence"])
                    if cleaned:
                        f.write(cleaned + '\n')
                        line_count += 1
        
        return line_count
    except Exception as e:
        print(f"  Failed to download ALFFA {lang_code}: {e}")
        return 0

def add_synthetic_data(lang_name: str, output_file: Path) -> int:
    """Add high-quality synthetic data for a language."""
    if lang_name not in SYNTHETIC_DATA:
        return 0
    
    synthetic = SYNTHETIC_DATA[lang_name]
    line_count = 0
    
    with output_file.open('a', encoding='utf-8') as f:
        # Add dialogues
        for dialogue in synthetic.get("dialogues", []):
            f.write(f"<dialogue> {dialogue}\n")
            line_count += 1
        
        # Add stories
        for story in synthetic.get("stories", []):
            f.write(f"<story> {story}\n")
            line_count += 1
        
        # Add news
        for news in synthetic.get("news", []):
            f.write(f"<news> {news}\n")
            line_count += 1
        
        # Add instructions
        for instruction in synthetic.get("instructions", []):
            f.write(f"<instruction> {instruction}\n")
            line_count += 1
    
    return line_count

def collect_language_data(lang_name: str, lang_config: Dict, output_dir: Path) -> Dict[str, int]:
    """Collect data for a specific language from multiple sources."""
    lang_code = lang_config["code"]
    output_file = output_dir / f"{lang_name}_enhanced_raw.txt"
    
    print(f"\n{'='*50}")
    print(f"Collecting data for {lang_config['name']} ({lang_code})")
    print(f"{'='*50}")
    
    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    total_lines = 0
    source_stats = {}
    
    # Download from each source
    for source in lang_config["sources"]:
        source_name = source["name"].lower()
        source_file = output_dir / f"{lang_name}_{source_name}.txt"
        
        if source["name"] == "OSCAR":
            lines = download_oscar_data(lang_code, source_file)
        elif source["name"] == "CC-100":
            lines = download_cc100_data(lang_code, source_file)
        elif source["name"] == "JW300":
            lines = download_jw300_data(lang_code, source_file)
        elif source["name"] == "Global Voices":
            lines = download_global_voices_data(lang_code, source_file)
        elif source["name"] == "Masakhane":
            lines = download_masakhane_data(lang_code, source_file)
        elif source["name"] == "ALFFA":
            lines = download_alffa_data(lang_code, source_file)
        else:
            print(f"  Unknown source: {source['name']}")
            continue
        
        source_stats[source["name"]] = lines
        total_lines += lines
        print(f"  {source['name']}: {lines:,} lines")
    
    # Combine all sources
    print(f"\nCombining sources...")
    with output_file.open('w', encoding='utf-8') as out_f:
        for source in lang_config["sources"]:
            source_name = source["name"].lower()
            source_file = output_dir / f"{lang_name}_{source_name}.txt"
            
            if source_file.exists():
                with source_file.open('r', encoding='utf-8') as in_f:
                    for line in in_f:
                        out_f.write(line)
    
    # Add synthetic data
    synthetic_lines = add_synthetic_data(lang_name, output_file)
    source_stats["Synthetic"] = synthetic_lines
    total_lines += synthetic_lines
    
    print(f"Synthetic: {synthetic_lines:,} lines")
    print(f"Total: {total_lines:,} lines")
    
    return source_stats

def main():
    parser = argparse.ArgumentParser(
        description="Collect enhanced data for African LLM from multiple sources"
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=list(LANGUAGE_SOURCES.keys()),
        help="Languages to collect data for (default: all)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/enhanced_raw",
        help="Output directory for collected data (default: data/enhanced_raw)"
    )
    parser.add_argument(
        "--focus-hausa",
        action="store_true",
        help="Focus on Hausa data collection with extra sources"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🌍 ENHANCED AFRICAN LLM DATA COLLECTION")
    print("="*60)
    print(f"Languages: {', '.join(args.languages)}")
    print(f"Output directory: {output_dir}")
    if args.focus_hausa:
        print("Focus: Hausa with extra sources")
    print("="*60)
    
    all_stats = {}
    
    # Collect data for each language
    for lang_name in args.languages:
        if lang_name not in LANGUAGE_SOURCES:
            print(f"Warning: Unknown language {lang_name}, skipping")
            continue
        
        lang_config = LANGUAGE_SOURCES[lang_name]
        stats = collect_language_data(lang_name, lang_config, output_dir)
        all_stats[lang_name] = stats
    
    # Print summary
    print(f"\n{'='*60}")
    print("COLLECTION SUMMARY")
    print(f"{'='*60}")
    
    for lang_name, stats in all_stats.items():
        total = sum(stats.values())
        print(f"\n{lang_name.upper()}: {total:,} total lines")
        for source, lines in stats.items():
            if lines > 0:
                print(f"  {source}: {lines:,} lines")
    
    print(f"\n✅ Data collection completed!")
    print(f"Output directory: {output_dir}")
    print(f"\nNext steps:")
    print("1. Review collected data quality")
    print("2. Run enhanced data processing:")
    print("   python scripts/enhance_data_pipeline.py")
    print("3. Build enhanced dataset:")
    print("   python scripts/build_enhanced_dataset.py")

if __name__ == "__main__":
    main() 