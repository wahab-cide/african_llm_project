#!/usr/bin/env python3
"""
Enhanced data pipeline for African LLM with language tags, quality filtering,
and diverse content types (dialogue, fiction, children's stories).
"""

import re
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import unicodedata
import ftfy
from tqdm import tqdm
import json

# Language codes mapping
LANG_CODES = {
    "amharic": "am",
    "fulani": "ff", 
    "hausa": "ha",
    "somali": "so",
    "swahili": "sw",
    "yoruba": "yo"
}

# Content type tags
CONTENT_TYPES = {
    "dialogue": "<dialogue>",
    "fiction": "<fiction>", 
    "children": "<children>",
    "news": "<news>",
    "academic": "<academic>",
    "general": "<general>"
}

def detect_content_type(text: str) -> str:
    """Detect content type based on text patterns."""
    text_lower = text.lower()
    
    # Dialogue detection
    if re.search(r'["""].*["""]', text) or re.search(r'[-–—].*[-–—]', text):
        return "dialogue"
    
    # Children's content detection
    children_keywords = ['mtoto', 'yaro', 'child', 'kid', 'toy', 'game', 'play', 'story']
    if any(keyword in text_lower for keyword in children_keywords):
        return "children"
    
    # Fiction detection
    fiction_keywords = ['once', 'story', 'tale', 'legend', 'myth', 'adventure']
    if any(keyword in text_lower for keyword in fiction_keywords):
        return "fiction"
    
    # Academic detection
    academic_keywords = ['research', 'study', 'analysis', 'data', 'method', 'theory']
    if any(keyword in text_lower for keyword in academic_keywords):
        return "academic"
    
    # News detection
    news_keywords = ['news', 'report', 'announce', 'government', 'official']
    if any(keyword in text_lower for keyword in news_keywords):
        return "news"
    
    return "general"

def clean_text_enhanced(text: str) -> Optional[str]:
    """Enhanced text cleaning with better quality filters."""
    # Basic cleaning
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    if len(text) < 15:  # Increased minimum length
        return None
    
    # Remove lines with too many digits
    digit_ratio = sum(1 for c in text if c.isdigit()) / len(text)
    if digit_ratio >= 0.2:  # Reduced threshold
        return None
    
    # Remove lines with problematic patterns
    bad_patterns = [
        r'ም\.', r'\.com', r'www\.', r'http', r'@', r'#',
        r'\d{4}-\d{2}-\d{2}',  # Dates
        r'\d{2}:\d{2}:\d{2}',  # Times
        r'[A-Z]{5,}',  # All caps words
    ]
    
    for pattern in bad_patterns:
        if re.search(pattern, text):
            return None
    
    # Check for minimum meaningful content
    alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
    if alpha_ratio < 0.4:  # Increased threshold
        return None
    
    # Remove lines that are mostly punctuation
    punct_ratio = sum(1 for c in text if c in '.,!?;:()[]{}') / len(text)
    if punct_ratio > 0.3:
        return None
    
    return text

def add_language_tags(text: str, lang_code: str, content_type: str = None) -> str:
    """Add language and content type tags to text."""
    if content_type is None:
        content_type = detect_content_type(text)
    
    # Format: <lang> <content_type> text <|sep|> continuation...
    tag = f"<{lang_code}> {CONTENT_TYPES[content_type]}"
    
    # Split long text into chunks with separator
    if len(text) > 200:
        # Find good split points (sentences, periods)
        sentences = re.split(r'([.!?]+)', text)
        chunks = []
        current_chunk = ""
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
            
            if len(current_chunk + sentence) < 200:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Join chunks with separator
        return f"{tag} {chunks[0]} <|sep|> " + " <|sep|> ".join(chunks[1:])
    else:
        return f"{tag} {text}"

def create_dialogue_examples(lang_code: str) -> List[str]:
    """Create synthetic dialogue examples for each language."""
    dialogues = {
        "ha": [
            "Ina kwana? Lafiya lau. Yaya kake?",
            "Me kake yi? Ina aiki. Kana da yara?",
            "Ina zuwa gida. Sai an jima. Barka da rana."
        ],
        "sw": [
            "Habari? Nzuri. Ukoje?",
            "Unafanya nini? Ninafanya kazi. Una watoto?",
            "Ninaenda nyumbani. Kwaheri. Asante."
        ],
        "yo": [
            "Bawo ni? Dara. Se daadaa ni?",
            "Kini o n se? Mo n se ise. O ni omo?",
            "Mo n lo ile. O daabo. O se."
        ],
        "am": [
            "Dehna neh? Dehna. Indet neh?",
            "Min aleh? Sira neh. Lijoch alesh?",
            "Bet yemetah. Dehna hun. Ameseginalehu."
        ],
        "so": [
            "Iska warran? Wanaagsan. Sidee tahay?",
            "Maxaad qabanaysaa? Shaqo baan qabanayaa. Ilmo ma leedahay?",
            "Guriga ayaan aadayaa. Nabad gelyo. Mahadsanid."
        ],
        "ff": [
            "Jam na? Jam. No ngon?",
            "Ko honɗa? Miɗo golli. Aɗa woodi ɓiɓɓe?",
            "Miɗo yahay suudu. Jam. Jaraama."
        ]
    }
    
    lang_dialogues = dialogues.get(lang_code, dialogues["ha"])
    return [f"<{lang_code}> {CONTENT_TYPES['dialogue']} {d}" for d in lang_dialogues]

def create_children_stories(lang_code: str) -> List[str]:
    """Create simple children's stories for each language."""
    stories = {
        "ha": [
            "Wani yaro yana da kare. Kare yana son wasa. Yaro da kare suna tafiya a fili.",
            "Wata yarinya tana da bishiya. Bishiya tana da 'ya'ya. Yarinya tana cin 'ya'yan bishiya."
        ],
        "sw": [
            "Mtoto mmoja alikuwa na mbwa. Mbwa alipenda kucheza. Mtoto na mbwa walikwenda uwanjani.",
            "Msichana mmoja alikuwa na mti. Mti ulikuwa na matunda. Msichana alikula matunda ya mti."
        ],
        "yo": [
            "Omo kan ni aja. Aja fẹ́ràn eré. Omo ati aja lọ sí pápá.",
            "Obinrin kan ni igi. Igi ni èso. Obinrin jẹ èso igi."
        ]
    }
    
    lang_stories = stories.get(lang_code, stories["ha"])
    return [f"<{lang_code}> {CONTENT_TYPES['children']} {s}" for s in lang_stories]

def process_language_file(file_path: Path, lang_code: str, output_dir: Path) -> Dict[str, int]:
    """Process a single language file with enhanced cleaning and tagging."""
    output_file = output_dir / f"{lang_code}_enhanced.txt"
    stats = {"total": 0, "cleaned": 0, "dialogue": 0, "children": 0, "fiction": 0, "news": 0, "academic": 0, "general": 0}
    
    enhanced_lines = []
    
    # Read and process original file
    with file_path.open('r', encoding='utf-8', errors='replace') as f:
        for line in tqdm(f, desc=f"Processing {lang_code}"):
            stats["total"] += 1
            line = line.strip()
            
            if not line:
                continue
            
            cleaned = clean_text_enhanced(line)
            if cleaned:
                stats["cleaned"] += 1
                content_type = detect_content_type(cleaned)
                stats[content_type] += 1
                
                tagged_text = add_language_tags(cleaned, lang_code, content_type)
                enhanced_lines.append(tagged_text)
    
    # Add synthetic examples for diversity
    dialogue_examples = create_dialogue_examples(lang_code)
    children_examples = create_children_stories(lang_code)
    
    enhanced_lines.extend(dialogue_examples)
    enhanced_lines.extend(children_examples)
    
    stats["dialogue"] += len(dialogue_examples)
    stats["children"] += len(children_examples)
    
    # Shuffle and save
    random.shuffle(enhanced_lines)
    
    with output_file.open('w', encoding='utf-8') as f:
        for line in enhanced_lines:
            f.write(line + '\n')
    
    return stats

def main():
    """Main processing function."""
    input_dir = Path("data/enhanced_raw")
    output_dir = Path("data/enhanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each language
    all_stats = {}
    
    for lang_name, lang_code in LANG_CODES.items():
        input_file = input_dir / f"{lang_name}_enhanced_raw.txt"
        
        if not input_file.exists():
            print(f"Warning: {input_file} not found, skipping {lang_code}")
            continue
        
        print(f"\nProcessing {lang_name} ({lang_code})...")
        stats = process_language_file(input_file, lang_code, output_dir)
        all_stats[lang_code] = stats
        
        print(f"  Total lines: {stats['total']:,}")
        print(f"  Cleaned lines: {stats['cleaned']:,}")
        print(f"  Dialogue: {stats['dialogue']:,}")
        print(f"  Children: {stats['children']:,}")
        print(f"  Fiction: {stats['fiction']:,}")
        print(f"  News: {stats['news']:,}")
        print(f"  Academic: {stats['academic']:,}")
        print(f"  General: {stats['general']:,}")
    
    # Save statistics
    stats_file = output_dir / "processing_stats.json"
    with stats_file.open('w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"\n✅ Enhanced data processing completed!")
    print(f"Output directory: {output_dir}")
    print(f"Statistics saved to: {stats_file}")

if __name__ == "__main__":
    main() 