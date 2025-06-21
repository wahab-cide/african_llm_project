#!/usr/bin/env python3
"""
Enhanced interactive demo for African LLM with language-specific generation
and improved prompting capabilities.
"""

import argparse
import sys
from pathlib import Path
import torch
from transformers import GPT2LMHeadModel, GPT2Config
import sentencepiece as spm
import re
from typing import Optional, List, Dict

class EnhancedSentencePieceTokenizer:
    """Enhanced wrapper for SentencePiece tokenizer with language support."""
    
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        
        # Set special tokens
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.sep_token = "<|sep|>"
        
        # Get token IDs
        self.bos_token_id = self.sp.piece_to_id(self.bos_token)
        self.eos_token_id = self.sp.piece_to_id(self.eos_token)
        self.unk_token_id = self.sp.piece_to_id(self.unk_token)
        self.pad_token_id = self.sp.piece_to_id(self.pad_token)
        self.sep_token_id = self.sp.piece_to_id(self.sep_token)
        
        self.vocab_size = self.sp.get_piece_size()
    
    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode text to token IDs."""
        return self.sp.encode_as_ids(text)
    
    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode token IDs to text."""
        return self.sp.decode_ids(token_ids)
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs."""
        return [self.sp.piece_to_id(token) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert IDs to tokens."""
        return [self.sp.id_to_piece(id) for id in ids]

class EnhancedAfricanLLMDemo:
    """Enhanced demo for African LLM with language-specific generation."""
    
    def __init__(self, model_path: str, tokenizer_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = EnhancedSentencePieceTokenizer(tokenizer_path)
        
        # Load model
        print("Loading model...")
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Language configurations
        self.languages = {
            "am": {"name": "Amharic", "greeting": "Selam"},
            "ff": {"name": "Fulani", "greeting": "Jam na"},
            "ha": {"name": "Hausa", "greeting": "Sannu"},
            "so": {"name": "Somali", "greeting": "Iska warran"},
            "sw": {"name": "Swahili", "greeting": "Habari"},
            "yo": {"name": "Yoruba", "greeting": "Bawo ni"}
        }
        
        # Content type prompts
        self.content_prompts = {
            "dialogue": "Create a natural conversation:",
            "story": "Tell a short story:",
            "news": "Write a news headline and brief article:",
            "poem": "Write a short poem:",
            "instruction": "Give instructions on how to:"
        }
        
        print("✅ Model loaded successfully!")
    
    def generate_text(
        self, 
        prompt: str, 
        max_length: int = 100, 
        temperature: float = 0.8,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """Generate text with enhanced parameters."""
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        # Generate
        with torch.no_grad():
            output = self.model.generate(
                input_tensor,
                max_length=max_length + len(input_ids),
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2
            )
        
        # Decode and return generated text
        generated_ids = output[0][len(input_ids):]
        generated_text = self.tokenizer.decode(generated_ids)
        
        return generated_text.strip()
    
    def create_language_prompt(self, lang_code: str, content_type: str, user_input: str) -> str:
        """Create a properly formatted prompt with language and content tags."""
        lang_info = self.languages.get(lang_code, {"name": "Unknown", "greeting": "Hello"})
        
        if content_type == "dialogue":
            return f"<{lang_code}> <dialogue> {lang_info['greeting']} {user_input}"
        elif content_type == "story":
            return f"<{lang_code}> <fiction> {user_input}"
        elif content_type == "news":
            return f"<{lang_code}> <news> {user_input}"
        elif content_type == "poem":
            return f"<{lang_code}> <fiction> {user_input}"
        elif content_type == "instruction":
            return f"<{lang_code}> <general> {user_input}"
        else:
            return f"<{lang_code}> <general> {user_input}"
    
    def interactive_mode(self):
        """Run interactive demo mode."""
        print("\n" + "="*60)
        print("🌍 ENHANCED AFRICAN LLM DEMO")
        print("="*60)
        print("Available languages:")
        for code, info in self.languages.items():
            print(f"  {code}: {info['name']} ({info['greeting']})")
        
        print("\nContent types:")
        for content, desc in self.content_prompts.items():
            print(f"  {content}: {desc}")
        
        print("\nCommands:")
        print("  /help - Show this help")
        print("  /lang <code> - Set language (e.g., /lang sw)")
        print("  /type <type> - Set content type (e.g., /type dialogue)")
        print("  /temp <value> - Set temperature (0.1-2.0)")
        print("  /quit - Exit demo")
        print("="*60)
        
        # Default settings
        current_lang = "sw"
        current_type = "general"
        temperature = 0.8
        
        while True:
            try:
                # Get user input
                user_input = input(f"\n[{self.languages[current_lang]['name']}/{current_type}] > ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    if user_input == "/quit":
                        print("Goodbye! 👋")
                        break
                    elif user_input == "/help":
                        print("\nCommands:")
                        print("  /help - Show this help")
                        print("  /lang <code> - Set language (e.g., /lang sw)")
                        print("  /type <type> - Set content type (e.g., /type dialogue)")
                        print("  /temp <value> - Set temperature (0.1-2.0)")
                        print("  /quit - Exit demo")
                        continue
                    elif user_input.startswith("/lang "):
                        lang_code = user_input.split()[1].lower()
                        if lang_code in self.languages:
                            current_lang = lang_code
                            print(f"Language set to: {self.languages[lang_code]['name']}")
                        else:
                            print(f"Unknown language: {lang_code}")
                        continue
                    elif user_input.startswith("/type "):
                        content_type = user_input.split()[1].lower()
                        if content_type in self.content_prompts:
                            current_type = content_type
                            print(f"Content type set to: {content_type}")
                        else:
                            print(f"Unknown content type: {content_type}")
                        continue
                    elif user_input.startswith("/temp "):
                        try:
                            temp = float(user_input.split()[1])
                            if 0.1 <= temp <= 2.0:
                                temperature = temp
                                print(f"Temperature set to: {temp}")
                            else:
                                print("Temperature must be between 0.1 and 2.0")
                        except ValueError:
                            print("Invalid temperature value")
                        continue
                
                # Generate text
                print(f"\n🤖 Generating {self.languages[current_lang]['name']} text...")
                
                prompt = self.create_language_prompt(current_lang, current_type, user_input)
                generated = self.generate_text(
                    prompt, 
                    max_length=150, 
                    temperature=temperature
                )
                
                # Clean up generated text
                generated = re.sub(r'<[^>]+>', '', generated)  # Remove tags
                generated = re.sub(r'\s+', ' ', generated).strip()  # Clean whitespace
                
                print(f"\n📝 Generated text:")
                print(f"   {generated}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def batch_mode(self, lang_code: str, content_type: str, prompts: List[str]):
        """Run batch generation mode."""
        print(f"\n🤖 Batch generation for {self.languages[lang_code]['name']} ({content_type})")
        print("="*50)
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n{i}. Input: {prompt}")
            
            formatted_prompt = self.create_language_prompt(lang_code, content_type, prompt)
            generated = self.generate_text(formatted_prompt, max_length=100)
            
            # Clean up generated text
            generated = re.sub(r'<[^>]+>', '', generated)
            generated = re.sub(r'\s+', ' ', generated).strip()
            
            print(f"   Generated: {generated}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced African LLM Demo")
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/models/enhanced-v1/final",
        help="Path to trained model (default: outputs/models/enhanced-v1/final)"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="tokenization/htf_bpe_16k.model",
        help="Path to tokenizer model (default: tokenization/htf_bpe_16k.model)"
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["am", "ff", "ha", "so", "sw", "yo"],
        help="Language for batch mode"
    )
    parser.add_argument(
        "--content_type",
        type=str,
        choices=["dialogue", "story", "news", "poem", "instruction", "general"],
        default="general",
        help="Content type for batch mode"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        help="Prompts for batch mode"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    model_path = Path(args.model_path)
    tokenizer_path = Path(args.tokenizer_path)
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the enhanced model first:")
        print("python training/scripts/train.py --config training/configs/enhanced.yaml")
        sys.exit(1)
    
    if not tokenizer_path.exists():
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        sys.exit(1)
    
    # Initialize demo
    demo = EnhancedAfricanLLMDemo(str(model_path), str(tokenizer_path))
    
    # Run appropriate mode
    if args.language and args.prompts:
        demo.batch_mode(args.language, args.content_type, args.prompts)
    else:
        demo.interactive_mode()

if __name__ == "__main__":
    main() 