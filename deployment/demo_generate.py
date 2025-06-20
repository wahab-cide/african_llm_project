"""
Demo script for African Language LLM text generation.
Dependencies: transformers>=4.40, torch, sentencepiece
"""

import argparse
import time
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def load_model(model_dir):
    """Load model and tokenizer, return generation pipeline."""
    print(f"[DEBUG] Loading model from {model_dir}...")
    start_time = time.time()
    
    # check if we have a SentencePiece model
    spiece_model_path = os.path.join(model_dir, "spiece.model")
    print(f"[DEBUG] Checking for spiece.model at {spiece_model_path}")
    if os.path.exists(spiece_model_path):
        print("[DEBUG] Found spiece.model, using T5Tokenizer")
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained(model_dir)
    else:
        print("[DEBUG] No spiece.model, using AutoTokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print("[DEBUG] Tokenizer loaded")
    
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
    print("[DEBUG] Model loaded")
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        print("[DEBUG] Adding pad_token to tokenizer")
        tokenizer.pad_token = tokenizer.eos_token
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("[DEBUG] Pipeline created")
    
    load_time = time.time() - start_time
    print(f"Loaded in {load_time:.1f} s")
    return pipe


def interactive_generation(pipe, max_length=80):
    """Run interactive text generation loop."""
    print("Type your prompt in any African language (Amharic, Fulani, Hausa, Somali, Swahili, Yoruba)")
    print("Type 'q' or 'quit' to exit, Ctrl-C to interrupt")
    print("-" * 50)
    
    first_run = True
    
    try:
        while True:
            try:
                prompt = input(">>> ").strip()
                
                if prompt.lower() in ['q', 'quit']:
                    print("Goodbye!")
                    break
                
                if not prompt:
                    continue
                
                # Generate text
                result = pipe(
                    prompt, 
                    max_new_tokens=max_length, 
                    do_sample=True, 
                    top_p=0.95
                )
                
                generated_text = result[0]["generated_text"]
                
                # Print only  continuation 
                continuation = generated_text[len(prompt):].strip()
                if continuation:
                    print(continuation)
                print()
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
                
    except KeyboardInterrupt:
        print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description="Interactive African Language LLM demo")
    parser.add_argument(
        "--model_dir", 
        default="outputs/models/htf-v1-fast/final",
        help="Path to model directory (default: outputs/models/htf-v1-fast/final)"
    )
    parser.add_argument(
        "--length", 
        type=int, 
        default=80,
        help="Maximum number of tokens to generate (default: 80)"
    )
    
    args = parser.parse_args()
    
    # Load model
    pipe = load_model(args.model_dir)
    
    # Start interactive generation
    interactive_generation(pipe, args.length)


if __name__ == "__main__":
    # Usage examples:
    # python deployment/demo_generate.py                          # use default model dir
    # python deployment/demo_generate.py --model_dir path/to/model
    # python deployment/demo_generate.py --length 120             # generate longer text
    main() 