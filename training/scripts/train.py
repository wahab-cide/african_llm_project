import os
import random
import argparse
import yaml
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import sentencepiece as spm

import wandb


class SentencePieceTokenizerWrapper:
    """Wrapper for SentencePiece tokenizer to work with transformers."""
    
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        
        # Set special tokens
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        
        # Get token IDs
        self.bos_token_id = self.sp.piece_to_id(self.bos_token)
        self.eos_token_id = self.sp.piece_to_id(self.eos_token)
        self.unk_token_id = self.sp.piece_to_id(self.unk_token)
        self.pad_token_id = self.sp.piece_to_id(self.pad_token)
        
        self.vocab_size = self.sp.get_piece_size()
    
    def encode(self, text: str, **kwargs) -> list:
        """Encode text to token IDs."""
        return self.sp.encode_as_ids(text)
    
    def decode(self, token_ids: list, **kwargs) -> str:
        """Decode token IDs to text."""
        return self.sp.decode_ids(token_ids)
    
    def convert_tokens_to_ids(self, tokens: list) -> list:
        """Convert tokens to IDs."""
        return [self.sp.piece_to_id(token) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: list) -> list:
        """Convert IDs to tokens."""
        return [self.sp.id_to_piece(id) for id in ids]
    
    def save_pretrained(self, output_dir: str):
        """Save the tokenizer to the output directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the SentencePiece model
        model_path = os.path.join(output_dir, "spiece.model")
        with open(model_path, "wb") as f:
            f.write(self.sp.serialized_model_proto())
        
        # Create a minimal tokenizer config
        config = {
            "model_max_length": 256,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "special_tokens_map_file": None
        }
        
        import json
        config_path = os.path.join(output_dir, "tokenizer_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)


@dataclass
class Config:
    """Configuration class that can be populated from YAML."""
    
    # Model
    vocab_size: int = 16003
    n_embd: int = 384
    n_layer: int = 4
    n_head: int = 6
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 3
    n_positions: int = 512
    n_ctx: int = 512
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    
    # Data
    tokenized_path: str = "data/processed/tokenized_dataset"
    max_seq_len: int = 256
    
    # Training
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    lr: float = 2e-4
    warmup_ratio: float = 0.06
    fp16: bool = True
    num_epochs: int = 5
    logging_steps: int = 50
    
    # Logging
    project: str = "african-llm"
    run_id: str = "htf-v1"
    
    # Paths
    base_dir: Path = Path("outputs")
    tokenizer_path: Path = Path("tokenization/htf_bpe_16k.model")
    
    @property
    def output_dir(self) -> Path:
        return self.base_dir / "models" / self.run_id
    
    @property
    def logging_dir(self) -> Path:
        return self.base_dir / "logs" / self.run_id


def load_config_from_yaml(config_path: Path) -> Config:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    flat_config = {}
    for section, values in config_dict.items():
        if section == "model" and isinstance(values, dict):
            for k, v in values.items():
                if k == "name":
                    continue
                if k == "config" and isinstance(v, dict):
                    flat_config.update(v)
                else:
                    flat_config[k] = v
        elif isinstance(values, dict):
            flat_config.update(values)
        else:
            flat_config[section] = values
    
    # Convert string values to appropriate types
    if 'lr' in flat_config and isinstance(flat_config['lr'], str):
        flat_config['lr'] = float(flat_config['lr'])
    if 'warmup_ratio' in flat_config and isinstance(flat_config['warmup_ratio'], str):
        flat_config['warmup_ratio'] = float(flat_config['warmup_ratio'])
    if 'num_epochs' in flat_config and isinstance(flat_config['num_epochs'], str):
        flat_config['num_epochs'] = int(flat_config['num_epochs'])
    if 'logging_steps' in flat_config and isinstance(flat_config['logging_steps'], str):
        flat_config['logging_steps'] = int(flat_config['logging_steps'])
    if 'max_seq_len' in flat_config and isinstance(flat_config['max_seq_len'], str):
        flat_config['max_seq_len'] = int(flat_config['max_seq_len'])
    
    return Config(**flat_config)


def init_tokenizer(cfg: Config) -> SentencePieceTokenizerWrapper:
    """Initialize the tokenizer."""
    tokenizer = SentencePieceTokenizerWrapper(str(cfg.tokenizer_path))
    print(f"Loaded tokenizer with vocab size: {tokenizer.vocab_size}")
    return tokenizer


def init_model(cfg: Config) -> GPT2LMHeadModel:
    """Initialize the model."""
    model_cfg = GPT2Config(
        vocab_size=cfg.vocab_size,
        n_embd=cfg.n_embd,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        bos_token_id=cfg.bos_token_id,
        eos_token_id=cfg.eos_token_id,
        pad_token_id=cfg.pad_token_id,
    )
    return GPT2LMHeadModel(model_cfg)


def load_data(cfg: Config):
    """Load the tokenized dataset."""
    data_path = Path(cfg.tokenized_path)
    assert data_path.exists(), f"Tokenized dataset not found at {data_path}!"
    
    ds = load_from_disk(str(data_path))
    ds = ds.shuffle(seed=42)
    return ds.train_test_split(test_size=0.1)


class CustomDataCollator:
    """Custom data collator for our tokenized dataset."""
    
    def __init__(self, tokenizer: SentencePieceTokenizerWrapper, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, examples):
        # Extract input_ids from examples
        input_ids = [ex["input_ids"] for ex in examples]
        
        # Pad sequences to max_length
        padded_ids = []
        attention_masks = []
        
        for ids in input_ids:
            if len(ids) > self.max_length:
                ids = ids[:self.max_length]
            
            # Pad with pad_token_id
            padding_length = self.max_length - len(ids)
            padded_ids.append(ids + [self.tokenizer.pad_token_id] * padding_length)
            attention_masks.append([1] * len(ids) + [0] * padding_length)
        
        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_ids, dtype=torch.long)
        }


def train(cfg: Config):
    """Main training function."""
    
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Initialize components
    tokenizer = init_tokenizer(cfg)
    model = init_model(cfg)
    data = load_data(cfg)
    
    # Create custom data collator
    collator = CustomDataCollator(tokenizer, max_length=cfg.max_seq_len)
    
    # Initialize wandb
    wandb.init(
        project=cfg.project,
        config=cfg.__dict__,
        name=cfg.run_id
    )
    
    # Create output directories
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.logging_dir.mkdir(parents=True, exist_ok=True)
    
    # Training arguments
    args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=cfg.num_epochs,
        fp16=cfg.fp16,
        logging_dir=str(cfg.logging_dir),
        logging_steps=cfg.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch",
        seed=42,
        report_to=["wandb"],
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=data["train"],
        eval_dataset=data["test"],
    )
    
    # Train
    print(f"Starting training with config: {cfg}")
    trainer.train()
    trainer.save_model(cfg.output_dir / "final")
    
    print(f" Training completed! Model saved to {cfg.output_dir / 'final'}")


def main():
    parser = argparse.ArgumentParser(description="Train African language model")
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/base.yaml",
        help="Path to YAML configuration file (default: training/configs/base.yaml)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file {config_path} not found")
        return
    
    cfg = load_config_from_yaml(config_path)
    
    # Set environment variable
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Start training
    train(cfg)


if __name__ == "__main__":
    main()
