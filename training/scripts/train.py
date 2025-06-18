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
    PreTrainedTokenizerFast,
)

import wandb


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
    
    # Data
    tokenized_path: str = "data/processed/tokenized_dataset"
    max_seq_len: int = 256
    
    # Training
    batch_size: int = 16
    gradient_accum: int = 2
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
    
    # Flatten nested structure
    flat_config = {}
    for section, values in config_dict.items():
        if isinstance(values, dict):
            flat_config.update(values)
        else:
            flat_config[section] = values
    
    # Create Config instance with loaded values
    return Config(**flat_config)


def init_tokenizer(cfg: Config) -> PreTrainedTokenizerFast:
    """Initialize the tokenizer."""
    tok = PreTrainedTokenizerFast(
        tokenizer_file=str(cfg.tokenizer_path),
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    return tok


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
    
    # Create data collator
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )
    
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
        gradient_accumulation_steps=cfg.gradient_accum,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=cfg.num_epochs,
        fp16=cfg.fp16,
        logging_dir=str(cfg.logging_dir),
        logging_steps=cfg.logging_steps,
        save_strategy="epoch",
        evaluation_strategy="epoch",
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
    
    print(f"✅ Training completed! Model saved to {cfg.output_dir / 'final'}")


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
