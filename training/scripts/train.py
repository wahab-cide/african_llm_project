
import os, random
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

#  1. Static configuration
@dataclass
class Config:
    # Model
    vocab_size: int = 16003          
    n_embd: int = 384
    n_layer: int = 4
    n_head: int = 6                  

    # Training
    batch_size: int = 16             
    lr: float = 2e-4
    warmup_ratio: float = 0.06
    gradient_accum: int = 2

    # Paths
    run_name: str = "african-gpt-mini"
    base_dir: Path = Path("outputs")
    data_path: Path = Path("data/processed/tokenized_dataset")
    tokenizer_path: Path = Path("tokenization/htf_bpe_16k.model")

    @property
    def output_dir(self) -> Path:
        return self.base_dir / "models" / self.run_name

    @property
    def logging_dir(self) -> Path:
        return self.base_dir / "logs" / self.run_name


# 2. Init helpers
def init_tokenizer(cfg: Config) -> PreTrainedTokenizerFast:
    tok = PreTrainedTokenizerFast(
        tokenizer_file=str(cfg.tokenizer_path),
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    return tok


def init_model(cfg: Config) -> GPT2LMHeadModel:
    model_cfg = GPT2Config(
        vocab_size=cfg.vocab_size,
        n_embd=cfg.n_embd,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
    )
    return GPT2LMHeadModel(model_cfg)


def load_data(cfg: Config):
    assert cfg.data_path.exists(), "Tokenized dataset is missing!"
    ds = load_from_disk(cfg.data_path)
    ds = ds.shuffle(seed=42)
    return ds.train_test_split(test_size=0.1)


#  3. Training loop 
def train(cfg: Config):
 
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    tokenizer = init_tokenizer(cfg)
    model = init_model(cfg)

    data = load_data(cfg)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )

    wandb.init(project="african-llm", config=cfg.__dict__, name=cfg.run_name)

    args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accum,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=5,
        fp16=True,
        logging_dir=str(cfg.logging_dir),
        logging_steps=100,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        seed=42,
        report_to=["wandb"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=data["train"],
        eval_dataset=data["test"],
    )

    trainer.train()
    trainer.save_model(cfg.output_dir / "final")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false" 
    train(Config())
