# Training Guide

This guide covers the complete training pipeline for the African Language LLM project, from data preparation to model evaluation.

## Overview

The training pipeline consists of several stages:
1. **Data Collection** - Downloading and preparing corpora
2. **Preprocessing** - Cleaning and normalizing text
3. **Tokenization** - Training a shared multilingual tokenizer
4. **Dataset Creation** - Building tokenized training datasets
5. **Model Training** - Training the language model
6. **Evaluation** - Assessing model performance

## Stage 1: Data Collection

### Available Corpora

The project supports multiple data sources:

- **OSCAR**: Open Super-large Crawled Aggregated coRpus
- **CC-100**: Common Crawl 100 dataset
- **Custom datasets**: Language-specific collections

### Downloading Data

```bash
# Download all African language corpora
python training/scripts/fetch_african_corpora.py

# Download specific language
python training/scripts/fetch_langs_text.py --language hausa

# Download Hausa-specific data
python training/scripts/download_hausa_text.py
```

### Data Sources by Language

| Language | Primary Sources | Size (approx.) |
|----------|----------------|----------------|
| Amharic  | OSCAR, CC-100  | 200MB+         |
| Fulani   | OSCAR          | 10MB+          |
| Hausa    | OSCAR, CC-100  | 100MB+         |
| Somali   | OSCAR, CC-100  | 100MB+         |
| Swahili  | OSCAR, CC-100  | 8MB+           |
| Yoruba   | OSCAR, CC-100  | 26MB+          |

## Stage 2: Data Preprocessing

### Text Cleaning Pipeline

The preprocessing includes:
- Language detection and filtering
- Text normalization
- Duplicate removal
- Length filtering
- Special character handling

### Running Preprocessing

```bash
# Process raw data files
python scripts/preprocess_data.py \
  --input_dir data/raw \
  --output_dir data/processed \
  --min_length 10 \
  --max_length 1000
```

## Stage 3: Tokenization

### Training SentencePiece Tokenizer

```bash
# Train shared BPE tokenizer
python scripts/train_tokenizer.py \
  --input_files data/processed/*.txt \
  --model_prefix tokenization/htf_bpe_16k \
  --vocab_size 16003 \
  --model_type bpe \
  --character_coverage 0.9995
```

### Tokenizer Configuration

- **Vocabulary Size**: 16,003 tokens
- **Model Type**: BPE (Byte Pair Encoding)
- **Character Coverage**: 0.9995
- **Shared Vocabulary**: Across all languages

## Stage 4: Dataset Creation

### Building Tokenized Datasets

```bash
# Create full dataset
python scripts/build_tokenized_dataset.py \
  --data_dir data/processed \
  --tokenizer_path tokenization/htf_bpe_16k.model \
  --output_dir data/processed/tokenized_dataset \
  --max_length 256 \
  --num_proc 4

# Create smaller dataset for testing
python scripts/build_tokenized_dataset.py \
  --data_dir data/processed \
  --tokenizer_path tokenization/htf_bpe_16k.model \
  --output_dir data/processed/tokenized_dataset_small \
  --max_length 256 \
  --num_proc 4 \
  --max_examples_per_file 50000
```

### Dataset Statistics

After creation, you should see statistics like:
```
✅ Tokenized dataset created successfully!
Total examples: 2,400,000
Average sequence length: 128.5
Min/Max sequence length: 10/256
```

## Stage 5: Model Training

### Configuration Files

Training is configured via YAML files in `training/configs/`:

#### Base Configuration (`base.yaml`)

```yaml
model:
  name: "gpt2"
  config:
    vocab_size: 16003
    n_positions: 512
    n_ctx: 512
    n_embd: 768
    n_layer: 12
    n_head: 12

training:
  batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 5e-5
  num_train_epochs: 3
  warmup_steps: 100
  logging_steps: 100
  save_steps: 1000
  eval_steps: 1000
  fp16: false  # Disable for Apple Silicon
  dataloader_num_workers: 4
  remove_unused_columns: false

data:
  dataset_path: "data/processed/tokenized_dataset_small"
  max_length: 256
  text_column: "text"
  label_column: "input_ids"

logging:
  project: "african_llm"
  entity: "your_username"
  run_name: "htf-v1-fast"
```

#### Custom Configurations

Create custom configs for different experiments:

```yaml
# training/configs/large_model.yaml
model:
  name: "gpt2"
  config:
    vocab_size: 16003
    n_positions: 1024
    n_ctx: 1024
    n_embd: 1024
    n_layer: 24
    n_head: 16

training:
  batch_size: 2
  gradient_accumulation_steps: 16
  learning_rate: 3e-5
  num_train_epochs: 5
```

### Training Commands

#### Basic Training

```bash
# Train with default configuration
python training/scripts/train.py

# Train with custom configuration
python training/scripts/train.py --config training/configs/custom.yaml
```

#### Advanced Training Options

```bash
# Resume from checkpoint
python training/scripts/train.py \
  --resume_from_checkpoint outputs/models/checkpoint-1000

# Train with specific dataset
python training/scripts/train.py \
  --dataset_path data/processed/tokenized_dataset_small

# Override configuration parameters
python training/scripts/train.py \
  --learning_rate 1e-4 \
  --batch_size 8 \
  --num_train_epochs 5
```

### Training Monitoring

#### Weights & Biases

Training progress is automatically logged to W&B:

```bash
# View training dashboard
wandb login
# Open browser to https://wandb.ai/your_username/african_llm
```

#### Local Monitoring

```bash
# Start TensorBoard
tensorboard --logdir outputs/logs

# Monitor GPU usage
nvidia-smi -l 1  # For NVIDIA GPUs
```

### Training Optimization

#### Memory Optimization

For limited GPU memory:

```yaml
training:
  batch_size: 2
  gradient_accumulation_steps: 16
  gradient_checkpointing: true
  fp16: true  # If supported
  dataloader_num_workers: 2
```

#### Speed Optimization

For faster training:

```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 4
  dataloader_num_workers: 8
  fp16: true  # If supported
  dataloader_pin_memory: true
```

## Stage 6: Model Evaluation

### Evaluation Metrics

The training automatically tracks:
- **Training Loss**: Cross-entropy loss
- **Validation Loss**: On held-out data
- **Perplexity**: Model uncertainty
- **Learning Rate**: Current learning rate

### Manual Evaluation

```bash
# Evaluate trained model
python evaluation/evaluate_model.py \
  --model_path outputs/models/final \
  --test_data data/processed/test_dataset \
  --output_file evaluation_results.json
```

### Generation Testing

```bash
# Test text generation
python evaluation/test_generation.py \
  --model_path outputs/models/final \
  --prompt "Hello in Swahili:" \
  --max_length 50
```

## Training Tips

### 1. Start Small

Begin with a small dataset and model size:
- Use `tokenized_dataset_small` for initial testing
- Start with 6-layer model instead of 12-layer
- Use shorter sequences (256 tokens)

### 2. Monitor Resources

Keep an eye on:
- GPU memory usage
- Training speed (tokens/second)
- Loss convergence
- Learning rate schedule

### 3. Experiment Tracking

Use W&B for:
- Comparing different configurations
- Tracking hyperparameter sweeps
- Sharing results with team
- Reproducing experiments

### 4. Checkpointing

Save checkpoints regularly:
- Every 1000 steps for small models
- Every 5000 steps for large models
- Keep best checkpoint based on validation loss

## Troubleshooting

### Common Training Issues

#### 1. Out of Memory (OOM)

**Symptoms**: CUDA out of memory error
**Solutions**:
- Reduce batch size
- Increase gradient accumulation steps
- Enable gradient checkpointing
- Use smaller model size

#### 2. Slow Training

**Symptoms**: Low tokens/second
**Solutions**:
- Increase batch size if memory allows
- Use more data loader workers
- Enable mixed precision (fp16)
- Use faster storage (SSD)

#### 3. Loss Not Decreasing

**Symptoms**: Loss plateaus or increases
**Solutions**:
- Check learning rate (might be too high/low)
- Verify data quality
- Check for gradient clipping
- Monitor gradient norms

#### 4. W&B Connection Issues

**Symptoms**: Training continues but no W&B logging
**Solutions**:
- Check internet connection
- Verify W&B API key
- Try `wandb login` again
- Check W&B project settings

### Getting Help

- Check training logs in `outputs/logs/`
- Review W&B run details
- Open an issue on GitHub with:
  - Error messages
  - Configuration files
  - System specifications

## Next Steps

After successful training:

1. **Evaluate model performance** on test data
2. **Test text generation** in different languages
3. **Export model** for deployment
4. **Share results** with the community

---

**Note**: Training times vary significantly based on hardware, dataset size, and model configuration. A small model on a single GPU might take 2-4 hours, while larger models can take days or weeks. 