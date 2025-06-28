# African Language LLM Project

A comprehensive pipeline for training multilingual language models on African languages including Amharic, Fulani, Hausa, Somali, Swahili, and Yoruba.

## 🌍 Project Overview

This project aims to develop and train language models specifically for African languages, addressing the underrepresentation of these languages in current NLP research and applications. The pipeline includes data collection, preprocessing, tokenization, and model training with comprehensive logging and evaluation.

### Supported Languages
- **Amharic** (am) - Ethiopian Semitic language
- **Fulani** (af) - Niger-Congo language family
- **Hausa** (ha) - Chadic language family
- **Somali** (so) - Cushitic language family
- **Swahili** (sw) - Bantu language family
- **Yoruba** (yo) - Niger-Congo language family

## 🏗️ Project Structure

```
african_llm_project/
├── data/                          # Data directory
│   ├── raw/                      # Raw downloaded corpora
│   ├── processed/                # Cleaned and processed text files
│   └── augmented/                # Data augmentation outputs
├── scripts/                      # Data processing scripts
│   └── build_tokenized_dataset.py
├── training/                     # Training pipeline
│   ├── configs/                  # Training configurations
│   ├── scripts/                  # Training and data collection scripts
│   └── logs/                     # Training logs
├── tokenization/                 # Tokenizer models and vocabularies
├── evaluation/                   # Model evaluation scripts
├── deployment/                   # Model deployment utilities
├── docs/                         # Additional documentation
└── outputs/                      # Trained models and checkpoints
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git
- DVC (for data versioning)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/wahab-cide/african_llm_project.git
   cd african_llm_project
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up DVC (for data management)**
   ```bash
   dvc pull  # Download data files
   ```

## 📊 Data Pipeline

### 1. Data Collection

The project collects text data from various sources:
- **OSCAR**: Open Super-large Crawled Aggregated coRpus
- **CC-100**: Common Crawl 100 dataset
- **Custom datasets**: Language-specific corpora

### 2. Data Preprocessing

Raw text files are cleaned and processed:
- Text normalization
- Language detection and filtering
- Duplicate removal
- Length filtering

### 3. Tokenization

A shared SentencePiece BPE tokenizer is trained on all languages:
- Vocabulary size: 16,003 tokens
- Shared vocabulary across all languages
- Handles multilingual text effectively

### 4. Dataset Creation

Tokenized datasets are created for training:
- Streaming processing for large files
- Multiprocessing support
- Configurable sequence lengths

## 🎯 Training Pipeline

### Configuration

Training is configured via YAML files in `training/configs/`:

```yaml
# training/configs/base.yaml
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
```

### Training Commands

1. **Train with default configuration**
   ```bash
   python training/scripts/train.py
   ```

2. **Train with custom configuration**
   ```bash
   python training/scripts/train.py --config training/configs/custom.yaml
   ```

3. **Resume training from checkpoint**
   ```bash
   python training/scripts/train.py --resume_from_checkpoint outputs/models/checkpoint-1000
   ```

### Monitoring

Training progress is logged using:
- **Weights & Biases**: Experiment tracking and visualization
- **TensorBoard**: Local training metrics
- **Console output**: Real-time progress updates

## 🔧 Scripts Reference

### Data Processing

- `scripts/build_tokenized_dataset.py`: Create tokenized datasets from processed text
  ```bash
  python scripts/build_tokenized_dataset.py \
    --data_dir data/processed \
    --tokenizer_path tokenization/htf_bpe_16k.model \
    --output_dir data/processed/tokenized_dataset \
    --max_length 256 \
    --num_proc 4
  ```

### Training

- `training/scripts/train.py`: Main training script
- `training/scripts/fetch_african_corpora.py`: Download African language corpora
- `training/scripts/fetch_langs_text.py`: Fetch language-specific datasets

## 📈 Model Performance

### Training Metrics

The model training includes:
- **Loss**: Cross-entropy loss on next token prediction
- **Perplexity**: Model's uncertainty in predictions
- **Learning rate**: Dynamic learning rate scheduling
- **Gradient norms**: Training stability monitoring

### Evaluation Metrics

- **Perplexity**: On held-out validation data
- **BLEU Score**: For text generation quality
- **Language-specific metrics**: Custom evaluation for each language

## 🚀 Deployment

### Model Export

Trained models can be exported for deployment:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load trained model
model = AutoModelForCausalLM.from_pretrained("outputs/models/final")
tokenizer = AutoTokenizer.from_pretrained("outputs/models/final")

# Save for deployment
model.save_pretrained("deployment/model")
tokenizer.save_pretrained("deployment/model")
```

### Inference

```python
from transformers import pipeline

# Create text generation pipeline
generator = pipeline("text-generation", model="deployment/model")

# Generate text in different languages
text = generator("Hello in Swahili:", max_length=50)
```

##  Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Add tests** (if applicable)
5. **Commit your changes**
   ```bash
   git commit -m "Add feature: description"
   ```
6. **Push to your branch**
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to functions and classes
- Update documentation for new features
- Test your changes thoroughly

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- **OSCAR Dataset**: For providing multilingual text corpora
- **Hugging Face**: For the transformers library and model hub
- **Weights & Biases**: For experiment tracking
- **African NLP Community**: For inspiration and support


## 📚 Additional Resources

- [African NLP Workshop](https://africanlpworkshop.org/)
- [Masakhane Project](https://www.masakhane.io/)
- [Hugging Face African Language Models](https://huggingface.co/models?search=african)

