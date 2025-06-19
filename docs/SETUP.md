# Setup Guide

This guide provides detailed instructions for setting up the African LLM project on your local machine.

## System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space
- **GPU**: Optional but recommended for faster training

### Recommended Requirements
- **OS**: Linux (Ubuntu 20.04+) or macOS
- **Python**: 3.9 or higher
- **RAM**: 32GB or more
- **Storage**: 50GB free space
- **GPU**: NVIDIA GPU with 8GB+ VRAM

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/wahab-cide/african_llm_project.git
cd african_llm_project
```

### 2. Set Up Python Environment

#### Option A: Using venv (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

#### Option B: Using conda

```bash
# Create conda environment
conda create -n african_llm python=3.9
conda activate african_llm
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### 4. Install DVC (Data Version Control)

```bash
# Install DVC
pip install dvc

# Initialize DVC
dvc init

# Add remote storage (if using cloud storage)
dvc remote add -d myremote s3://your-bucket/path
# or for Google Cloud Storage:
# dvc remote add -d myremote gs://your-bucket/path
```

### 5. Download Data

```bash
# Pull data files (if using DVC)
dvc pull

# Or manually download data files to data/raw/ directory
```

## Configuration

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
# Copy example environment file
cp .env.example .env

# Edit the file with your settings
nano .env
```

Example `.env` file:
```env
# Weights & Biases
WANDB_API_KEY=your_wandb_api_key
WANDB_PROJECT=african_llm
WANDB_ENTITY=your_username

# Data paths
DATA_DIR=data
PROCESSED_DATA_DIR=data/processed
TOKENIZER_PATH=tokenization/htf_bpe_16k.model

# Training settings
BATCH_SIZE=4
LEARNING_RATE=5e-5
MAX_LENGTH=256
```

### 2. Weights & Biases Setup

1. Create a W&B account at [wandb.ai](https://wandb.ai)
2. Get your API key from your profile settings
3. Add the API key to your `.env` file
4. Login to W&B:
   ```bash
   wandb login
   ```

### 3. GPU Setup (Optional)

#### NVIDIA GPU (CUDA)

```bash
# Install CUDA toolkit (if not already installed)
# Follow instructions at: https://developer.nvidia.com/cuda-downloads

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Apple Silicon (MPS)

```bash
# PyTorch with MPS support is included in requirements.txt
# No additional setup needed for macOS with Apple Silicon
```

## Data Setup

### 1. Directory Structure

Ensure your data directory structure is correct:

```
data/
├── raw/                    # Raw downloaded corpora
│   ├── amharic/
│   ├── fulani/
│   ├── hausa/
│   ├── somali/
│   ├── swahili/
│   └── yoruba/
├── processed/              # Cleaned text files
│   ├── amharic.txt
│   ├── fulani.txt
│   ├── hausa.txt
│   ├── somali.txt
│   ├── swahili.txt
│   └── yoruba.txt
└── augmented/              # Data augmentation outputs
```

### 2. Download Corpora

```bash
# Download African language corpora
python training/scripts/fetch_african_corpora.py

# Or download specific language datasets
python training/scripts/fetch_langs_text.py --language hausa
```

### 3. Process Data

```bash
# Build tokenized dataset
python scripts/build_tokenized_dataset.py \
  --data_dir data/processed \
  --tokenizer_path tokenization/htf_bpe_16k.model \
  --output_dir data/processed/tokenized_dataset \
  --max_length 256 \
  --num_proc 4
```

## Verification

### 1. Test Installation

```bash
# Test Python imports
python -c "import torch; import transformers; import datasets; print('All imports successful!')"

# Test GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### 2. Test Data Pipeline

```bash
# Test tokenizer loading
python -c "import sentencepiece as spm; tokenizer = spm.SentencePieceProcessor(); tokenizer.load('tokenization/htf_bpe_16k.model'); print('Tokenizer loaded successfully!')"
```

### 3. Test Training Script

```bash
# Run a quick training test (small dataset)
python training/scripts/train.py --config training/configs/test.yaml
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
- Reduce batch size in training config
- Use gradient accumulation
- Enable gradient checkpointing

#### 2. Import Errors
- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt --force-reinstall`

#### 3. DVC Issues
- Check DVC installation: `dvc --version`
- Verify remote storage configuration
- Check file permissions

#### 4. W&B Connection Issues
- Verify API key in `.env` file
- Check internet connection
- Try `wandb login` again

### Getting Help

- Check the [GitHub Issues](https://github.com/wahab-cide/african_llm_project/issues)
- Review the [FAQ](FAQ.md)
- Contact the maintainers

## Next Steps

After successful setup:

1. **Review the training configuration** in `training/configs/base.yaml`
2. **Start with a small dataset** for testing
3. **Monitor training** with W&B
4. **Check the [Training Guide](TRAINING.md)** for detailed instructions

---

**Note**: This setup guide assumes you have basic familiarity with Python, Git, and command-line tools. If you encounter issues, please refer to the troubleshooting section or open an issue on GitHub. 