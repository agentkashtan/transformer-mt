# Transformer – English to French Translation

This project contains a PyTorch implementation of the Transformer model from [Attention Is All You Need](https://arxiv.org/abs/1706.03762), adapted and trained for English-to-French translation. It includes a full training and inference pipeline using the OPUS Books dataset, with custom byte-pair encoding tokenizers trained from scratch.

## Overview
- Encoder-decoder Transformer model (from "Attention Is All You Need")
- Trained on OPUS Books (en-fr)
- Custom byte-pair encoding (BPE) tokenizers for English and French
- Training and model configuration is handled via `config.py`

## Repository Structure

- `transformer.py` — model definition
- `train.py` — training script
- `inference.py` — translation script
- `tokenizer.py` — tokenizer loading or training logic
- `config.py` — model and training hyperparameters
- `tokenizer_en/` — trained English tokenizer
- `tokenizer_fr/` — trained French tokenizer
- `requirements.txt` — snapshot of the development environment

## Setup

Create a fresh environment (optional, but recommended):

```bash
conda create -n transformer python=3.10
conda activate transformer
```

Install required libraries:

```bash
pip install torch torchvision tokenizers datasets tqdm pyyaml
```
⚠️Make sure you have a working PyTorch installation with CUDA support. You can install the appropriate version from: https://pytorch.org/get-started/locally/

## Usage

### Training

To start training from scratch:

```bash
python train.py
```
- Training parameters can be modified in config.py
- Saves a checkpoint every 5 epochs

### Inference

To run inference with a trained model checkpoint:

```bash
python inference.py path/to/checkpoint.pt
```
This will generate translations using the saved model and tokenizers.

## Training Details

- Training was done on a single RTX 4090 GPU.
- Total training time: ~15 hours for 30 epochs (~30 minutes per epoch).
- Batch size: 16 (used ~18 GB of VRAM including model and data).
- If you have more or less GPU memory, you can adjust the `batch_size` parameter in `config.py` accordingly.

## Notes

- Tokenizers for English and French are already included in the repository (`tokenizer_en/` and `tokenizer_fr/`). You do not need to download or prepare them manually.
- If these folders are missing, running `train.py` will automatically train new BPE tokenizers from the dataset and save them.
- You can also use your own tokenizers by placing them in the same folder structure. If you do, make sure to update `src_len`, `tgt_len`, and `vocab_size` in `config.py` to match your tokenizer’s sequence length and vocabulary size.
