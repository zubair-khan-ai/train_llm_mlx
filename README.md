Train a GPT-style language model from scratch on the TinyStories dataset using either PyTorch (with MPS acceleration) or Apple's MLX framework.

Overview
This notebook provides a complete pipeline for:

Loading and exploring the TinyStories dataset
Training a BPE tokenizer from scratch
Building a GPT architecture (Transformer decoder)
Training with PyTorch (MPS) or MLX
Text generation with various sampling strategies
Requirements
macOS with Apple Silicon (M1/M2/M3/M4)
Python 3.10+
PyTorch 2.0+ with MPS support
MLX (optional, for MLX training path)
Installation
pip install torch torchvision torchaudio
pip install mlx  # Optional, for MLX training
pip install datasets tokenizers transformers
pip install matplotlib tqdm
Switching Between PyTorch and MLX
The notebook supports both training frameworks. Here's how to switch between them:

Option 1: Automatic Selection (Recommended)
The notebook includes a select_training_framework() function that automatically detects available hardware:

framework = select_training_framework()
# Returns 'mlx' if MLX is available, otherwise 'pytorch'
Option 2: Manual Selection
To force a specific framework, set the USE_MLX variable before training:

# Force PyTorch training (with MPS acceleration)
USE_MLX = False

# Force MLX training
USE_MLX = True
Framework Comparison
Feature	PyTorch (MPS)	MLX
Memory Efficiency	Good	Better (unified memory)
Training Speed	Fast	Comparable
Checkpoint Format	.pt	.safetensors
Ecosystem	Mature	Growing
Usage
Quick Test (Recommended First Run)
Run the "Quick Training Test" cell to validate your setup:

Uses a small data subset (5000 samples)
Trains for 200 steps
Completes in a few minutes
Full Training
Uncomment and run the "Full Training Configuration" cell:

Trains on the full TinyStories dataset
50,000+ steps for meaningful results
Estimated time: 4-8 hours on M4 Max
Text Generation
After training, use the interactive inference cell:

run_interactive_inference(model, tokenizer)
Example prompts:

"Once upon a time, there was a little"
"The brave knight set out on a journey to"
"In a magical forest, a small rabbit named"
Project Structure
.
├── train_llm_from_scratch.ipynb  # Main notebook
├── models/
│   ├── checkpoints/              # Saved model weights
│   └── tokenizer/                # Trained tokenizer
├── data/
│   ├── tinystories/              # Raw dataset (downloaded)
│   └── tokenized/                # Pre-tokenized data
└── utils/                        # Helper utilities
Model Architecture
Type: GPT (decoder-only Transformer)
Layers: 8 Transformer blocks
Attention Heads: 8
Embedding Dimension: 512
Vocabulary Size: 16,384 (BPE)
Context Length: 512 tokens
Parameters: ~34M
License
MIT License
