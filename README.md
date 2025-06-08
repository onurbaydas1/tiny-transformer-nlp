# Tiny Transformer NLP

A minimal transformer-based language model written entirely from scratch in PyTorch — including a custom BPE tokenizer, training loop, and sampling-based text generation.

This project demonstrates the inner workings of a basic NLP model without relying on HuggingFace or any pretrained libraries.

---

## 🧠 Features

- ✅ Custom BPE tokenizer (no dependencies)
- ✅ Tiny transformer model (2 layers, multi-head attention)
- ✅ Pure PyTorch training and inference
- ✅ Text generation with top-k sampling
- ✅ GPU support
- ✅ Ready-to-use trained `.pt` model included

---

## 🚀 Usage

```bash
git clone https://github.com/onurbaydas1/tiny-transformer-nlp.git
cd tiny-transformer-nlp
pip install -r requirements.txt

# Train from scratch
python train.py

# Generate text
python generate.py
