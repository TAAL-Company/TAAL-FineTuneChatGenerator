# TAAL — Fine-tuned TinyLlama 1.1B

TAAL is a project for fine-tuning [TinyLlama 1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) on custom conversation data using LoRA (Low-Rank Adaptation), then chatting with the result — entirely on CPU. It also includes a from-scratch GPT-style model that can be trained on any text dataset.

---

## Table of Contents

- [TAAL — Fine-tuned TinyLlama 1.1B](#taal--fine-tuned-tinyllama-11b)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Project Structure](#project-structure)
  - [Requirements](#requirements)
  - [Quickstart: Fine-tune \& Chat (TinyLlama)](#quickstart-fine-tune--chat-tinyllama)
    - [1. Install dependencies](#1-install-dependencies)
    - [2. Prepare your training data](#2-prepare-your-training-data)
    - [3. Fine-tune](#3-fine-tune)
    - [4. Chat](#4-chat)
  - [From-Scratch GPT Training](#from-scratch-gpt-training)
    - [1. Prepare data](#1-prepare-data)
    - [2. Train](#2-train)
  - [Generate Data And Fine-Tune (One Command)](#generate-data-and-fine-tune-one-command)
  - [Configuration Reference](#configuration-reference)
    - [`finetune.py`](#finetunepy)
    - [`data_finetune_pipeline.py`](#data_finetune_pipelinepy)
    - [`chat_taal.py`](#chat_taalpy)
    - [`train.py` (from-scratch GPT)](#trainpy-from-scratch-gpt)
    - [`prepare_data.py`](#prepare_datapy)
  - [Output Files](#output-files)
  - [Improving Your Model](#improving-your-model)
  - [Base Model](#base-model)

---

## Project Overview

| Track | What it does |
|---|---|
| **Fine-tune track** | Adapts TinyLlama 1.1B to your own Q&A data with LoRA — only ~1% of parameters are trained, keeping memory usage low. |
| **From-scratch track** | Trains a small GPT (≈3M params) on raw text (e.g. TinyStories) using a character-level tokenizer. |

Both tracks run on a standard CPU. A GPU will make training significantly faster if available.

---

## Project Structure

```
TAAL-finetune - chat/
├── finetune.py       ← Fine-tune TinyLlama on your JSONL data (LoRA)
├── chat_taal.py      ← Interactive chat with the fine-tuned model
├── model.py          ← Minimal GPT implementation (from-scratch track)
├── train.py          ← Train the from-scratch GPT model
├── prepare_data.py   ← Download & tokenize TinyStories dataset
├── my_data.jsonl     ← Your training data (Human/Assistant pairs)
├── out-finetune/     ← LoRA adapter weights saved after fine-tuning
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   ├── checkpoint-50/
│   └── checkpoint-75/
└── README.md         ← This file
```

---

## Requirements

- Python 3.9+
- ~4 GB free RAM (8 GB recommended)
- ~3 GB free disk space (for the base model download)

---

## Quickstart: Fine-tune & Chat (TinyLlama)

### 1. Install dependencies

```bash
pip install transformers datasets "trl==0.8.6" peft accelerate torch
```

### 2. Prepare your training data

Edit `my_data.jsonl`. Each line must be a JSON object with a `"text"` field using the Human/Assistant format:

```jsonl
{"text": "### Human: Who are you?\n### Assistant: I am TAAL, an AI assistant."}
{"text": "### Human: What is Python?\n### Assistant: Python is a versatile programming language used for AI, web dev, and more."}
```

The included `my_data.jsonl` already has 100 example pairs covering topics like AI, programming, and general knowledge.

### 3. Fine-tune

```bash
set PYTHONUTF8=1
python finetune.py
```

- Downloads TinyLlama (~2.2 GB on first run)
- Trains for 3 epochs with LoRA (r=8, alpha=16)
- Saves checkpoints to `out-finetune/` every 50 steps
- Typical loss: ~2.4 → ~1.4
- Estimated time: ~20 minutes on CPU

### 4. Chat

```bash
python chat_taal.py
```

An interactive prompt will appear. Type your message and press Enter. Press `Ctrl+C` to quit.

```
==================================================
  TAAL 1.1B
  Type your message, Ctrl+C to quit
==================================================

You: What is machine learning?

TAAL: Machine learning is a branch of AI where computers learn patterns
from data instead of being explicitly programmed...
```

---

## Generate Data And Fine-Tune (One Command)

Use `data_finetune_pipeline.py` to generate new QA examples and optionally start fine-tuning immediately.

### Template mode (offline)

```bash
python data_finetune_pipeline.py --mode template --count 120 --run-finetune
```

### Azure mode (AI-generated data)

Set server-side environment variables first (do not hardcode keys in frontend code):

```bash
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com"
export AZURE_OPENAI_API_KEY="<your-key>"
export AZURE_OPENAI_DEPLOYMENT="<your-chat-deployment>"
export AZURE_OPENAI_API_VERSION="2025-01-01-preview"
```

Then run:

```bash
python data_finetune_pipeline.py --mode azure --count 150 --topics "AI,Python,Azure" --run-finetune
```

This script:
- Generates unique JSONL examples
- Avoids duplicates based on question text
- Combines `my_data.jsonl` + generated file
- Continues fine-tuning from `out-finetune/` when available

---

## From-Scratch GPT Training

This track trains a small GPT model entirely from scratch on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset using a character-level tokenizer.

### 1. Prepare data

```bash
python prepare_data.py
```

- Downloads 5,000 stories from TinyStories (configurable via `NUM_STORIES`)
- Builds a character-level vocabulary
- Saves tokenized `train.bin`, `val.bin`, and `meta.pkl` to `data/chat/`

### 2. Train

```bash
python train.py
```

- Trains a GPT with 4 layers, 4 heads, 256-dim embeddings (~3M parameters)
- Runs for 1,000 iterations by default
- Evaluates on the validation set every 100 iterations
- Saves the model to `out-chat/`

---

## Configuration Reference

### `finetune.py`

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | HuggingFace base model |
| `DATA_FILES` | `['my_data.jsonl']` | Training data file list |
| `OUTPUT_DIR` | `out-finetune` | Where to save the adapter |
| `CONTINUE_FROM_ADAPTER` | `out-finetune` | Existing adapter path for continued fine-tuning |
| `MAX_SEQ_LEN` | `256` | Maximum token sequence length |
| `EPOCHS` | `3` | Number of training passes |

You can override these at runtime with env vars:
- `TAAL_DATA_FILES` (comma-separated)
- `TAAL_OUTPUT_DIR`
- `TAAL_CONTINUE_FROM_ADAPTER`
- `TAAL_MAX_SEQ_LEN`
- `TAAL_EPOCHS`

### `data_finetune_pipeline.py`

| Option | Example | Description |
|---|---|---|
| `--mode` | `template` or `azure` | Data generation source |
| `--count` | `120` | Number of examples to generate |
| `--topics` | `"AI,Python,Azure"` | Topic list for generation |
| `--run-finetune` | flag | Launches `finetune.py` after generation |
| `--epochs` | `3` | Epochs passed to finetune |
| `--output-dir` | `out-finetune` | Adapter output directory |
| `--continue-from` | `out-finetune` | Adapter checkpoint to continue from |

**LoRA config:** `r=8`, `lora_alpha=16`, `lora_dropout=0.05`

### `chat_taal.py`

| Variable | Default | Description |
|---|---|---|
| `BASE_MODEL` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Base model to load |
| `FINETUNED` | `out-finetune` | Path to the LoRA adapter |
| `MAX_TOKENS` | `200` | Max tokens to generate per reply |
| `TEMPERATURE` | `0.7` | Sampling temperature (lower = more focused) |

### `train.py` (from-scratch GPT)

| Variable | Default | Description |
|---|---|---|
| `max_iters` | `1000` | Total training iterations |
| `n_layer` | `4` | Number of transformer layers |
| `n_head` | `4` | Number of attention heads |
| `n_embd` | `256` | Embedding dimension |
| `block_size` | `64` | Context window size (tokens) |
| `learning_rate` | `1e-3` | Peak learning rate |
| `batch_size` | `8` | Training batch size |
| `device` | `cpu` | Set to `cuda` if you have a GPU |

### `prepare_data.py`

| Variable | Default | Description |
|---|---|---|
| `NUM_STORIES` | `5000` | Number of TinyStories examples to use |
| `OUTPUT_DIR` | `data/chat` | Where to save tokenized data |

---

## Output Files

After fine-tuning, `out-finetune/` contains:

| File | Description |
|---|---|
| `adapter_config.json` | LoRA adapter hyperparameters |
| `adapter_model.safetensors` | Trained LoRA weights |
| `tokenizer.json` | Tokenizer vocabulary and rules |
| `tokenizer_config.json` | Tokenizer settings |
| `checkpoint-50/` | Intermediate checkpoint at step 50 |
| `checkpoint-75/` | Intermediate checkpoint at step 75 |

---

## Improving Your Model

- **More data** — Add more examples to `my_data.jsonl`. Quality matters more than quantity. Aim for 200–500 diverse, well-written pairs.
- **More epochs** — Increase `EPOCHS` in `finetune.py` (try 5–10 for small datasets).
- **Focused answers** — Lower `TEMPERATURE` in `chat_taal.py` (e.g. `0.5`) to reduce randomness.
- **Longer responses** — Increase `MAX_TOKENS` in `chat_taal.py`.
- **Larger LoRA rank** — Increase `r` in `finetune.py`'s `LoraConfig` (e.g. `r=16`) for more expressive adapters at the cost of more memory.

---

## Base Model

**TinyLlama/TinyLlama-1.1B-Chat-v1.0**
- 1.1 billion parameters
- License: Apache 2.0
- https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
