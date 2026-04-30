"""
prepare_data.py — Download and tokenize the training dataset
Run this ONCE before training:

    python prepare_data.py
"""

import os
import pickle
import numpy as np
from datasets import load_dataset

# ══════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════
NUM_STORIES = 5000        # increase for better model (10000, 50000...)
OUTPUT_DIR  = 'data/chat'
# ══════════════════════════════════════════

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("📥 Downloading TinyStories dataset...")
ds = load_dataset('roneneldan/TinyStories', split=f'train[:{NUM_STORIES}]')

input_file = os.path.join(OUTPUT_DIR, 'input.txt')
print(f"💾 Saving {NUM_STORIES} stories...")
with open(input_file, 'w', encoding='utf-8') as f:
    for item in ds:
        f.write(item['text'] + '\n')

print("🔤 Tokenizing...")
with open(input_file, 'r', encoding='utf-8') as f:
    data = f.read()

chars    = sorted(list(set(data)))
vocab_size = len(chars)
stoi     = {ch: i for i, ch in enumerate(chars)}
itos     = {i: ch for i, ch in enumerate(chars)}

n         = int(len(data) * 0.9)
train_ids = [stoi[c] for c in data[:n]]
val_ids   = [stoi[c] for c in data[n:]]

np.array(train_ids, dtype=np.uint16).tofile(os.path.join(OUTPUT_DIR, 'train.bin'))
np.array(val_ids,   dtype=np.uint16).tofile(os.path.join(OUTPUT_DIR, 'val.bin'))

with open(os.path.join(OUTPUT_DIR, 'meta.pkl'), 'wb') as f:
    pickle.dump({'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}, f)

print(f"\n✅ Done!")
print(f"   Vocab size   : {vocab_size}")
print(f"   Train tokens : {len(train_ids):,}")
print(f"   Val tokens   : {len(val_ids):,}")
print(f"\nNext step → run: python train.py")
