"""
train.py — Train your LLM from scratch
No nanoGPT needed. Just run:

    python train.py

Make sure you've run prepare_data.py first!
"""

import os
import time
import math
import pickle
import numpy as np
import torch
from model import GPT, GPTConfig

# ══════════════════════════════════════════
#  CONFIG — tweak these
# ══════════════════════════════════════════
out_dir       = 'out-chat'
data_dir      = 'data/chat'

# Training
max_iters     = 1000
eval_interval = 100
eval_iters    = 50
log_interval  = 10

# Model size (~3M params, CPU-friendly)
n_layer       = 4
n_head        = 4
n_embd        = 256
block_size    = 64
dropout       = 0.2

# Optimizer
batch_size    = 8        # reduce to 4 if slow
learning_rate = 1e-3
min_lr        = 1e-4
warmup_iters  = 100
lr_decay_iters = 1000
beta1, beta2  = 0.9, 0.99

device        = 'cpu'   # change to 'cuda' if you have a GPU
# ══════════════════════════════════════════

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)

# ── Load data ─────────────────────────────
def load_data(split):
    path = os.path.join(data_dir, f'{split}.bin')
    data = np.fromfile(path, dtype=np.uint16)
    return torch.from_numpy(data.astype(np.int64))

train_data = load_data('train')
val_data   = load_data('val')

with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']
print(f"Vocab size: {vocab_size} | Train tokens: {len(train_data):,}")

# ── Batch sampler ─────────────────────────
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x  = torch.stack([data[i:i+block_size]   for i in ix])
    y  = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ── Model ─────────────────────────────────
config = GPTConfig(
    block_size=block_size,
    vocab_size=vocab_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
)
model = GPT(config).to(device)

# ── Optimizer ─────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2))

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    return min_lr + 0.5 * (learning_rate - min_lr) * (1 + math.cos(math.pi * decay))

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}
    for split in ['train', 'val']:
        L = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            L[k] = loss.item()
        losses[split] = L.mean()
    model.train()
    return losses

# ── Training loop ─────────────────────────
print("\n🚀 Training started!\n")
best_val_loss = float('inf')
t0 = time.time()

for iter in range(max_iters + 1):
    lr = get_lr(iter)
    for g in optimizer.param_groups:
        g['lr'] = lr

    # Eval
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter:4d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            ckpt = {
                'model': model.state_dict(),
                'config': vars(config),
                'iter': iter,
                'val_loss': best_val_loss,
            }
            torch.save(ckpt, os.path.join(out_dir, 'ckpt.pt'))
            print(f"  💾 Checkpoint saved (val loss: {best_val_loss:.4f})")

    if iter == max_iters:
        break

    X, Y = get_batch('train')
    _, loss = model(X, Y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if iter % log_interval == 0:
        dt = (time.time() - t0) * 1000
        print(f"  iter {iter}: loss {loss.item():.4f}, time {dt:.0f}ms")
        t0 = time.time()

print(f"\n✅ Training complete! Model saved to {out_dir}/ckpt.pt")
print(f"   Best val loss: {best_val_loss:.4f}")
print(f"\nNext step → run: python chat.py")
