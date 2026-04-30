"""
finetune.py — Fine-tune TinyLlama 1.1B on your custom data
Run: python finetune.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset

# ══════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_FILE = "my_data.jsonl"
OUTPUT_DIR = "out-finetune"
MAX_SEQ_LEN = 256
EPOCHS = 3
# ══════════════════════════════════════════

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print("Loading dataset...")
dataset = load_dataset("json", data_files=DATA_FILE, split="train")
print(f"   {len(dataset)} examples loaded")

# LoRA — trains only 1% of parameters, saves memory
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

print("Starting fine-tuning...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    max_seq_length=MAX_SEQ_LEN,
    dataset_text_field="text",
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=50,
        fp16=False,
        bf16=False,
    ),
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nDone! Model saved to {OUTPUT_DIR}")
print("Next step: python chat_taal.py")
