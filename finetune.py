"""
finetune.py — Fine-tune TinyLlama 1.1B on your custom data
Run: python finetune.py
"""

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from importlib.metadata import version as pkg_version

# ══════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_FILE = "my_data.jsonl"
OUTPUT_DIR = "out-finetune"
MAX_SEQ_LEN = 256
EPOCHS = 3
# ══════════════════════════════════════════


def _version_tuple(v: str):
    # Keep only numeric parts so we can compare versions without extra deps.
    nums = []
    for part in v.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if digits == "":
            break
        nums.append(int(digits))
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums[:3])


def _check_dependency_compatibility():
    peft_v = pkg_version("peft")
    trl_v = pkg_version("trl")
    transformers_v = pkg_version("transformers")
    print(
        f"Versions -> peft: {peft_v}, trl: {trl_v}, transformers: {transformers_v}")

    # trl 0.8.x can fail with very new transformers where Trainer API changed.
    if _version_tuple(trl_v) < (0, 9, 0) and _version_tuple(transformers_v) >= (4, 52, 0):
        raise RuntimeError(
            "Incompatible packages detected: trl "
            f"{trl_v} with transformers {transformers_v}.\n"
            "Fix by pinning compatible versions:\n"
            "  python -m pip install --upgrade 'transformers==4.41.2' "
            "'trl==0.8.6' 'peft==0.10.0' 'accelerate<1.0'"
        )

    # Newer peft releases expect symbols not available in older transformers.
    if _version_tuple(peft_v) >= (0, 12, 0) and _version_tuple(transformers_v) < (4, 44, 0):
        raise RuntimeError(
            "Incompatible packages detected: peft "
            f"{peft_v} with transformers {transformers_v}.\n"
            "Fix by pinning compatible versions:\n"
            "  python -m pip install --upgrade 'transformers==4.41.2' "
            "'trl==0.8.6' 'peft==0.10.0' 'accelerate<1.0'"
        )


_check_dependency_compatibility()


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
