"""
finetune.py — Fine-tune TinyLlama 1.1B on your custom data
Run: python finetune.py
"""

import argparse
import os

from datasets import load_dataset
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from importlib.metadata import version as pkg_version

# ══════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_FILES = ["my_data.jsonl", "generated_data.jsonl"]
OUTPUT_DIR = "out-finetune"
CONTINUE_FROM_ADAPTER = "out-finetune"
MAX_SEQ_LEN = 256
EPOCHS = 3
# ══════════════════════════════════════════


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune TinyLlama with LoRA")
    parser.add_argument(
        "--data-files",
        default=",".join(DATA_FILES),
        help="Comma-separated JSONL data files.",
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--continue-from", default=CONTINUE_FROM_ADAPTER)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    return parser.parse_args()


args = _parse_args()
DATA_FILES = [p.strip() for p in args.data_files.split(",") if p.strip()]
DATA_FILES = [p for p in DATA_FILES if os.path.exists(p)]
if not DATA_FILES:
    raise RuntimeError(
        "No dataset files found. Provide valid --data-files paths.")


OUTPUT_DIR = args.output_dir
CONTINUE_FROM_ADAPTER = args.continue_from
MAX_SEQ_LEN = args.max_seq_len
EPOCHS = args.epochs


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
tokenizer_source = MODEL_NAME
if CONTINUE_FROM_ADAPTER and os.path.exists(CONTINUE_FROM_ADAPTER):
    tokenizer_source = CONTINUE_FROM_ADAPTER

tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

resume_with_adapter = (
    CONTINUE_FROM_ADAPTER
    and os.path.exists(os.path.join(CONTINUE_FROM_ADAPTER, "adapter_config.json"))
)

if resume_with_adapter:
    print(f"Resuming from adapter: {CONTINUE_FROM_ADAPTER}")
    model = PeftModel.from_pretrained(
        base_model,
        CONTINUE_FROM_ADAPTER,
        is_trainable=True,
    )
else:
    print("No existing adapter found. Starting fresh LoRA fine-tune.")
    model = base_model

print("Loading dataset...")
dataset = load_dataset("json", data_files=DATA_FILES, split="train")
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
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=50,
    fp16=False,
    bf16=False,
)

if resume_with_adapter:
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        args=training_args,
    )
else:
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        args=training_args,
    )

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nDone! Model saved to {OUTPUT_DIR}")
print("Next step: python chat_taal.py")
