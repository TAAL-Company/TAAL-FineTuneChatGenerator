"""
chat_taal.py — Chat with your fine-tuned TinyLlama
Run: python chat_taal.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FINETUNED = "out-finetune"
DEVICE = "cpu"
MAX_TOKENS = 200
TEMPERATURE = 0.7

print("Loading your TAAL model...")
tokenizer = AutoTokenizer.from_pretrained(FINETUNED)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(model, FINETUNED)
model.eval().to(DEVICE)
print("Ready!\n")

print("=" * 50)
print("  TAAL 1.1B")
print("  Type your message, Ctrl+C to quit")
print("=" * 50)

while True:
    try:
        user = input("\nYou: ").strip()
        if not user:
            continue

        prompt = f"### Human: {user}\n### Assistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = response.split("### Assistant:")[-1].strip()
        print(f"\nTAAL: {answer}")

    except KeyboardInterrupt:
        print("\n\nBye!")
        break
