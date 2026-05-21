"""
Generate training data and launch fine-tuning in one command.

Examples:
  python data_finetune_pipeline.py --mode template --count 120 --run-finetune
  python data_finetune_pipeline.py --mode azure --count 150 --topics "AI,Python,Azure" --run-finetune

Azure mode expects these environment variables:
    Use CLI args instead:
    --azure-endpoint
    --azure-api-key
    --azure-deployment
    --azure-api-version
"""

import argparse
import datetime
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib import error, request


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT_DIR / "generated_data.jsonl"
DEFAULT_BASE_DATA = ROOT_DIR / "my_data.jsonl"


DEFAULT_TOPICS = [
    "General knowledge",
    "World geography",
    "History basics",
    "Science facts",
    "Health and wellness",
    "Daily life tips",
    "Personal finance basics",
    "Education and learning",
    "Travel and culture",
    "Communication skills",
]

TEMPLATE_PROMPTS = [
    "What is {topic}?",
    "Explain {topic} simply.",
    "How can I start learning {topic}?",
    "Give me a practical example of {topic}.",
    "What are common mistakes in {topic}?",
]

TEMPLATE_ANSWERS = [
    "{topic} is an important concept in modern software and AI. Start with core vocabulary, then practice with small projects every day.",
    "A good way to learn {topic} is to combine theory and hands-on work: read one short guide, then build one tiny exercise immediately.",
    "For {topic}, focus on fundamentals first, then repeat through practice. Consistency beats intensity when building real skill.",
    "Use {topic} in a real workflow: define a simple goal, implement it step by step, and review what worked and what failed.",
    "With {topic}, avoid overcomplicating early. Keep your first version small, measurable, and easy to improve.",
]


@dataclass
class Example:
    human: str
    assistant: str

    def as_record(self):
        return {"text": f"### Human: {self.human}\\n### Assistant: {self.assistant}"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate JSONL data and run fine-tuning.")
    parser.add_argument(
        "--mode", choices=["template", "azure"], default="template")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of examples to generate.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Generated JSONL output path.")
    parser.add_argument(
        "--base-data",
        default=str(DEFAULT_BASE_DATA),
        help="Existing dataset JSONL to combine with generated data.",
    )
    parser.add_argument(
        "--update-base-data",
        action="store_true",
        help="Append generated examples into --base-data (with backup).",
    )
    parser.add_argument(
        "--backup-dir",
        default="dataset_backups",
        help="Directory for base dataset backups before update.",
    )
    parser.add_argument(
        "--topics",
        default=",".join(DEFAULT_TOPICS),
        help="Comma-separated topics for generation.",
    )
    parser.add_argument("--run-finetune", action="store_true",
                        help="Run finetune.py after generation.")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Epochs for finetune run.")
    parser.add_argument(
        "--output-dir",
        default="out-finetune",
        help="Output adapter directory passed to finetune.py.",
    )
    parser.add_argument(
        "--continue-from",
        default="out-finetune",
        help="Existing adapter directory for continued fine-tuning.",
    )
    parser.add_argument("--azure-endpoint", default="")
    parser.add_argument("--azure-api-key", default="")
    parser.add_argument("--azure-deployment", default="")
    parser.add_argument("--azure-api-version", default="2025-01-01-preview")
    return parser.parse_args()


def read_existing_prompts(paths):
    prompts = set()
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text", "")
                marker = "### Human:"
                if marker in text:
                    part = text.split("### Assistant:")[0].replace(
                        marker, "").strip().lower()
                    if part:
                        prompts.add(part)
    return prompts


def generate_template_examples(count, topics, existing_prompts):
    results = []
    guard = 0
    while len(results) < count and guard < count * 20:
        guard += 1
        topic = random.choice(topics)
        q = random.choice(TEMPLATE_PROMPTS).format(topic=topic)
        key = q.strip().lower()
        if key in existing_prompts:
            continue
        a = random.choice(TEMPLATE_ANSWERS).format(topic=topic)
        results.append(Example(human=q, assistant=a))
        existing_prompts.add(key)
    return results


def _azure_chat(messages, endpoint, api_key, deployment, api_version):
    endpoint = endpoint.rstrip("/")

    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    payload = {
        "messages": messages,
    }

    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "api-key": api_key,
        },
    )

    try:
        with request.urlopen(req, timeout=90) as resp:
            body = resp.read().decode("utf-8")
    except error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Azure request failed: {e.code} {body}") from e
    except error.URLError as e:
        raise RuntimeError(f"Azure request failed: {e}") from e

    data = json.loads(body)
    return data["choices"][0]["message"]["content"]


def generate_azure_examples(count, topics, existing_prompts, endpoint, api_key, deployment, api_version):
    examples = []
    system = (
        "You generate high-quality short QA pairs for supervised fine-tuning. "
        "Return only JSON with keys human and assistant."
    )

    guard = 0
    while len(examples) < count and guard < count * 10:
        guard += 1
        topic = random.choice(topics)
        user = (
            "Create one unique QA example about this topic: "
            f"{topic}. Keep the answer concise and accurate."
        )
        raw = _azure_chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            endpoint=endpoint,
            api_key=api_key,
            deployment=deployment,
            api_version=api_version,
        )

        # Strip markdown fences if model returns them.
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.replace("json", "", 1).strip()

        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue

        human = str(obj.get("human", "")).strip()
        assistant = str(obj.get("assistant", "")).strip()
        if not human or not assistant:
            continue

        key = human.lower()
        if key in existing_prompts:
            continue

        existing_prompts.add(key)
        examples.append(Example(human=human, assistant=assistant))

    return examples


def save_jsonl(path, examples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.as_record(), ensure_ascii=False) + "\n")


def append_to_base_dataset(base_data_path, examples, backup_dir):
    base_data_path.parent.mkdir(parents=True, exist_ok=True)

    if base_data_path.exists():
        backup_path = Path(backup_dir).resolve() / (
            f"{base_data_path.stem}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}{base_data_path.suffix}"
        )
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text(base_data_path.read_text(
            encoding="utf-8"), encoding="utf-8")
        print(f"Backup created: {backup_path}")

    mode = "a" if base_data_path.exists() else "w"
    with base_data_path.open(mode, encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.as_record(), ensure_ascii=False) + "\n")

    print(f"Updated base dataset: {base_data_path} (+{len(examples)} rows)")


def run_finetune(base_data, generated_data, epochs, output_dir, continue_from):
    combined_data = f"{base_data},{generated_data}"
    cmd = [
        sys.executable,
        str(ROOT_DIR / "finetune.py"),
        "--data-files",
        combined_data,
        "--epochs",
        str(epochs),
        "--output-dir",
        output_dir,
        "--continue-from",
        continue_from,
    ]
    print("Starting fine-tune with combined data files:")
    print("  ", combined_data)
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def main():
    args = parse_args()
    output_path = Path(args.output).resolve()
    base_data_path = Path(args.base_data).resolve()
    topics = [t.strip() for t in args.topics.split(",") if t.strip()]

    if not topics:
        raise RuntimeError("At least one topic is required.")

    existing_prompts = read_existing_prompts([base_data_path, output_path])

    if args.mode == "template":
        generated = generate_template_examples(
            args.count, topics, existing_prompts)
    else:
        if not args.azure_endpoint or not args.azure_api_key or not args.azure_deployment:
            raise RuntimeError(
                "Azure mode requires --azure-endpoint, --azure-api-key, and --azure-deployment."
            )
        generated = generate_azure_examples(
            args.count,
            topics,
            existing_prompts,
            endpoint=args.azure_endpoint,
            api_key=args.azure_api_key,
            deployment=args.azure_deployment,
            api_version=args.azure_api_version,
        )

    if not generated:
        raise RuntimeError(
            "No examples were generated. Try different topics or a smaller count.")

    save_jsonl(output_path, generated)
    print(f"Saved {len(generated)} examples -> {output_path}")

    if args.update_base_data:
        append_to_base_dataset(
            base_data_path=base_data_path,
            examples=generated,
            backup_dir=args.backup_dir,
        )

    if args.run_finetune:
        run_finetune(
            base_data=str(base_data_path),
            generated_data=str(output_path),
            epochs=args.epochs,
            output_dir=args.output_dir,
            continue_from=args.continue_from,
        )
    else:
        print("Generation complete. Use --run-finetune to start training automatically.")


if __name__ == "__main__":
    main()
