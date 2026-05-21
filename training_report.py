"""
Build a training report from Hugging Face Trainer checkpoints.

Examples:
  python training_report.py --output-dir out-finetune
  python training_report.py --output-dir out-finetune --report-dir out-finetune/reports
"""

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build training reports from trainer_state.json files.")
    parser.add_argument("--output-dir", default="out-finetune",
                        help="Fine-tune output directory.")
    parser.add_argument("--report-dir", default="",
                        help="Directory to write reports. Defaults to <output-dir>/reports.")
    return parser.parse_args()


def find_trainer_states(output_dir: Path):
    states = []
    root_state = output_dir / "trainer_state.json"
    if root_state.exists():
        states.append(root_state)

    for ckpt_dir in sorted(output_dir.glob("checkpoint-*")):
        state = ckpt_dir / "trainer_state.json"
        if state.exists():
            states.append(state)

    return states


def load_log_history(state_files):
    history = []
    for path in state_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for row in payload.get("log_history", []):
            if isinstance(row, dict):
                history.append(row)
    return history


def dedupe_step_metrics(log_history):
    by_step = {}
    for row in log_history:
        step = row.get("step")
        if step is None:
            continue

        item = by_step.setdefault(int(step), {
            "step": int(step),
            "loss": None,
            "learning_rate": None,
            "epoch": None,
            "eval_loss": None,
        })
        if "loss" in row:
            item["loss"] = row["loss"]
        if "learning_rate" in row:
            item["learning_rate"] = row["learning_rate"]
        if "epoch" in row:
            item["epoch"] = row["epoch"]
        if "eval_loss" in row:
            item["eval_loss"] = row["eval_loss"]

    return [by_step[k] for k in sorted(by_step)]


def write_reports(metrics, output_dir: Path, report_dir: Path):
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = report_dir / "training_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "loss", "eval_loss", "learning_rate", "epoch"],
        )
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)

    best_loss = None
    best_step = None
    for row in metrics:
        loss = row.get("loss")
        if loss is None:
            continue
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_step = row["step"]

    summary = {
        "output_dir": str(output_dir),
        "report_dir": str(report_dir),
        "total_logged_steps": len(metrics),
        "best_train_loss": best_loss,
        "best_train_step": best_step,
    }

    summary_path = report_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_md = report_dir / "training_report.md"
    report_md.write_text(
        "\n".join(
            [
                "# TAAL Training Report",
                "",
                f"- Output directory: {output_dir}",
                f"- Logged steps: {len(metrics)}",
                f"- Best train loss: {best_loss if best_loss is not None else 'N/A'}",
                f"- Best train step: {best_step if best_step is not None else 'N/A'}",
                "",
                "## Artifacts",
                "",
                "- training_metrics.csv",
                "- training_summary.json",
                "",
                "## Live Monitoring",
                "",
                f"Run: tensorboard --logdir {output_dir / 'runs'}",
            ]
        ),
        encoding="utf-8",
    )

    return csv_path, summary_path, report_md


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    report_dir = Path(args.report_dir).resolve(
    ) if args.report_dir else output_dir / "reports"

    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    state_files = find_trainer_states(output_dir)
    if not state_files:
        raise FileNotFoundError(
            "No trainer_state.json files found. Run fine-tuning first or check --output-dir."
        )

    log_history = load_log_history(state_files)
    if not log_history:
        raise RuntimeError("No log_history found in trainer_state.json files.")

    metrics = dedupe_step_metrics(log_history)
    csv_path, summary_path, report_md = write_reports(
        metrics, output_dir, report_dir)

    print("Training report created:")
    print(f"  - {csv_path}")
    print(f"  - {summary_path}")
    print(f"  - {report_md}")


if __name__ == "__main__":
    main()
