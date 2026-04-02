import argparse
import json
import os
from pathlib import Path


def _load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

_load_env()

from hf_client import DEFAULT_MODEL, load_model
from eval_mmlu import eval_mmlu
from eval_hellaswag import eval_hellaswag
from eval_arc import eval_arc
from eval_gsm8k import eval_gsm8k

PAPER_RESULTS = {
    "mmlu":      {"mistral_7b": 62.5, "mixtral_8x7b": 70.6},
    "hellaswag": {"mistral_7b": 81.0, "mixtral_8x7b": 84.4},
    "arc":       {"mistral_7b": 54.9, "mixtral_8x7b": 59.7},
    "gsm8k":     {"mistral_7b": 50.0, "mixtral_8x7b": 74.4},
}

ALL_BENCHMARKS = ["mmlu", "hellaswag", "arc", "gsm8k"]


def print_table(scores):
    header = f"{'Benchmark':<26} {'Ours':>8}   {'Mistral 7B (paper)':>20}   {'Mixtral 8x7B (paper)':>20}"
    print("\n" + header)
    print("-" * len(header))

    rows = [
        ("MMLU (5-shot)",          "mmlu"),
        ("HellaSwag (0-shot)",     "hellaswag"),
        ("ARC-Challenge (0-shot)", "arc"),
        ("GSM8K (5-shot)",         "gsm8k"),
    ]

    for label, key in rows:
        ours = f"{scores[key]*100:.1f}%" if key in scores else "  n/a "
        m7b  = f"{PAPER_RESULTS[key]['mistral_7b']:.1f}%"
        mx   = f"{PAPER_RESULTS[key]['mixtral_8x7b']:.1f}%"
        print(f"{label:<26} {ours:>8}   {m7b:>20}   {mx:>20}")

    print()
    print("Paper results use fp16 weights. Ours use Mistral-7B-v0.1 fp16 via HuggingFace Transformers.")


def main():
    parser = argparse.ArgumentParser(description="Run Mixtral paper benchmark replication.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--benchmarks", nargs="+", choices=ALL_BENCHMARKS, default=ALL_BENCHMARKS)
    parser.add_argument("--output", default="results")
    parser.add_argument("--full", action="store_true", help="Disable sampling, use full splits")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN") and not Path.home().joinpath(".cache/huggingface/token").exists():
        print("Warning: HF_TOKEN not set. Set it with: export HF_TOKEN=your_token")
        print("You may need to accept the Mistral model terms at huggingface.co/mistralai/Mistral-7B-v0.1")
        print()

    load_model(args.model)

    Path(args.output).mkdir(exist_ok=True)
    scores = {}
    samples = 0 if args.full else None

    if "mmlu" in args.benchmarks:
        print("\n=== MMLU (5-shot) ===")
        kw = {} if samples is None else {"samples": samples}
        scores["mmlu"] = eval_mmlu(model_id=args.model, output_dir=args.output, **kw)

    if "hellaswag" in args.benchmarks:
        print("\n=== HellaSwag (0-shot) ===")
        kw = {} if samples is None else {"samples": samples}
        scores["hellaswag"] = eval_hellaswag(model_id=args.model, output_dir=args.output, **kw)

    if "arc" in args.benchmarks:
        print("\n=== ARC-Challenge (0-shot) ===")
        kw = {} if samples is None else {"samples": samples}
        scores["arc"] = eval_arc(model_id=args.model, output_dir=args.output, **kw)

    if "gsm8k" in args.benchmarks:
        print("\n=== GSM8K (5-shot) ===")
        kw = {} if samples is None else {"samples": samples}
        scores["gsm8k"] = eval_gsm8k(model_id=args.model, output_dir=args.output, **kw)

    print_table(scores)

    summary = {
        "model": args.model,
        "scores": {k: round(v * 100, 2) for k, v in scores.items()},
        "paper": PAPER_RESULTS,
    }
    summary_path = Path(args.output) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
