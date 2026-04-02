import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

from hf_client import DEFAULT_MODEL, score_completion


def clean_text(text):
    text = re.sub(r"\[.*?\]", "", text)
    return text.strip()


def eval_hellaswag(model_id=DEFAULT_MODEL, samples=500, output_dir="results"):
    Path(output_dir).mkdir(exist_ok=True)
    dataset = list(load_dataset("Rowan/hellaswag", split="validation"))
    subset = dataset if samples == 0 else dataset[:samples]

    correct = 0
    for i, row in enumerate(subset):
        context = clean_text(row["ctx"])
        endings = [clean_text(e) for e in row["endings"]]
        gold = int(row["label"])

        scores = [
            score_completion(context, " " + ending, model_id=model_id)
            for ending in endings
        ]
        pred = scores.index(max(scores))

        if pred == gold:
            correct += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(subset)}] running acc: {correct/(i+1)*100:.1f}%")

    acc = correct / len(subset) if subset else 0.0
    print(f"\nHellaSwag: {acc*100:.1f}% ({correct}/{len(subset)})")

    results = {"correct": correct, "total": len(subset), "accuracy": acc}
    out_path = Path(output_dir) / "hellaswag_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    eval_hellaswag(model_id=args.model, samples=args.samples, output_dir=args.output)
