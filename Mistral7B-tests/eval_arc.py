import argparse
import json
from pathlib import Path

from datasets import load_dataset

from hf_client import DEFAULT_MODEL, score_choices

LETTERS = ["A", "B", "C", "D", "E"]


def normalize_key(key):
    if key.isdigit():
        return LETTERS[int(key) - 1]
    return key.upper()


def format_prompt(row):
    question = row["question"].strip()
    choices = row["choices"]
    labels = [normalize_key(l) for l in choices["label"]]
    texts = choices["text"]

    prompt = f"Question: {question}\n"
    for label, text in zip(labels, texts):
        prompt += f"{label}. {text}\n"
    prompt += "Answer:"
    return prompt, labels


def eval_arc(model_id=DEFAULT_MODEL, samples=300, output_dir="results"):
    Path(output_dir).mkdir(exist_ok=True)
    dataset = list(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"))
    subset = dataset if samples == 0 else dataset[:samples]

    correct = 0
    for i, row in enumerate(subset):
        prompt, labels = format_prompt(row)
        logits = score_choices(prompt, labels, model_id=model_id)
        pred = max(logits, key=logits.get)
        gold = normalize_key(row["answerKey"])

        if pred == gold:
            correct += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(subset)}] running acc: {correct/(i+1)*100:.1f}%")

    acc = correct / len(subset) if subset else 0.0
    print(f"\nARC-Challenge: {acc*100:.1f}% ({correct}/{len(subset)})")

    results = {"correct": correct, "total": len(subset), "accuracy": acc}
    out_path = Path(output_dir) / "arc_challenge_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    eval_arc(model_id=args.model, samples=args.samples, output_dir=args.output)
