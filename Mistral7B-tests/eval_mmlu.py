import argparse
import json
from pathlib import Path

from datasets import load_dataset

from hf_client import DEFAULT_MODEL, score_choices

MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]

CHOICES = ["A", "B", "C", "D"]


def format_question(row, include_answer=False):
    q = row["question"].strip()
    options = row["choices"]
    text = f"Question: {q}\n"
    for i, opt in enumerate(options):
        text += f"{CHOICES[i]}. {opt}\n"
    text += "Answer:"
    if include_answer:
        text += f" {CHOICES[row['answer']]}"
    return text


def build_prompt(few_shot_examples, test_row):
    prompt = "The following are multiple choice questions (with answers).\n\n"
    for ex in few_shot_examples:
        prompt += format_question(ex, include_answer=True) + "\n\n"
    prompt += format_question(test_row)
    return prompt


def eval_mmlu(model_id=DEFAULT_MODEL, samples=20, output_dir="results"):
    Path(output_dir).mkdir(exist_ok=True)
    results = {}
    total_correct = total_count = 0

    for subject in MMLU_SUBJECTS:
        val_data = list(load_dataset("cais/mmlu", subject, split="validation"))
        test_data = list(load_dataset("cais/mmlu", subject, split="test"))

        few_shot = val_data[:5]
        test_subset = test_data if samples == 0 else test_data[:samples]

        correct = 0
        for row in test_subset:
            prompt = build_prompt(few_shot, row)
            logits = score_choices(prompt, CHOICES, model_id=model_id)
            pred = max(logits, key=logits.get)
            gold = CHOICES[row["answer"]]
            if pred == gold:
                correct += 1

        acc = correct / len(test_subset) if test_subset else 0.0
        results[subject] = {"correct": correct, "total": len(test_subset), "accuracy": acc}
        total_correct += correct
        total_count += len(test_subset)
        print(f"  {subject}: {acc*100:.1f}% ({correct}/{len(test_subset)})")

    overall = total_correct / total_count if total_count else 0.0
    results["overall"] = {"correct": total_correct, "total": total_count, "accuracy": overall}
    print(f"\nMMlu overall: {overall*100:.1f}%")

    out_path = Path(output_dir) / "mmlu_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return overall


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    eval_mmlu(model_id=args.model, samples=args.samples, output_dir=args.output)
