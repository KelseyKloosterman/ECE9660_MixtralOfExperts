import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

from hf_client import DEFAULT_MODEL, generate

FEW_SHOT_EXAMPLES = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. #### 6",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. #### 8",
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. #### 9",
    },
]


def build_prompt(question):
    prompt = ""
    for ex in FEW_SHOT_EXAMPLES:
        prompt += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"
    prompt += f"Question: {question}\nAnswer:"
    return prompt


def extract_answer(response):
    match = re.search(r"####\s*([\d,]+)", response)
    if match:
        return match.group(1).replace(",", "").strip()
    numbers = re.findall(r"[\d,]+", response)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return None


def eval_gsm8k(model_id=DEFAULT_MODEL, samples=200, output_dir="results"):
    Path(output_dir).mkdir(exist_ok=True)
    dataset = list(load_dataset("openai/gsm8k", "main", split="test"))
    subset = dataset if samples == 0 else dataset[:samples]

    correct = 0
    for i, row in enumerate(subset):
        prompt = build_prompt(row["question"])
        response = generate(prompt, model_id=model_id, max_tokens=256)
        pred = extract_answer(response)
        gold = extract_answer(row["answer"])
        if pred is not None and gold is not None and pred == gold:
            correct += 1
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(subset)}] running acc: {correct/(i+1)*100:.1f}%")

    acc = correct / len(subset) if subset else 0.0
    print(f"\nGSM8K: {acc*100:.1f}% ({correct}/{len(subset)})")

    results = {"correct": correct, "total": len(subset), "accuracy": acc}
    out_path = Path(output_dir) / "gsm8k_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    eval_gsm8k(model_id=args.model, samples=args.samples, output_dir=args.output)
