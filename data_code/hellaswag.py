import json
import time

from model.model import load_model, generate_response

# Settings for HellaSwag
VAL_PATH = "../data/hellaswag/hellaswag_val.jsonl"
N_SAMPLES = 3000
MAX_TOKENS = 256
LABELS = ["A", "B", "C", "D"]

def load_data(val: str) -> list:
    """ Loads the data from the json

    Args:
        val (str): Path to the data

    Returns:
        list: The loaded data
    """
    with open(val, "r") as f:
        val_data = [json.loads(line) for line in f if line.strip()]

    return val_data

def format_example(example: dict, include_answer: bool = False) -> str:
    """ Formats the example

    Args:
        example (dict): The example to format
        include_answer (bool, optional): Whether to include the correct answer in the prompt. Defaults to False.

    Returns:
        str: The formatted example
    """
    
    ctx = example["ctx"]
    endings = example["endings"]
 
    prompt = f"Context: {ctx}\n"
    for label, ending in zip(LABELS, endings):
        prompt += f"{label}. {ending}\n"
    prompt += "Answer:"
 
    if include_answer:
        correct_letter = LABELS[int(example["label"])]
        prompt += f" {correct_letter}"
 
    return prompt

def extract_predicted_label(response: str) -> str:
    for char in response.strip().upper():
        if char in LABELS:
            return char
    return ""

def evaluate(client, val_data: list) -> tuple:
    """ Evalutes the model on the given data

    Args:
        client (_type_): The model client
        val_data (list): The data to evaluate on

    Returns:
        tuple: The evaluation results and final results
    """
    print(f"""
        {'=' * 60}
        Dataset: HellaSwag
        Samples: {N_SAMPLES}
        Few-shot K: 0
        {'=' * 60}
          """)
    
    correct = 0
    results = []
    
    # Evaluate on the first N_SAMPLES examples
    for i, example in enumerate(val_data[:N_SAMPLES]):
        expected_label = LABELS[int(example["label"])]
        prompt = format_example(example)

        # Retry logic for API rate limits
        for attempt in range(3):
            try:
                response = generate_response(client, prompt, max_tokens=MAX_TOKENS, temperature=0.0)
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  API error: {e} — retrying in 5s...")
                    time.sleep(5)
                else:
                    print(f"  API error after 3 attempts: {e}")
                    response = ""
 
        # Checks correctness
        predicted_label = extract_predicted_label(response)
        is_correct = predicted_label == expected_label
        correct += int(is_correct)
 
        results.append({
            "index": i + 1,
            "ctx": example["ctx"],
            "endings": example["endings"],
            "expected": expected_label,
            "predicted": predicted_label,
            "correct": is_correct,
            "response": response,
        })

        print(f"{i + 1} / {N_SAMPLES}")
        print(f"Context: {example['ctx']}")
        for label, ending in zip(LABELS, example["endings"]):
            marker = " ←" if label == expected_label else ""
            print(f"{label}. {ending}{marker}")
        print(f"Expected: {expected_label}")
        print(f"Predicted: {predicted_label}")
        print(f"Correct: {'✓' if is_correct else '✗'}")

        # Print intermediate accuracy every 10 samples
        if (i + 1) % 10 == 0:
            print(f"\n  --- Running accuracy: {correct}/{i+1} = {correct/(i+1):.2%} ---")
 
        # Sleep briefly to avoid hitting API rate limits
        time.sleep(0.5)
    
    # Final results
    total    = len(results)
    accuracy = correct / total if total > 0 else 0.0

    print(f"""
        {'=' * 60}
        Results - HellaSwag
        {'=' * 60}
        Samples Evaluated: {total}
        Correct: {correct}
        Accuracy: {accuracy:.2%}
        {'=' * 60}
         """
        )
    
    final_results = {
        "dataset": "HellaSwag",
        "total_samples": total,
        "correct": correct,
        "accuracy": f"{accuracy:.2%}",
    }

    return results, final_results

if __name__ == "__main__":
    val_data = load_data(VAL_PATH)
    client = load_model()
    results, final_results = evaluate(client, val_data)

    with open("results/hellaswag_results.json", "w") as f:
        json.dump(final_results, f, indent=4)
