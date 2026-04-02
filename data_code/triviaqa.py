import json
import time

from model.model import load_model, generate_response

# TriviaQA Settings
TRAIN_PATH = "../data/triviaqa-rc/qa/web-train.json"
DEV_PATH = "../data/triviaqa-rc/qa/verified-web-dev.json"
N_SAMPLES = 407
FEW_SHOT_K = 5
MAX_TOKENS = 50

def load_data(train: str, dev: str) -> list:
    """ Loads the data from the json file

    Args:
        train (str): Path to the train json file
        dev (str): Path to the dev json file

    Returns:
        list: The data from the json file
    """

    with open(train, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    with open(dev, "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    return train_data["Data"], dev_data["Data"]

def build_few_shot_prefix(train_data: list, k: int = 4) -> str:
    """ Creates the k-few shot for training

    Args:
        train_data (list): The training data
        k (int, optional): The number of few-shot examples to use. Defaults to 4.

    Returns:
        str: The few-shot prefix
    """
    prefix = ""
    for ex in train_data[:k]:
        question = ex["Question"]
        answer   = ex["Answer"]["Value"]
        prefix  += f"Question: {question}\nAnswer: {answer}\n\n"
    return prefix


def format_prompt(few_shot_prefix: str, example: dict) -> str:
    """ Formats the prompt for the given example

    Args:
        few_shot_prefix (str): The few-shot prefix
        example (dict): The example that needs to be formatted
    Returns:
        str: The formatted prompt
    """

    return f"{few_shot_prefix} Question: {example['Question']}\nAnswer:"

def is_correct(response: str, answer: str) -> bool:
    """ Checks if the response is correct

    Args:
        response (str): The response from the model
        answer (str): The correct answer

    Returns:
        bool: True if the response is correct, False otherwise
    """
    response_normalized = response.strip().lower()
    aliases = answer.get("NormalizedAliases", [])
 
    return any(alias in response_normalized for alias in aliases)

def evaluate(client, dev_data: list) -> tuple:
    """ Evalutes the model on the given data

    Args:
        client: The model client
        dev_data (list): The data to evaluate on
    
    Returns:
        tuple: The evaluation results and final results
    """
    print(f"""
        {'=' * 60}
        Dataset: TriviaQA (verified-web-dev)
        Samples: {N_SAMPLES}
        Few-shot K: {FEW_SHOT_K}
        {'=' * 60}
          """)
    
    few_shot_prefix = build_few_shot_prefix(train_data, k=FEW_SHOT_K)
    correct = 0
    results = []
    
    # Evaluate on the first N_SAMPLES examples
    for i, example in enumerate(dev_data[:N_SAMPLES]):
        prompt = format_prompt(few_shot_prefix, example)

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

        # Checks for correctness
        correct_answer = example["Answer"]["Value"]
        aliases = example["Answer"]["NormalizedAliases"]
        is_cor = is_correct(response, example["Answer"])
        correct += int(is_cor)
 
        results.append({
            "index": i + 1,
            "question": example["Question"],
            "expected": correct_answer,
            "aliases": aliases,
            "response": response,
            "correct": is_cor,
        })

        print(f"""
            [{i + 1}/{N_SAMPLES}]
            Question: {example['Question']}
            Expected: {correct_answer}
            Response: {response}
            Correct: {'✓' if is_cor else '✗'}
              """)

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
        Results - TriviaQA (verified-web-dev)
        {'=' * 60}
        Samples Evaluated: {total}
        Correct: {correct}
        Accuracy: {accuracy:.2%}
        {'=' * 60}
         """
        )
    
    final_results = {
        "dataset": "TriviaQA (verified-web-dev)",
        "total_samples": total,
        "correct": correct,
        "accuracy": f"{accuracy:.2%}",
    }

    return results, final_results

if __name__ == "__main__":
    train_data, dev_data = load_data(TRAIN_PATH, DEV_PATH)
    client = load_model()
    results, final_results = evaluate(client, dev_data)

    with open("results/triviaqa_results.json", "w") as f:
        json.dump(final_results, f, indent=4)

