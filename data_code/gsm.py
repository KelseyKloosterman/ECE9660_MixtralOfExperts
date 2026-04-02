import json
import re
import time

from model.model import load_model, generate_response

# Settings for GSM8K
TRAIN_PATH = "../data/grade-school-math/train.jsonl"
TEST_PATH = "../data/grade-school-math/test.jsonl"
N_SAMPLES = 1319
FEW_SHOT_K = 4
MAX_TOKENS = 256

def load_gsm_data(train: str, test: str) -> tuple[list, list]:
    """ Loads the data from jsonl

    Args:
        train (str): The path to the training data file
        test (str): The path to the test data file

    Returns:
        tuple[list, list]: The loaded training and test data
    """
    with open(train, "r") as f:
        train_data = [json.loads(line) for line in f if line.strip()]

    with open(test, "r") as f:
        test_data = [json.loads(line) for line in f if line.strip()]

    return train_data, test_data

def extract_answer(answer_str: str) -> str:
    """ Extracts the Answer

    Args:
        answer_str (str): The answer string to extract

    Returns:
        str: The extracted answer
    """
    return answer_str.split("####")[-1].strip().replace(",", "")
    
def clean_scratchpad(answer_str: str) -> str:
    """ Cleans the scratchpad for interpretation

    Args:
        answer_str (str): The scratchpad string to clean

    Returns:
        str: The cleaned scratchpad string
    """
    return re.sub(r'<<[^>]+>>', '', answer_str).strip()

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
        prefix += f"Question: {ex['question']}\nAnswer: {clean_scratchpad(ex['answer'])}\n\n"
    return prefix

def extract_predicted_answer(response: str) -> str:
    """ Extracts the predicted answer from the model response

    Args:
        response (str): The model response

    Returns:
        str: The extracted predicted answer 
    """
    if "####" in response:
        return response.split("####")[-1].strip().replace(",", "")
    numbers = re.findall(r'\d+(?:\.\d+)?', response.replace(",", ""))
    return numbers[-1] if numbers else ""

def evaluate(client, train_data: list, test_data: list, k: int = 4, max_tokens: int = 256, temperature: float = 0.0) -> tuple:
    """ Evalutes the model on the given data

    Args:
        client (_type_): The model client
        train_data (list): The training data
        test_data (list): The test data
        k (int, optional): The number of few-shot examples to use. Defaults to 4.
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 256.
        temperature (float, optional): The temperature for sampling. Defaults to 0.0.

    Returns:
        tuple: The evaluation results and final results
    """
    print(f"""
        {'=' * 60}
        Dataset: GSM8K
        Samples: {N_SAMPLES}
        Few-shot K: {FEW_SHOT_K}
        {'=' * 60}
          """)
    
    # Sets the few shot for training
    few_shot_prefix = build_few_shot_prefix(train_data, k)
    correct = 0
    results = []

    # Evaluate on the first N_SAMPLES examples
    for i, example in enumerate(test_data[:N_SAMPLES]):
        expected = extract_answer(example["answer"])
        prompt = few_shot_prefix + f"Question: {example['question']}\nAnswer:"

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
        predicted = extract_predicted_answer(response)
        is_correct = predicted == expected
        correct += int(is_correct)
 
        results.append({
            "index": i + 1,
            "question": example["question"],
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "response": response,
        })


        print(f"""
            {i + 1} / {N_SAMPLES}
            \tQuestion: {example['question']}
            \tExpected: {expected}
            \tPredicted: {predicted}
            \tCorrect: {'✓' if is_correct else '✗'}
            \tResponse: {response}
            """)

        # Print intermediate accuracy every 10 samples
        if (i + 1) % 10 == 0:
                print(f"\n  --- Running accuracy: {correct}/{i+1} = {correct/(i+1):.2%} ---")
        
        # Sleep briefly to avoid hitting API rate limits
        time.sleep(0.5)
    
    # Final results
    total = len(results)
    accuracy = correct / total if total > 0 else 0.0

    print(f"""
        {'=' * 60}
        Results - GSM8K:
        {'=' * 60}
        Samples Evaluated: {total}
        Correct: {correct}
        Accuracy: {accuracy:.2%}
        {'=' * 60}
        """)
    
    final_results = {
        "dataset": "HellaSwag",
        "total_samples": total,
        "correct": correct,
        "accuracy": f"{accuracy:.2%}",
    }

    return results, final_results

if __name__ == "__main__":
    client = load_model()

    train, test = load_gsm_data(TRAIN_PATH, TEST_PATH)

    result, final_results = evaluate(client, train, test, k=FEW_SHOT_K, max_tokens=MAX_TOKENS, temperature=0.0)

    with open("results/gsm8k_results.json", "w") as f:
        json.dump(final_results, f, indent=4)

    