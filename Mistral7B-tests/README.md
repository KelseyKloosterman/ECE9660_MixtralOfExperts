

# Results

| Benchmark | Ours | Mistral 7B (paper) | Mixtral 8x7B (paper) |
|---|---|---|---|
| MMLU (5-shot) | 61.6% | 62.5% | 70.6% |
| HellaSwag (0-shot) | 61.2% | 81.0% | 84.4% |
| ARC-Challenge (0-shot) | 71.7% | 54.9% | 59.7% |
| GSM8K (5-shot) | 23.5% | 50.0% | 74.4% |

## Setup

```
pip install -r requirements.txt
python run_evals.py
```

Requires a HuggingFace token in a `.env` file:
```
HF_TOKEN=your_token_here
```

## Files

- `hf_client.py` — model loading, logit scoring, and generation via MLX
- `eval_mmlu.py` — 5-shot MMLU across 57 subjects (default 20 samples each)
- `eval_hellaswag.py` — 0-shot HellaSwag, sequence likelihood scoring (default 500 samples)
- `eval_arc.py` — 0-shot ARC-Challenge, logit scoring (default 300 samples)
- `eval_gsm8k.py` — 5-shot GSM8K with chain-of-thought generation (default 200 samples)
- `run_evals.py` — runs all benchmarks and prints the comparison table
