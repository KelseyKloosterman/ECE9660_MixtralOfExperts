import os
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm import generate as _mlx_generate

DEFAULT_MODEL = "mlx-community/Mistral-7B-v0.1-q"

# Load HF_TOKEN from .env if present
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

_model = None
_tokenizer = None


def load_model(model_id=DEFAULT_MODEL):
    global _model, _tokenizer
    if _model is not None:
        return
    print(f"Loading {model_id}...")
    _model, _tokenizer = load(model_id)
    print("Model loaded.")


def _forward(token_ids):
    """Forward pass, returns logits [seq_len, vocab]."""
    logits = _model(mx.array([token_ids]))
    mx.eval(logits)
    return logits[0]


def score_choices(prompt, choices, model_id=DEFAULT_MODEL):
    """
    Return {choice: logit} for each choice at the next token position after
    the prompt. Used for MMLU and ARC (logit-based scoring).
    """
    load_model(model_id)
    prompt_ids = _tokenizer.encode(prompt)
    last_logits = _forward(prompt_ids)[-1]  # logit predicting next token

    scores = {}
    for choice in choices:
        # Find which token the model would see for this choice after the prompt
        full_ids = _tokenizer.encode(prompt + " " + choice)
        if len(full_ids) > len(prompt_ids):
            token_id = full_ids[len(prompt_ids)]
        else:
            full_ids = _tokenizer.encode(prompt + choice)
            token_id = full_ids[len(prompt_ids)] if len(full_ids) > len(prompt_ids) else full_ids[-1]
        scores[choice] = last_logits[token_id].item()

    return scores


def score_completion(context, completion, model_id=DEFAULT_MODEL):
    """
    Return the mean log-probability of completion tokens given context.
    Used for HellaSwag (sequence likelihood scoring).
    """
    load_model(model_id)
    ctx_ids = _tokenizer.encode(context)
    full_ids = _tokenizer.encode(context + completion)
    ctx_len = len(ctx_ids)

    completion_ids = full_ids[ctx_len:]
    if not completion_ids:
        return float("-inf")

    logits = _forward(full_ids)
    log_probs = mx.log(mx.softmax(logits.astype(mx.float32), axis=-1))

    total = 0.0
    for i, token_id in enumerate(completion_ids):
        logit_pos = ctx_len - 1 + i
        total += log_probs[logit_pos, token_id].item()

    return total / len(completion_ids)


def generate(prompt, model_id=DEFAULT_MODEL, max_tokens=256):
    """Greedy generation. Used for GSM8K."""
    load_model(model_id)
    return _mlx_generate(_model, _tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
