"""test_correctness.py — cached and uncached generation must produce identical tokens.

Run BEFORE any timing. A silent bug in cache indexing produces plausible but
wrong text; the only reliable check is token-by-token equality against the
uncached reference path.

In FP32 this must pass exactly. (FP16 may diverge by a token or two on Ampere+
due to tensor-core accumulator differences — we'll cover that later.)

Run: python3 test_correctness.py
"""

import torch
from model import GPT
from inference import KVCache, generate

MODEL_TYPE   = 'gpt2'
N_NEW_TOKENS = 20
PROMPT_IDS   = [15496]  # 'Hello'


def main():
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GPT.from_pretrained(MODEL_TYPE).to(device).eval()
    cfg = model.config
    prompt = torch.tensor([PROMPT_IDS], dtype=torch.long, device=device)

    # 1. Uncached reference
    ids_uncached = generate(model, prompt, max_new_tokens=N_NEW_TOKENS, cache=None)[0].tolist()

    # 2. Cached
    cache = KVCache(
        n_layer=cfg.n_layer, B=1, max_seq_len=1024,
        n_head=cfg.n_head, head_dim=cfg.n_embd // cfg.n_head,
        device=device, dtype=torch.float32,
    )
    ids_cached = generate(model, prompt, max_new_tokens=N_NEW_TOKENS, cache=cache)[0].tolist()

    print(f"uncached: {ids_uncached}")
    print(f"cached:   {ids_cached}")

    if ids_uncached == ids_cached:
        print(f"\nOK — cached path produces identical {N_NEW_TOKENS + 1}-token sequence")
    else:
        first = next(i for i, (a, b) in enumerate(zip(ids_uncached, ids_cached)) if a != b)
        print(f"\nFAIL — diverges at position {first}")
        raise AssertionError(f"cached path diverged at position {first}")


if __name__ == '__main__':
    main()
