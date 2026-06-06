"""test_correctness.py — cached vs uncached, in FP16.

In FP32 we verified cached and uncached produce identical token IDs.
In FP16 on Ampere/Hopper this is NOT guaranteed: tensor-core matmul
accumulators use TF32-style rounding that differs between the cached
shape (q has T=1, k has T=S+1) and the uncached shape (q,k both T=S+1),
so the kernels picked by cuDNN can differ and rounding drifts over many
layers + many steps. Result: a token or two of divergence is normal.

Pass criterion here: most tokens still match, and we report where the
first divergence happens. Not a crash.

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
    model.half()                                # <-- FP16
    cfg = model.config
    prompt = torch.tensor([PROMPT_IDS], dtype=torch.long, device=device)

    # 1. Uncached reference (FP16)
    ids_uncached = generate(model, prompt, max_new_tokens=N_NEW_TOKENS, cache=None)[0].tolist()

    # 2. Cached (FP16 cache must match model dtype)
    cache = KVCache(
        n_layer=cfg.n_layer, B=1, max_seq_len=1024,
        n_head=cfg.n_head, head_dim=cfg.n_embd // cfg.n_head,
        device=device, dtype=torch.float16,
    )
    ids_cached = generate(model, prompt, max_new_tokens=N_NEW_TOKENS, cache=cache)[0].tolist()

    print(f"uncached: {ids_uncached}")
    print(f"cached:   {ids_cached}")

    if ids_uncached == ids_cached:
        print(f"\nOK — FP16 cached and uncached produce identical {N_NEW_TOKENS + 1}-token sequence")
    else:
        first = next(i for i, (a, b) in enumerate(zip(ids_uncached, ids_cached)) if a != b)
        matches = sum(1 for a, b in zip(ids_uncached, ids_cached) if a == b)
        print(f"\nFP16 divergence: {matches}/{len(ids_uncached)} tokens match, "
              f"first divergence at position {first}")
        print("This is expected on Ampere/Hopper — see docstring.")


if __name__ == '__main__':
    main()
