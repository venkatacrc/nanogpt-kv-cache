"""test_backends.py - correctness across attention backends.

Reference baseline:  manual + uncached  (greedy, full recompute every step).
Tests:               { manual, SDPA } x { uncached, cached } x { FP32, FP16 }.

In FP32, manual+cached is bit-exact vs manual+uncached (validated in step-4).
SDPA uses a different reduction order so raw logits may differ by ~1e-5; greedy
argmax tokens should still match for a few hundred steps.

In FP16, every combination accumulates its own noise. We report the FIRST step
where the token sequence diverges from the reference (rather than asserting
equality), so we can see how each backend behaves.

Run: python3 test_backends.py
"""

import torch
from model import GPT, CausalSelfAttention
from inference import KVCache

MODEL_TYPE = 'gpt2'
PROMPT_IDS = [15496]   # 'Hello'
N_NEW      = 100


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def gen_uncached(model, prompt, N):
    idx = prompt
    for _ in range(N):
        logits = model(idx)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_token], dim=1)
    return idx[:, prompt.size(1):]   # only the newly generated tokens


@torch.no_grad()
def gen_cached(model, cfg, prompt, N, dtype):
    cache = KVCache(
        n_layer=cfg.n_layer, B=1, max_seq_len=cfg.block_size,
        n_head=cfg.n_head, head_dim=cfg.n_embd // cfg.n_head,
        device=prompt.device, dtype=dtype,
    )
    tokens = []
    logits = model(prompt, cache)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tokens.append(next_token.item())
    for _ in range(N - 1):
        logits = model(next_token, cache)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens.append(next_token.item())
    return torch.tensor([tokens], dtype=torch.long, device=prompt.device)


def first_diff(a, b):
    """Index of first differing token; -1 if identical over the compared length."""
    n = min(a.size(1), b.size(1))
    diff = (a[:, :n] != b[:, :n])
    if not diff.any():
        return -1
    return diff[0].nonzero()[0].item()


def set_backend(name):
    CausalSelfAttention.USE_SDPA = (name == 'SDPA')
    CausalSelfAttention.USE_FA3  = (name == 'FA3')


def run_one_precision(model, cfg, prompt, dtype, prec_name):
    # Reference: manual + uncached at this precision.
    set_backend('manual')
    ref = gen_uncached(model, prompt, N_NEW)
    sync()

    rows = []  # (backend, path, first_diff)
    for backend in ['manual', 'SDPA', 'FA3']:
        for path in ['uncached', 'cached']:
            if backend == 'manual' and path == 'uncached':
                continue  # this IS the reference
            set_backend(backend)
            try:
                if path == 'uncached':
                    tok = gen_uncached(model, prompt, N_NEW)
                else:
                    tok = gen_cached(model, cfg, prompt, N_NEW, dtype)
                rows.append((backend, path, first_diff(ref, tok)))
            except Exception as e:
                rows.append((backend, path, str(e)[:30]))
    set_backend('manual')   # restore default for caller

    for backend, path, d in rows:
        if isinstance(d, str):
            verdict = f'ERR'
        else:
            verdict = 'MATCH' if d < 0 else f'@step {d}'
        print(f"{prec_name:>5}  {backend:>7}  {path:>9}  {str(d):>11}  {verdict:>10}")


def main():
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GPT.from_pretrained(MODEL_TYPE).to(device).eval()
    cfg = model.config
    prompt = torch.tensor([PROMPT_IDS], dtype=torch.long, device=device)
    print(f"loaded {MODEL_TYPE} on {device}; greedy generation of {N_NEW} tokens")
    print(f"reference = manual + uncached at each precision\n")

    print(f"{'prec':>5}  {'backend':>7}  {'path':>9}  {'first_diff':>11}  {'verdict':>10}")

    # FP32 first (model is float by default).
    run_one_precision(model, cfg, prompt, torch.float32, 'FP32')
    print()

    # FP16: half the model once and re-run.
    model.half()
    run_one_precision(model, cfg, prompt, torch.float16, 'FP16')


if __name__ == '__main__':
    main()
