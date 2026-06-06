"""compile.py - torch.compile as a cache rescue.

Tests two compile modes on the cached path of gpt2-xl FP16, P=1, B=1, N=500:

  - default:         kernel fusion + autotuning. Handles dynamic shapes
                     gracefully. Cuts some per-kernel cost but does not
                     remove dispatcher/python overhead.
  - reduce-overhead: adds CUDA graphs on top. Big win when shapes are
                     STATIC. Our K cache slice cache.k[..., :S+T, :] has
                     a growing length S → expect per-shape recompiles
                     and/or CUDA graph re-capture. So this is also a
                     diagnostic: how much does dynamic-shape attention
                     cost us under reduce-overhead?

Run: python3 compile.py
"""

import time
import statistics
import torch
from model import GPT
from inference import KVCache

MODEL_TYPE      = 'gpt2-xl'
N_NEW_TOKENS    = 500
PROMPT_LEN      = 1
PROMPT_TOKEN_ID = 15496
NUM_TRIALS      = 3


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def bench_uncached(model, prompt, N):
    idx = prompt
    sync(); t0 = time.perf_counter()
    logits = model(idx)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    idx = torch.cat([idx, next_token], dim=1)
    sync(); ttft = time.perf_counter() - t0
    for _ in range(N - 1):
        logits = model(idx)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_token], dim=1)
    sync(); total = time.perf_counter() - t0
    return {'ttft_ms': ttft*1000, 'total_ms': total*1000}


@torch.no_grad()
def bench_cached(model, cfg, prompt, N):
    B = prompt.size(0)
    cache = KVCache(
        n_layer=cfg.n_layer, B=B, max_seq_len=cfg.block_size,
        n_head=cfg.n_head, head_dim=cfg.n_embd // cfg.n_head,
        device=prompt.device, dtype=torch.float16,
    )
    sync(); t0 = time.perf_counter()
    logits = model(prompt, cache)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    sync(); ttft = time.perf_counter() - t0
    for _ in range(N - 1):
        logits = model(next_token, cache)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    sync(); total = time.perf_counter() - t0
    return {'ttft_ms': ttft*1000, 'total_ms': total*1000}


def median_runs(fn, n=NUM_TRIALS):
    results = [fn() for _ in range(n)]
    return {
        'ttft_ms':  statistics.median([r['ttft_ms']  for r in results]),
        'total_ms': statistics.median([r['total_ms'] for r in results]),
    }


def main():
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GPT.from_pretrained(MODEL_TYPE).to(device).eval()
    model.half()
    cfg = model.config
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"loaded {MODEL_TYPE} ({n_params:.1f}M, FP16) on {device}")

    prompt = torch.tensor([[PROMPT_TOKEN_ID] * PROMPT_LEN], dtype=torch.long, device=device)

    # ---- Eager baselines ----
    _ = bench_uncached(model, prompt, N=3)
    _ = bench_cached(model, cfg, prompt, N=3)
    _ = bench_uncached(model, prompt, N=N_NEW_TOKENS)   # JIT pollution buffer
    sync()
    r_unc = median_runs(lambda: bench_uncached(model, prompt, N_NEW_TOKENS))
    r_cac = median_runs(lambda: bench_cached(model, cfg, prompt, N_NEW_TOKENS))

    # ---- Compiled: default mode ----
    print("compiling (mode='default') ...")
    t0 = time.perf_counter()
    model_def = torch.compile(model, mode='default', fullgraph=False)
    _ = bench_cached(model_def, cfg, prompt, N=N_NEW_TOKENS)   # absorb compile
    sync()
    print(f"  compile+warmup: {time.perf_counter()-t0:.1f}s")
    r_def = median_runs(lambda: bench_cached(model_def, cfg, prompt, N_NEW_TOKENS))

    # ---- Compiled: reduce-overhead mode ----
    print("compiling (mode='reduce-overhead') ...")
    t0 = time.perf_counter()
    model_ro = torch.compile(model, mode='reduce-overhead', fullgraph=False)
    _ = bench_cached(model_ro, cfg, prompt, N=N_NEW_TOKENS)
    sync()
    print(f"  compile+warmup: {time.perf_counter()-t0:.1f}s")
    r_ro = median_runs(lambda: bench_cached(model_ro, cfg, prompt, N_NEW_TOKENS))

    print()
    print(f"=== {MODEL_TYPE} | FP16 | P={PROMPT_LEN} | B=1 | N={N_NEW_TOKENS} | "
          f"median of {NUM_TRIALS} ===")
    print(f"{'path':>28}  {'total(ms)':>10}  {'TTFT(ms)':>9}  "
          f"{'TPOT(ms)':>9}  {'vs uncached':>12}")
    rows = [
        ('uncached_eager',            r_unc),
        ('cached_eager',              r_cac),
        ('cached_compile_default',    r_def),
        ('cached_compile_reduceoh',   r_ro),
    ]
    for name, r in rows:
        tpot = (r['total_ms'] - r['ttft_ms']) / (N_NEW_TOKENS - 1)
        speedup = r_unc['total_ms'] / r['total_ms']
        print(f"{name:>28}  {r['total_ms']:>10.0f}  {r['ttft_ms']:>9.1f}  "
              f"{tpot:>9.2f}  {speedup:>11.2f}x")


if __name__ == '__main__':
    main()
