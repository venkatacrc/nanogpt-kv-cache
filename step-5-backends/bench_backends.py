"""bench_backends.py - manual vs SDPA vs FlashAttention-3 wall-clock comparison.

gpt2-xl, FP16, B=1. Two sub-experiments:
  1. PREFILL sweep   - single forward at seq_len S in {128, 256, 512, 1024}.
                       No cache. Compute-bound regime.
  2. CACHED DECODE   - pre-fill cache to S in {128, 256, 512}, then time the
                       mean per-token decode latency over N=50 steps.
                       Launch-bound regime.

Expectations:
  - SDPA beats manual everywhere (Flash-style fused kernel).
  - FA3 should beat SDPA at LONG prefill (more compute to fuse), but loses
    at short prefill / B=1 decode because each call pays for two
    [B,H,S,D] -> [B,S,H,D] transpose+contiguous() copies that the
    SDPA path avoids.

Run: python3 bench_backends.py  (H100 only - FA3 requires Hopper)
"""

import time
import statistics
import torch
from model import GPT, CausalSelfAttention
from inference import KVCache

MODEL_TYPE      = 'gpt2-xl'
PREFILL_LENS    = [128, 256, 512, 1024]
DECODE_LENS     = [128, 256, 512]
N_DECODE        = 50
NUM_TRIALS      = 5
PROMPT_TOKEN_ID = 15496


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def time_prefill(model, S, device):
    """Single forward at sequence length S, no cache. Returns ms."""
    prompt = torch.tensor([[PROMPT_TOKEN_ID] * S], dtype=torch.long, device=device)
    sync(); t0 = time.perf_counter()
    _ = model(prompt)
    sync(); return (time.perf_counter() - t0) * 1000


@torch.no_grad()
def time_cached_decode(model, cfg, S_pre, N, device):
    """Pre-fill cache with S_pre tokens (not timed), then time N single-token
    decode steps. Returns mean ms per decode step."""
    cache = KVCache(
        n_layer=cfg.n_layer, B=1, max_seq_len=cfg.block_size,
        n_head=cfg.n_head, head_dim=cfg.n_embd // cfg.n_head,
        device=device, dtype=torch.float16,
    )
    pre = torch.tensor([[PROMPT_TOKEN_ID] * S_pre], dtype=torch.long, device=device)
    _ = model(pre, cache)   # prefill (not timed; warms cache)
    sync()
    token = torch.tensor([[PROMPT_TOKEN_ID]], dtype=torch.long, device=device)
    t0 = time.perf_counter()
    for _ in range(N):
        _ = model(token, cache)
    sync()
    return (time.perf_counter() - t0) * 1000 / N


def median_of(fn, n=NUM_TRIALS):
    return statistics.median([fn() for _ in range(n)])


def main():
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GPT.from_pretrained(MODEL_TYPE).to(device).eval()
    model.half()
    cfg = model.config
    print(f"loaded {MODEL_TYPE} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M, "
          f"FP16) on {device}")

    # Backend setters. Both flags False = manual.
    def set_backend(name):
        CausalSelfAttention.USE_SDPA = (name == 'sdpa')
        CausalSelfAttention.USE_FA3  = (name == 'fa3')

    backends = ['manual', 'sdpa', 'fa3']

    # Warmup: touch every (backend, shape) pair so JIT/autotune is paid before timing.
    print("warming up ...")
    for name in backends:
        set_backend(name)
        for S in PREFILL_LENS:
            _ = time_prefill(model, S, device)
        for S in DECODE_LENS:
            _ = time_cached_decode(model, cfg, S, 3, device)
    sync()

    print()
    print(f"=== {MODEL_TYPE} | FP16 | PREFILL | median of {NUM_TRIALS} ===")
    print(f"{'S':>6}  {'manual(ms)':>11}  {'SDPA(ms)':>10}  {'FA3(ms)':>9}  "
          f"{'SDPA/man':>9}  {'FA3/man':>8}  {'FA3/SDPA':>9}")
    for S in PREFILL_LENS:
        set_backend('manual')
        t_m = median_of(lambda S=S: time_prefill(model, S, device))
        set_backend('sdpa')
        t_s = median_of(lambda S=S: time_prefill(model, S, device))
        set_backend('fa3')
        t_f = median_of(lambda S=S: time_prefill(model, S, device))
        print(f"{S:>6}  {t_m:>11.2f}  {t_s:>10.2f}  {t_f:>9.2f}  "
              f"{t_m/t_s:>8.2f}x  {t_m/t_f:>7.2f}x  {t_s/t_f:>8.2f}x")

    print()
    print(f"=== {MODEL_TYPE} | FP16 | CACHED DECODE TPOT | "
          f"N={N_DECODE} | median of {NUM_TRIALS} ===")
    print(f"{'S_pre':>6}  {'manual(ms)':>11}  {'SDPA(ms)':>10}  {'FA3(ms)':>9}  "
          f"{'SDPA/man':>9}  {'FA3/man':>8}  {'FA3/SDPA':>9}")
    for S in DECODE_LENS:
        set_backend('manual')
        t_m = median_of(lambda S=S: time_cached_decode(model, cfg, S, N_DECODE, device))
        set_backend('sdpa')
        t_s = median_of(lambda S=S: time_cached_decode(model, cfg, S, N_DECODE, device))
        set_backend('fa3')
        t_f = median_of(lambda S=S: time_cached_decode(model, cfg, S, N_DECODE, device))
        print(f"{S:>6}  {t_m:>11.2f}  {t_s:>10.2f}  {t_f:>9.2f}  "
              f"{t_m/t_s:>8.2f}x  {t_m/t_f:>7.2f}x  {t_s/t_f:>8.2f}x")


if __name__ == '__main__':
    main()
