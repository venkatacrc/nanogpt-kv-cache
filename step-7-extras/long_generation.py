"""long_generation.py — cache speedup vs number of generated tokens N.

Tests the obvious reader question: "does the cache eventually win if I
just generate longer?"

At N=500, P=1, B=1, gpt2-xl FP16, the cache lost (0.91x in step-4).
The uncached path's per-step cost grows with sequence length, so at
some N it should overtake the cached path's flat per-step cost.

This script sweeps N at fixed P=1, B=1, gpt2-xl, FP16. We expect:
  - N=100:   cached path's TTFT/per-step overhead dominates → small loss
  - N=300:   uncached still well below launch-floor crossover
  - N=500:   near break-even (~0.91x as in step-2)
  - N=800:   uncached's late-step seq_len grows → cache starts to win
  - N=1000:  cache should win by a clearer margin

GPT-2 family has block_size=1024, so P + N <= 1024. With P=1, N <= 1023.

Run: python3 long_generation.py
"""

import time
import statistics
import torch
from model import GPT
from inference import KVCache

MODEL_TYPE      = 'gpt2-xl'
N_VALUES        = [100, 300, 500, 800, 1000]
PROMPT_LEN      = 1
BATCH_SIZE      = 1
PROMPT_TOKEN_ID = 15496   # 'Hello'
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
    return {'ttft_ms': ttft * 1000, 'total_ms': total * 1000,
            'tpot_ms': (total - ttft) / (N - 1) * 1000}


@torch.no_grad()
def bench_cached(model, prompt, N):
    cfg = model.config
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
    return {'ttft_ms': ttft * 1000, 'total_ms': total * 1000,
            'tpot_ms': (total - ttft) / (N - 1) * 1000}


def main():
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GPT.from_pretrained(MODEL_TYPE).to(device).eval()
    model.half()
    cfg = model.config
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"loaded {MODEL_TYPE} ({n_params:.1f}M, FP16) on {device}")

    max_N = max(N_VALUES)
    if max_N + PROMPT_LEN > cfg.block_size:
        raise RuntimeError(
            f"N_max + P = {max_N + PROMPT_LEN} > block_size = {cfg.block_size}; "
            f"reduce max N or use a model with larger context."
        )

    warm_prompt = torch.tensor([[PROMPT_TOKEN_ID] * PROMPT_LEN] * BATCH_SIZE,
                               dtype=torch.long, device=device)
    _ = bench_uncached(model, warm_prompt, N=3)
    _ = bench_cached(model,   warm_prompt, N=3)
    _ = bench_uncached(model, warm_prompt, N=max_N)
    sync()

    print()
    print(f"=== {MODEL_TYPE} | FP16 | B={BATCH_SIZE} | P={PROMPT_LEN} | "
          f"N sweep | median of {NUM_TRIALS} ===")
    print(f"{'N':>5}  {'uncached(ms)':>13}  {'cached(ms)':>11}  "
          f"{'speedup':>9}  {'unc TPOT':>10}  {'cac TPOT':>10}")

    for N in N_VALUES:
        prompt = torch.tensor([[PROMPT_TOKEN_ID] * PROMPT_LEN] * BATCH_SIZE,
                              dtype=torch.long, device=device)
        unc_totals, cac_totals = [], []
        unc_tpots,  cac_tpots  = [], []
        for _ in range(NUM_TRIALS):
            r_unc = bench_uncached(model, prompt, N)
            r_cac = bench_cached(model,   prompt, N)
            unc_totals.append(r_unc['total_ms']); cac_totals.append(r_cac['total_ms'])
            unc_tpots.append(r_unc['tpot_ms']);   cac_tpots.append(r_cac['tpot_ms'])
        unc_med = statistics.median(unc_totals)
        cac_med = statistics.median(cac_totals)
        speedup = unc_med / cac_med
        print(f"{N:>5}  {unc_med:>13.0f}  {cac_med:>11.0f}  "
              f"{speedup:>8.2f}x  {statistics.median(unc_tpots):>10.2f}  "
              f"{statistics.median(cac_tpots):>10.2f}")


if __name__ == '__main__':
    main()
