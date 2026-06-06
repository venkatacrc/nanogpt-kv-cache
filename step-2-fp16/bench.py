"""bench.py — measure both paths side-by-side.

Reports TTFT, TPOT, total time, peak GPU memory for the uncached reference
path and the cached path, plus the speedup ratio. Single trial each.
Run: python3 bench.py
"""

import time
import torch
from model import GPT
from inference import KVCache

MODEL_TYPE   = 'gpt2-xl'   # gpt2, gpt2-medium, gpt2-large, gpt2-xl
N_NEW_TOKENS = 500
PROMPT_IDS   = [15496]     # 'Hello'


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def bench_uncached(model, prompt, N):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    idx = prompt
    sync(); t0 = time.perf_counter()

    # First step (TTFT): full forward on the prompt
    logits = model(idx)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    idx = torch.cat([idx, next_token], dim=1)
    sync(); ttft = time.perf_counter() - t0

    # Remaining N-1 steps
    for _ in range(N - 1):
        logits = model(idx)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_token], dim=1)
    sync(); total = time.perf_counter() - t0

    tpot = (total - ttft) / (N - 1) if N > 1 else float('nan')
    mem  = (torch.cuda.max_memory_allocated() / (1024**3)
            if torch.cuda.is_available() else 0.0)
    return {'ttft_ms': ttft*1000, 'total_ms': total*1000,
            'tpot_ms': tpot*1000, 'peak_mem_gb': mem}


@torch.no_grad()
def bench_cached(model, prompt, N):
    cfg = model.config
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    cache = KVCache(
        n_layer=cfg.n_layer, B=1, max_seq_len=1024,
        n_head=cfg.n_head, head_dim=cfg.n_embd // cfg.n_head,
        device=prompt.device, dtype=torch.float32,
    )

    sync(); t0 = time.perf_counter()

    # First step (TTFT): prefill on prompt; first generated token comes from prefill logits
    logits = model(prompt, cache)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    sync(); ttft = time.perf_counter() - t0

    # Remaining N-1 decode steps (each = single-token forward against cache)
    for _ in range(N - 1):
        logits = model(next_token, cache)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    sync(); total = time.perf_counter() - t0

    tpot = (total - ttft) / (N - 1) if N > 1 else float('nan')
    mem  = (torch.cuda.max_memory_allocated() / (1024**3)
            if torch.cuda.is_available() else 0.0)
    return {'ttft_ms': ttft*1000, 'total_ms': total*1000,
            'tpot_ms': tpot*1000, 'peak_mem_gb': mem}


def main():
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GPT.from_pretrained(MODEL_TYPE).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"loaded {MODEL_TYPE} ({n_params:.1f}M params) on {device}")

    prompt = torch.tensor([PROMPT_IDS], dtype=torch.long, device=device)

    # Warmup each path once (small N) so timing is steady-state
    _ = bench_uncached(model, prompt, N=3)
    _ = bench_cached(model, prompt, N=3)
    sync()

    r_unc = bench_uncached(model, prompt, N_NEW_TOKENS)
    r_cac = bench_cached(model,   prompt, N_NEW_TOKENS)

    speedup = r_unc['total_ms'] / r_cac['total_ms']

    print()
    print(f"=== {MODEL_TYPE} | {device.upper()} | N={N_NEW_TOKENS} ===")
    print(f"  {'':24}  {'uncached':>10}  {'cached':>10}  {'cached/unc':>11}")
    print(f"  {'TTFT (ms)':24}  {r_unc['ttft_ms']:>10.1f}  {r_cac['ttft_ms']:>10.1f}"
          f"  {r_cac['ttft_ms']/r_unc['ttft_ms']:>10.2f}x")
    print(f"  {'TPOT (ms/token)':24}  {r_unc['tpot_ms']:>10.2f}  {r_cac['tpot_ms']:>10.2f}"
          f"  {r_cac['tpot_ms']/r_unc['tpot_ms']:>10.2f}x")
    print(f"  {'total (ms)':24}  {r_unc['total_ms']:>10.1f}  {r_cac['total_ms']:>10.1f}"
          f"  {r_cac['total_ms']/r_unc['total_ms']:>10.2f}x")
    print(f"  {'peak GPU memory (GB)':24}  {r_unc['peak_mem_gb']:>10.2f}  {r_cac['peak_mem_gb']:>10.2f}"
          f"  {r_cac['peak_mem_gb']/max(r_unc['peak_mem_gb'], 1e-9):>10.2f}x")
    print()
    print(f"  cache speedup (total): {speedup:.2f}x")


if __name__ == '__main__':
    main()
