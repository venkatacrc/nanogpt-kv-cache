"""model_size.py — cache speedup vs model size, FP16, B=1, P=1.

Tests the launch-floor hypothesis from step-3.

Hypothesis: smaller models have less compute per layer but roughly the
same number of kernel launches, so the cached path is dominated by the
launch floor regardless of model size. The uncached path, in contrast,
scales with model size. Therefore the cache should help LESS at smaller
models (lower speedup at gpt2 than at gpt2-xl) — or even lose more.

Same settings as step-2-fp16/bench.py but with median-of-3 + beefed-up
warmup (lesson from step-4-rescues).

Models: gpt2 (124M), gpt2-medium (354M), gpt2-large (774M), gpt2-xl (1557M).

Run: python3 model_size.py
"""

import time
import statistics
import torch
from model import GPT
from inference import KVCache

MODELS          = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
N_NEW_TOKENS    = 500
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


def run_one_model(model_type, device):
    model = GPT.from_pretrained(model_type).to(device).eval()
    model.half()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  loaded {model_type} ({n_params:.1f}M, FP16)")

    warm_prompt = torch.tensor([[PROMPT_TOKEN_ID] * PROMPT_LEN] * BATCH_SIZE,
                               dtype=torch.long, device=device)
    _ = bench_uncached(model, warm_prompt, N=3)
    _ = bench_cached(model,   warm_prompt, N=3)
    _ = bench_uncached(model, warm_prompt, N=N_NEW_TOKENS)
    sync()

    prompt = torch.tensor([[PROMPT_TOKEN_ID] * PROMPT_LEN] * BATCH_SIZE,
                          dtype=torch.long, device=device)
    unc_totals, cac_totals = [], []
    unc_tpots,  cac_tpots  = [], []
    for _ in range(NUM_TRIALS):
        r_unc = bench_uncached(model, prompt, N_NEW_TOKENS)
        r_cac = bench_cached(model,   prompt, N_NEW_TOKENS)
        unc_totals.append(r_unc['total_ms']); cac_totals.append(r_cac['total_ms'])
        unc_tpots.append(r_unc['tpot_ms']);   cac_tpots.append(r_cac['tpot_ms'])

    del model
    torch.cuda.empty_cache()

    return {
        'n_params_M': n_params,
        'unc_total_ms': statistics.median(unc_totals),
        'cac_total_ms': statistics.median(cac_totals),
        'unc_tpot_ms':  statistics.median(unc_tpots),
        'cac_tpot_ms':  statistics.median(cac_tpots),
    }


def main():
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"=== FP16 | B={BATCH_SIZE} | P={PROMPT_LEN} | N={N_NEW_TOKENS} | "
          f"model size sweep | median of {NUM_TRIALS} ===")

    results = []
    for m in MODELS:
        r = run_one_model(m, device)
        r['name'] = m
        results.append(r)

    print()
    print(f"{'model':>15}  {'params(M)':>10}  {'unc(ms)':>10}  {'cac(ms)':>10}  "
          f"{'unc TPOT':>9}  {'cac TPOT':>9}  {'speedup':>9}")
    for r in results:
        speedup = r['unc_total_ms'] / r['cac_total_ms']
        print(f"{r['name']:>15}  {r['n_params_M']:>10.1f}  "
              f"{r['unc_total_ms']:>10.0f}  {r['cac_total_ms']:>10.0f}  "
              f"{r['unc_tpot_ms']:>9.2f}  {r['cac_tpot_ms']:>9.2f}  "
              f"{speedup:>8.2f}x")


if __name__ == '__main__':
    main()
