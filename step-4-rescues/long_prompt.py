"""long_prompt.py — cache speedup grows with prompt length.

Same model (gpt2-xl), same precision (FP16) as step-2. The only thing we
vary here is the prompt length P. As P grows, every uncached decode step
has to re-attend the entire prefix (its compute scales with seq length),
while the cached path stays at the launch floor. Speedup grows with P.

Run: python3 long_prompt.py
"""

import time
import statistics
import torch
from model import GPT
from inference import KVCache

MODEL_TYPE      = 'gpt2-xl'
N_NEW_TOKENS    = 500
PROMPT_LENS     = [1, 128, 512]
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
    return {'ttft_ms': ttft*1000, 'total_ms': total*1000}


@torch.no_grad()
def bench_cached(model, prompt, N):
    cfg = model.config
    cache = KVCache(
        n_layer=cfg.n_layer, B=1, max_seq_len=cfg.block_size,
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


def main():
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GPT.from_pretrained(MODEL_TYPE).to(device).eval()
    model.half()
    print(f"loaded {MODEL_TYPE} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M, FP16) on {device}")

    # Warmup: small N at seq=32 to wake up cuDNN, then ONE full-length pass at the
    # longest P to JIT every kernel the real sweep will hit (avoids first-trial pollution).
    warm = torch.tensor([[PROMPT_TOKEN_ID] * 32], dtype=torch.long, device=device)
    _ = bench_uncached(model, warm, N=3)
    _ = bench_cached(model,   warm, N=3)
    long_warm = torch.tensor([[PROMPT_TOKEN_ID] * max(PROMPT_LENS)], dtype=torch.long, device=device)
    _ = bench_uncached(model, long_warm, N=N_NEW_TOKENS)
    sync()

    print()
    print(f"=== {MODEL_TYPE} | FP16 | N={N_NEW_TOKENS} | prompt-length sweep | median of {NUM_TRIALS} ===")
    print(f"{'P':>5}  {'uncached(ms)':>13}  {'cached(ms)':>11}  {'speedup':>9}  {'TTFT_unc':>10}  {'TTFT_cac':>10}")

    for P in PROMPT_LENS:
        if P + N_NEW_TOKENS > model.config.block_size:
            print(f"  skipping P={P}: P + N exceeds block_size {model.config.block_size}")
            continue
        prompt = torch.tensor([[PROMPT_TOKEN_ID] * P], dtype=torch.long, device=device)
        unc_totals, cac_totals, unc_ttfts, cac_ttfts = [], [], [], []
        for _ in range(NUM_TRIALS):
            r_unc = bench_uncached(model, prompt, N_NEW_TOKENS)
            r_cac = bench_cached(model,   prompt, N_NEW_TOKENS)
            unc_totals.append(r_unc['total_ms']); cac_totals.append(r_cac['total_ms'])
            unc_ttfts.append(r_unc['ttft_ms']);   cac_ttfts.append(r_cac['ttft_ms'])
        unc_med = statistics.median(unc_totals)
        cac_med = statistics.median(cac_totals)
        speedup = unc_med / cac_med
        print(f"{P:>5}  {unc_med:>13.0f}  {cac_med:>11.0f}  "
              f"{speedup:>8.2f}x  {statistics.median(unc_ttfts):>10.1f}  {statistics.median(cac_ttfts):>10.1f}")


if __name__ == '__main__':
    main()
