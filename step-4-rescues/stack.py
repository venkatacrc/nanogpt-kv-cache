"""stack.py - do rescues compose? Stack torch.compile + batching.

Tests whether the per-rescue improvements measured in batch.py and compile.py
multiply when combined. All at gpt2-xl, FP16, P=1, N=500.

Predicted (from prior runs):
  batch rescue (B=1 -> B=8) on cached_eager:
      S_eager(B=8) = T_unc(B=8) / T_cac_e(B=8) ~ 2.91x
  compile rescue on cached path at B=1:
      compile_factor = T_cac_e(B=1) / T_cac_c(B=1) ~ 1.86x
  if independent:
      S_compile(B=8) = S_eager(B=8) * compile_factor ~ 5.4x

Run: python3 stack.py
"""

import time
import statistics
import torch
from model import GPT
from inference import KVCache

MODEL_TYPE      = 'gpt2-xl'
N_NEW_TOKENS    = 500
PROMPT_LEN      = 1
BATCH_SIZES     = [1, 8]
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
    print(f"loaded {MODEL_TYPE} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M, FP16) on {device}")

    # One compiled wrapper; PyTorch specializes per input shape, so the first call
    # at each B will trigger a compile. We absorb both via warmup below.
    print("wrapping model with torch.compile(mode='default') ...")
    model_c = torch.compile(model, mode='default', fullgraph=False)

    results = {}
    for B in BATCH_SIZES:
        prompt = torch.tensor([[PROMPT_TOKEN_ID] * PROMPT_LEN] * B,
                              dtype=torch.long, device=device)

        # Eager warmup + JIT-pollution buffer
        _ = bench_uncached(model, prompt, N=3)
        _ = bench_cached(model, cfg, prompt, N=3)
        _ = bench_uncached(model, prompt, N=N_NEW_TOKENS)
        sync()

        # Compile warmup for this B (absorbs per-shape compile)
        print(f"compiling cached path for B={B} ...")
        tc = time.perf_counter()
        _ = bench_cached(model_c, cfg, prompt, N=N_NEW_TOKENS)
        sync()
        print(f"  compile+warmup B={B}: {time.perf_counter()-tc:.1f}s")

        results[(B, 'uncached_eager')]  = median_runs(
            lambda p=prompt: bench_uncached(model, p, N_NEW_TOKENS))
        results[(B, 'cached_eager')]    = median_runs(
            lambda p=prompt: bench_cached(model,   cfg, p, N_NEW_TOKENS))
        results[(B, 'cached_compile')]  = median_runs(
            lambda p=prompt: bench_cached(model_c, cfg, p, N_NEW_TOKENS))

    print()
    print(f"=== {MODEL_TYPE} | FP16 | P={PROMPT_LEN} | N={N_NEW_TOKENS} | "
          f"stacking compile + batch | median of {NUM_TRIALS} ===")
    print(f"{'B':>3}  {'path':>22}  {'total(ms)':>10}  {'TPOT(ms)':>9}  {'vs unc(B)':>10}")
    for B in BATCH_SIZES:
        r_unc = results[(B, 'uncached_eager')]
        for name in ['uncached_eager', 'cached_eager', 'cached_compile']:
            r = results[(B, name)]
            tpot = (r['total_ms'] - r['ttft_ms']) / (N_NEW_TOKENS - 1)
            spd  = r_unc['total_ms'] / r['total_ms']
            print(f"{B:>3}  {name:>22}  {r['total_ms']:>10.0f}  {tpot:>9.2f}  "
                  f"{spd:>9.2f}x")
        print()

    # ---- stacking analysis ----
    T_unc_B1 = results[(1, 'uncached_eager')]['total_ms']
    T_cac_e_B1 = results[(1, 'cached_eager')]['total_ms']
    T_cac_c_B1 = results[(1, 'cached_compile')]['total_ms']
    T_unc_B8 = results[(8, 'uncached_eager')]['total_ms']
    T_cac_e_B8 = results[(8, 'cached_eager')]['total_ms']
    T_cac_c_B8 = results[(8, 'cached_compile')]['total_ms']

    S_eager_B8     = T_unc_B8 / T_cac_e_B8           # batch rescue alone
    compile_factor = T_cac_e_B1 / T_cac_c_B1         # compile rescue on cached (B=1)
    S_predicted    = S_eager_B8 * compile_factor     # if rescues are independent
    S_measured     = T_unc_B8 / T_cac_c_B8           # measured combined

    print("---- stacking analysis (cache vs uncached @ B=8) ----")
    print(f"  batch rescue alone   (cached_eager   B=8):   {S_eager_B8:>5.2f}x")
    print(f"  compile factor       (cache speedup @ B=1):  {compile_factor:>5.2f}x")
    print(f"  predicted product    (independent rescues):  {S_predicted:>5.2f}x")
    print(f"  measured combined    (cached_compile B=8):   {S_measured:>5.2f}x")
    print(f"  stacking efficiency  (measured / predicted): "
          f"{S_measured/S_predicted*100:>5.0f}%")


if __name__ == '__main__':
    main()
