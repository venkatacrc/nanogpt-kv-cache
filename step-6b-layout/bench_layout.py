"""bench_layout.py - does the [B, S, H, D] cache layout rescue FA3 decode?

Same model config as step-6 (head_dim=128, n_layer=48, n_embd=1536, ~1.4B params)
but with the FA3-native cache layout (see inference.py docstring).

We re-run the prefill + decode sweeps at the SAME (S, S_pre) values as step-6
so the comparison is apples-to-apples. Prediction:
  - Decode: FA3 should now WIN (or at least match) SDPA, because the per-step
    K/V transpose tax is gone. SDPA may pay its own (smaller) transpose tax
    but should remain near launch-floor.
  - Prefill: similar to step-6. Transposes were already amortized over
    quadratic attention compute; layout change is roughly neutral at long S,
    maybe slightly favors FA3 at short S because FA3 no longer needs
    .contiguous() copies.

Run: python3 bench_layout.py  (H100 only)
"""

import time
import statistics
import torch
from model import GPT, GPTConfig, CausalSelfAttention
from inference import KVCache

CFG = GPTConfig(
    block_size=16384,
    vocab_size=50257,
    n_layer=48,
    n_head=12,
    n_embd=1536,
    bias=True,
    dropout=0.0,
)

PREFILL_LENS    = [128, 512, 2048, 4096, 8192, 16384]
DECODE_LENS     = [128, 512, 2048, 4096]
N_DECODE        = 50
NUM_TRIALS      = 5
PROMPT_TOKEN_ID = 15496


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def set_backend(name):
    CausalSelfAttention.USE_SDPA = (name == 'sdpa')
    CausalSelfAttention.USE_FA3  = (name == 'fa3')


@torch.no_grad()
def time_prefill(model, S, device):
    prompt = torch.tensor([[PROMPT_TOKEN_ID] * S], dtype=torch.long, device=device)
    sync(); t0 = time.perf_counter()
    _ = model(prompt)
    sync(); return (time.perf_counter() - t0) * 1000


@torch.no_grad()
def time_cached_decode(model, cfg, S_pre, N, device):
    cache = KVCache(
        n_layer=cfg.n_layer, B=1, max_seq_len=cfg.block_size,
        n_head=cfg.n_head, head_dim=cfg.n_embd // cfg.n_head,
        device=device, dtype=torch.float16,
    )
    pre = torch.tensor([[PROMPT_TOKEN_ID] * S_pre], dtype=torch.long, device=device)
    _ = model(pre, cache)
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

    model = GPT(CFG).to(device).eval()
    model.half()
    cfg = model.config
    head_dim = cfg.n_embd // cfg.n_head
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"built custom GPT: n_layer={cfg.n_layer}, n_head={cfg.n_head}, "
          f"n_embd={cfg.n_embd}, head_dim={head_dim}, "
          f"block_size={cfg.block_size}")
    print(f"  ({n_params:.0f}M params, FP16, [B, S, H, D] cache) on {device}")

    backends = ['sdpa', 'fa3']
    print("warming up ...")
    for name in backends:
        set_backend(name)
        for S in PREFILL_LENS:
            _ = time_prefill(model, S, device)
        for S in DECODE_LENS:
            _ = time_cached_decode(model, cfg, S, 3, device)
    sync()

    print()
    print(f"=== custom ([B,S,H,D] cache, head_dim={head_dim}, "
          f"block_size={cfg.block_size}) | FP16 | PREFILL | median of {NUM_TRIALS} ===")
    print(f"{'S':>6}  {'SDPA(ms)':>10}  {'FA3(ms)':>9}  {'FA3/SDPA':>9}")
    for S in PREFILL_LENS:
        set_backend('sdpa')
        t_s = median_of(lambda S=S: time_prefill(model, S, device))
        set_backend('fa3')
        t_f = median_of(lambda S=S: time_prefill(model, S, device))
        print(f"{S:>6}  {t_s:>10.2f}  {t_f:>9.2f}  {t_s/t_f:>8.2f}x")

    print()
    print(f"=== custom ([B,S,H,D] cache, head_dim={head_dim}, "
          f"block_size={cfg.block_size}) | FP16 | CACHED DECODE TPOT | "
          f"N={N_DECODE} | median of {NUM_TRIALS} ===")
    print(f"{'S_pre':>6}  {'SDPA(ms)':>10}  {'FA3(ms)':>9}  {'FA3/SDPA':>9}")
    for S in DECODE_LENS:
        set_backend('sdpa')
        t_s = median_of(lambda S=S: time_cached_decode(model, cfg, S, N_DECODE, device))
        set_backend('fa3')
        t_f = median_of(lambda S=S: time_cached_decode(model, cfg, S, N_DECODE, device))
        print(f"{S:>6}  {t_s:>10.2f}  {t_f:>9.2f}  {t_s/t_f:>8.2f}x")


if __name__ == '__main__':
    main()
