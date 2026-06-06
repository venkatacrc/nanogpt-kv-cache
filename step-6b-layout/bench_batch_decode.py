"""bench_batch_decode.py - does batching rescue FA3 in decode?

step-6b's B=1 result: FA3 decode is within ~4% of SDPA even with FA3-native
[B, S, H, D] cache layout. The remaining gap is SDPA's specialized q_len=1
decode fast-path. Hypothesis: at larger batch sizes, FA3's parallelism over
the batch dimension finally pulls ahead, because:

  - SDPA's q_len=1 fast-path doesn't scale per-batch-item; per-step launch
    overhead is amortized across B but compute scales linearly with B.
  - FA3's kernel processes (B, H, q_len, k_len) tiles natively and is highly
    parallel along B; its per-token-per-batch cost should be lower.

If FA3 wins at B>=4, we've shown that the WHEN-DOES-FA3-WIN answer for
single-stream decode is: only with batching (or layout, or both).

Run: python3 bench_batch_decode.py  (H100 only)
"""

import time
import statistics
import torch
from model import GPT, GPTConfig, CausalSelfAttention
from inference import KVCache

# Same model as bench_layout.py.
CFG = GPTConfig(
    block_size=16384,
    vocab_size=50257,
    n_layer=48,
    n_head=12,
    n_embd=1536,
    bias=True,
    dropout=0.0,
)

BATCH_SIZES     = [1, 4, 8]
S_PREFILLS      = [128, 4096]
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
def time_cached_decode(model, cfg, S_pre, N, B, device):
    """Pre-fill cache (not timed) then time N single-token decode steps.
    Returns mean ms per decode step."""
    # Only allocate as much cache as we need (B=8, max_seq=16384 would be 38 GB).
    cache_seq = S_pre + N
    cache = KVCache(
        n_layer=cfg.n_layer, B=B, max_seq_len=cache_seq,
        n_head=cfg.n_head, head_dim=cfg.n_embd // cfg.n_head,
        device=device, dtype=torch.float16,
    )
    pre = torch.full((B, S_pre), PROMPT_TOKEN_ID, dtype=torch.long, device=device)
    _ = model(pre, cache)
    sync()
    token = torch.full((B, 1), PROMPT_TOKEN_ID, dtype=torch.long, device=device)
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
          f"n_embd={cfg.n_embd}, head_dim={head_dim}")
    print(f"  ({n_params:.0f}M params, FP16, [B, S, H, D] cache) on {device}")

    print("warming up ...")
    for name in ['sdpa', 'fa3']:
        set_backend(name)
        for B in BATCH_SIZES:
            for S in S_PREFILLS:
                _ = time_cached_decode(model, cfg, S, 3, B, device)
    sync()

    for S_pre in S_PREFILLS:
        print()
        print(f"=== custom ([B,S,H,D] cache, head_dim={head_dim}) | FP16 | "
              f"DECODE TPOT | S_pre={S_pre} | N={N_DECODE} | median of {NUM_TRIALS} ===")
        print(f"{'B':>3}  {'SDPA(ms)':>10}  {'FA3(ms)':>9}  {'FA3/SDPA':>9}")
        for B in BATCH_SIZES:
            set_backend('sdpa')
            t_s = median_of(lambda B=B, S=S_pre:
                            time_cached_decode(model, cfg, S, N_DECODE, B, device))
            set_backend('fa3')
            t_f = median_of(lambda B=B, S=S_pre:
                            time_cached_decode(model, cfg, S, N_DECODE, B, device))
            print(f"{B:>3}  {t_s:>10.2f}  {t_f:>9.2f}  {t_s/t_f:>8.2f}x")


if __name__ == '__main__':
    main()
