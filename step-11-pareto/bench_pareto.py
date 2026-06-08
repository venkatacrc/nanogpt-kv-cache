"""bench_pareto.py - throughput vs per-user latency sweep.

The "production tradeoff" plot. Increasing batch size B raises aggregate
throughput (tokens/sec across all users) but also raises per-user latency
(TPOT in ms/token). Plotting both gives the operator's decision curve.

Workload:
  - custom 1.4B model, FP16, GQA 6:1 (matches step-8 setup)
  - prompt P=128, generate N=128 (realistic chat assistant)
  - batch B sweep: 1, 2, 4, 8, 16, 32, 64, 128, 256

Memory budget at B=256 (sanity check): KV cache is
  48 layers * 2 KV heads * 128 dim * 2 bytes * 256 batch * 256 seq ~= 3.2 GB.
Model is ~2.5 GB FP16. Activations a few GB. Total well under H100's 80 GB.

We use FA3 as the backend (best prefill from step-6/step-8) and a
[B, S, H, D] cache (FA3-native layout).

Outputs per batch size:
  - prefill_ms:  total prefill time (one forward at S=P)
  - tpot_ms:     mean decode time per token
  - total_ms:    prefill + N * tpot
  - tps_user:    tokens/sec for a single user  = 1000 / tpot
  - tps_total:   tokens/sec aggregate across batch = B * tps_user

The Pareto curve x = tps_user, y = tps_total reveals the sweet spot.

Run: python3 bench_pareto.py   (H100 only)
"""

import time
import statistics
import torch
from model import GPT, GPTConfig, CausalSelfAttention
from inference import KVCache

CFG = GPTConfig(
    block_size=2048,
    vocab_size=50257,
    n_layer=48,
    n_head=12,
    n_kv_head=2,      # GQA 6:1 (matches step-8)
    n_embd=1536,
    bias=True,
)

BATCH_SIZES     = [1, 2, 4, 8, 16, 32, 64, 128, 256]
PROMPT_LEN      = 128
N_DECODE        = 128
NUM_TRIALS      = 5
PROMPT_TOKEN_ID = 15496


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def set_backend(name):
    CausalSelfAttention.USE_SDPA = (name == 'sdpa')
    CausalSelfAttention.USE_FA3  = (name == 'fa3')


@torch.no_grad()
def time_workload(model, cfg, B, P, N, device):
    """One full prefill+decode run. Returns (prefill_ms, total_ms, tpot_ms)."""
    prompt = torch.tensor([[PROMPT_TOKEN_ID] * P] * B, dtype=torch.long, device=device)
    cache = KVCache(
        n_layer=cfg.n_layer, B=B, max_seq_len=cfg.block_size,
        n_kv_head=cfg.n_kv_head, head_dim=cfg.n_embd // cfg.n_head,
        device=device, dtype=torch.float16,
    )
    sync(); t0 = time.perf_counter()
    logits = model(prompt, cache)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    sync(); t_prefill = (time.perf_counter() - t0) * 1000

    for _ in range(N - 1):
        logits = model(next_token, cache)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    sync(); total_ms = (time.perf_counter() - t0) * 1000

    decode_ms = total_ms - t_prefill
    tpot_ms = decode_ms / (N - 1)
    return {'prefill_ms': t_prefill, 'total_ms': total_ms, 'tpot_ms': tpot_ms}


def median_runs(fn, n=NUM_TRIALS):
    runs = [fn() for _ in range(n)]
    return {k: statistics.median([r[k] for r in runs]) for k in runs[0]}


def main():
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GPT(CFG).to(device).eval().half()
    head_dim = CFG.n_embd // CFG.n_head
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"loaded custom GPT: n_layer={CFG.n_layer}, n_head={CFG.n_head}, "
          f"n_kv_head={CFG.n_kv_head}, head_dim={head_dim}")
    print(f"  ({n_params:.0f}M params, FP16, GQA 6:1, [B,S,H,D] cache, FA3 backend) on {device}")

    set_backend('fa3')

    # Warmup across all batch sizes so each has its kernels cached.
    print("warming up ...")
    for B in BATCH_SIZES:
        _ = time_workload(model, CFG, B, PROMPT_LEN, 5, device)
    sync()

    print()
    print(f"=== throughput vs latency sweep | P={PROMPT_LEN} | N={N_DECODE} | "
          f"FA3 | median of {NUM_TRIALS} ===")
    print(f"  {'B':>3}  {'prefill(ms)':>11}  {'TPOT(ms)':>9}  "
          f"{'total(ms)':>10}  {'tps_user':>9}  {'tps_total':>10}")

    for B in BATCH_SIZES:
        r = median_runs(lambda B=B: time_workload(model, CFG, B, PROMPT_LEN, N_DECODE, device))
        tps_user  = 1000.0 / r['tpot_ms']
        tps_total = B * tps_user
        print(f"  {B:>3}  {r['prefill_ms']:>11.1f}  {r['tpot_ms']:>9.2f}  "
              f"{r['total_ms']:>10.0f}  {tps_user:>9.1f}  {tps_total:>10.1f}")


if __name__ == '__main__':
    main()
