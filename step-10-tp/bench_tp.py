"""bench_tp.py - compare 1-GPU baseline vs Tensor Parallel (TP-2) on a
Llama-3 8B-class custom model. Tests both SDPA and FA3 backends.

Launch:
  torchrun --nproc_per_node=1 bench_tp.py      # 1-GPU baseline
  torchrun --nproc_per_node=2 bench_tp.py      # TP-2 across 2 GPUs

The same script handles both cases via env vars set by torchrun.

What we expect:
  - At ~5.7B params (single H100 has plenty of room), TP-2 should LOSE
    latency to 1-GPU: each layer adds a small NCCL all-reduce (≤ few hundred
    microseconds) that exceeds the savings from halved compute.
  - Prefill might be closer to break-even (compute scales with S, all-reduce
    doesn't); decode pays per-layer NCCL overhead with very little compute
    relief, so decode TPOT should grow noticeably under TP.
  - FA3 vs SDPA: ratios should be roughly preserved; TP is orthogonal.
  - Real value of TP at this scale isn't latency — it's memory headroom
    (each rank holds half the weights, half the cache).
"""

import os
import math
import time
import statistics
import torch
import torch.distributed as dist
from model import GPT, GPTConfig, TPState, CausalSelfAttention
from inference import KVCache

CFG = GPTConfig(
    block_size=8192,
    vocab_size=50257,
    n_layer=32,
    n_head=32,
    n_kv_head=8,
    n_embd=4096,
    bias=False,
)

PREFILL_LENS    = [512, 2048, 4096]
DECODE_LENS     = [128, 1024, 4096]
N_DECODE        = 50
NUM_TRIALS      = 5
PROMPT_TOKEN_ID = 15496


def setup_tp():
    """Initialize NCCL if launched via torchrun, else single-GPU mode."""
    rank       = int(os.environ.get('RANK',       '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    if world_size > 1:
        dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    return TPState(rank=rank, world_size=world_size), local_rank


def sync(tp):
    torch.cuda.synchronize()
    if tp.world_size > 1:
        dist.barrier()


def set_backend(name):
    CausalSelfAttention.USE_SDPA = (name == 'sdpa')
    CausalSelfAttention.USE_FA3  = (name == 'fa3')


@torch.no_grad()
def time_prefill(model, tp, S, device):
    prompt = torch.tensor([[PROMPT_TOKEN_ID] * S], dtype=torch.long, device=device)
    sync(tp); t0 = time.perf_counter()
    _ = model(prompt)
    sync(tp); return (time.perf_counter() - t0) * 1000


@torch.no_grad()
def time_cached_decode(model, cfg, tp, S_pre, N, device):
    cache = KVCache(
        n_layer=cfg.n_layer, B=1, max_seq_len=cfg.block_size,
        n_kv_head_local=cfg.n_kv_head // tp.world_size,
        head_dim=cfg.n_embd // cfg.n_head,
        device=device, dtype=torch.float16,
    )
    pre = torch.tensor([[PROMPT_TOKEN_ID] * S_pre], dtype=torch.long, device=device)
    _ = model(pre, cache)
    sync(tp)
    token = torch.tensor([[PROMPT_TOKEN_ID]], dtype=torch.long, device=device)
    t0 = time.perf_counter()
    for _ in range(N):
        _ = model(token, cache)
    sync(tp)
    return (time.perf_counter() - t0) * 1000 / N


def median_of(fn, n=NUM_TRIALS):
    return statistics.median([fn() for _ in range(n)])


def main():
    torch.manual_seed(42)
    tp, local_rank = setup_tp()
    device = f'cuda:{local_rank}'

    model = GPT(CFG, tp).to(device).eval().half()
    head_dim = CFG.n_embd // CFG.n_head
    n_local_params = sum(p.numel() for p in model.parameters()) / 1e9
    # total params: replicated parts count once, sharded parts × world_size
    # rough rule of thumb (replicated embeddings dominate at small models):
    n_total_params = n_local_params * tp.world_size  # upper bound

    if tp.rank == 0:
        mode = f"TP-{tp.world_size}" if tp.world_size > 1 else "1-GPU"
        print(f"=== {mode} | Llama-3 8B-class | "
              f"n_layer={CFG.n_layer}, n_head={CFG.n_head}, "
              f"n_kv_head={CFG.n_kv_head}, n_embd={CFG.n_embd}, "
              f"head_dim={head_dim} ===")
        print(f"  per-rank params: {n_local_params:.2f}B  "
              f"(world_size={tp.world_size})")

    # Warmup: touch every (backend, S) combination we'll measure.
    for name in ['sdpa', 'fa3']:
        set_backend(name)
        for S in PREFILL_LENS:
            _ = time_prefill(model, tp, S, device)
        for S in DECODE_LENS:
            _ = time_cached_decode(model, CFG, tp, S, 3, device)
    sync(tp)

    if tp.rank == 0:
        print()
        print(f"PREFILL | FP16 | median of {NUM_TRIALS}")
        print(f"  {'S':>6}  {'SDPA(ms)':>10}  {'FA3(ms)':>9}  {'SDPA/FA3':>9}")
    for S in PREFILL_LENS:
        set_backend('sdpa')
        t_s = median_of(lambda S=S: time_prefill(model, tp, S, device))
        set_backend('fa3')
        t_f = median_of(lambda S=S: time_prefill(model, tp, S, device))
        if tp.rank == 0:
            print(f"  {S:>6}  {t_s:>10.2f}  {t_f:>9.2f}  {t_s / t_f:>8.2f}x")

    if tp.rank == 0:
        print()
        print(f"DECODE TPOT | N={N_DECODE} | median of {NUM_TRIALS}")
        print(f"  {'S_pre':>6}  {'SDPA(ms)':>10}  {'FA3(ms)':>9}  {'SDPA/FA3':>9}")
    for S in DECODE_LENS:
        set_backend('sdpa')
        t_s = median_of(lambda S=S: time_cached_decode(model, CFG, tp, S, N_DECODE, device))
        set_backend('fa3')
        t_f = median_of(lambda S=S: time_cached_decode(model, CFG, tp, S, N_DECODE, device))
        if tp.rank == 0:
            print(f"  {S:>6}  {t_s:>10.2f}  {t_f:>9.2f}  {t_s / t_f:>8.2f}x")

    if tp.world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
