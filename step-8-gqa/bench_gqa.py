"""bench_gqa.py - does GQA make FA3 finally beat SDPA at decode?

Same model architecture as step-6 / step-6b (head_dim=128, n_layer=48,
n_embd=1536, block_size=16384), but we sweep the KV-head count:

  n_kv_head = 12  (MHA baseline; matches step-6b)
  n_kv_head = 4   (3:1 group)
  n_kv_head = 2   (6:1 group, similar to Llama-3 70B's 8:1)

Predictions:
  - Decode TPOT (q_len=1, bandwidth-bound on K/V load):
      * SDPA: TPOT drops noticeably as n_kv_head shrinks
      * FA3:  TPOT drops too, possibly more (FA3 has tighter kernel for GQA)
      * The FA3/SDPA ratio should improve in FA3's favor as GQA gets aggressive
  - Prefill (compute-bound on attention matmuls):
      * Smaller KV head count means smaller K, V matrices, but the same
        Q matrix. Compute drops modestly. Both backends should drop together.

Run: python3 bench_gqa.py   (H100 only)
"""

import time
import statistics
import torch
from model import GPT, GPTConfig, CausalSelfAttention
from inference import KVCache

BASE_CFG = dict(
    block_size=16384,
    vocab_size=50257,
    n_layer=48,
    n_head=12,
    n_embd=1536,
    bias=True,
)

KV_HEAD_CASES   = [12, 4, 2]   # MHA, GQA 3:1, GQA 6:1
PREFILL_LENS    = [512, 2048, 4096, 8192]
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
        n_kv_head=cfg.n_kv_head, head_dim=cfg.n_embd // cfg.n_head,
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


def build_model(n_kv_head, device):
    cfg = GPTConfig(**BASE_CFG, n_kv_head=n_kv_head)
    torch.manual_seed(42)
    model = GPT(cfg).to(device).eval().half()
    return model, cfg


def warmup(model, cfg, device):
    for name in ['sdpa', 'fa3']:
        set_backend(name)
        for S in PREFILL_LENS:
            _ = time_prefill(model, S, device)
        for S in DECODE_LENS:
            _ = time_cached_decode(model, cfg, S, 3, device)
    sync()


def run_one_kv_head(n_kv_head, device):
    model, cfg = build_model(n_kv_head, device)
    head_dim   = cfg.n_embd // cfg.n_head
    n_params   = sum(p.numel() for p in model.parameters()) / 1e6
    label = "MHA" if n_kv_head == cfg.n_head else f"GQA n_kv_head={n_kv_head}"
    print()
    print(f"### {label}  (n_head={cfg.n_head}, n_kv_head={cfg.n_kv_head}, "
          f"head_dim={head_dim}, {n_params:.0f}M params)")
    warmup(model, cfg, device)

    print(f"  PREFILL | FP16 | median of {NUM_TRIALS}")
    print(f"  {'S':>6}  {'SDPA(ms)':>10}  {'FA3(ms)':>9}  {'SDPA/FA3':>9}")
    for S in PREFILL_LENS:
        set_backend('sdpa')
        t_s = median_of(lambda S=S: time_prefill(model, S, device))
        set_backend('fa3')
        t_f = median_of(lambda S=S: time_prefill(model, S, device))
        print(f"  {S:>6}  {t_s:>10.2f}  {t_f:>9.2f}  {t_s / t_f:>8.2f}x")

    print(f"  DECODE TPOT | N={N_DECODE} | median of {NUM_TRIALS}")
    print(f"  {'S_pre':>6}  {'SDPA(ms)':>10}  {'FA3(ms)':>9}  {'SDPA/FA3':>9}")
    for S in DECODE_LENS:
        set_backend('sdpa')
        t_s = median_of(lambda S=S: time_cached_decode(model, cfg, S, N_DECODE, device))
        set_backend('fa3')
        t_f = median_of(lambda S=S: time_cached_decode(model, cfg, S, N_DECODE, device))
        print(f"  {S:>6}  {t_s:>10.2f}  {t_f:>9.2f}  {t_s / t_f:>8.2f}x")

    del model
    torch.cuda.empty_cache()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== GQA backend sweep | custom 1.4B-ish | block_size={BASE_CFG['block_size']} ===")
    for n_kv in KV_HEAD_CASES:
        run_one_kv_head(n_kv, device)


if __name__ == '__main__':
    main()
