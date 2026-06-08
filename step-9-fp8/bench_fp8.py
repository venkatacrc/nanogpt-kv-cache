"""bench_fp8.py - FA3 attention in FP8 vs FP16 (prefill, custom 1.4B-ish model).

H100's tensor cores are 2x faster at FP8 than FP16. For attention specifically,
that means QK^T and softmax(QK^T)@V matmuls should run roughly 2x faster at
long sequence length (compute-bound regime).

We measure prefill only. Decode is bandwidth-bound on K/V load; FP8 helps
only if the cache is STORED in FP8 too, which is a separate refactor we
don't do here.

What this script tests:
  - FA3-FP16: baseline (Q, K, V in FP16, FA3 fp16 kernel)
  - FA3-FP8:  cast Q, K, V to E4M3 right before FA3, dispatch to FP8 kernel

Two failure modes are possible and handled gracefully:
  1. The prebuilt FA3 wheel doesn't expose FP8 dispatch -> RuntimeError
  2. The kernel runs but produces NaN/wrong outputs from the naive scaling

We report what happened either way.

Run: python3 bench_fp8.py   (H100 only)
"""

import time
import statistics
import torch
from model import GPT, GPTConfig, CausalSelfAttention

CFG = GPTConfig(
    block_size=16384,
    vocab_size=50257,
    n_layer=48,
    n_head=12,
    n_embd=1536,
    bias=True,
)

PREFILL_LENS    = [512, 2048, 4096, 8192, 16384]
NUM_TRIALS      = 5
PROMPT_TOKEN_ID = 15496


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def set_backend(name):
    CausalSelfAttention.USE_SDPA    = (name == 'sdpa')
    CausalSelfAttention.USE_FA3     = (name == 'fa3')
    CausalSelfAttention.USE_FA3_FP8 = (name == 'fa3_fp8')


@torch.no_grad()
def time_prefill(model, S, device):
    prompt = torch.tensor([[PROMPT_TOKEN_ID] * S], dtype=torch.long, device=device)
    sync(); t0 = time.perf_counter()
    _ = model(prompt)
    sync(); return (time.perf_counter() - t0) * 1000


def median_of(fn, n=NUM_TRIALS):
    return statistics.median([fn() for _ in range(n)])


def smoke_test(model, device):
    """One-shot sanity: does FP8 path even run? Does it produce finite logits?"""
    prompt = torch.tensor([[PROMPT_TOKEN_ID] * 128], dtype=torch.long, device=device)

    set_backend('fa3')
    fp16_logits = model(prompt).float()

    set_backend('fa3_fp8')
    try:
        fp8_logits = model(prompt).float()
    except Exception as e:
        return False, f"FP8 dispatch failed: {type(e).__name__}: {e}"

    if not torch.isfinite(fp8_logits).all():
        return False, "FP8 output contains NaN or Inf"

    diff = (fp8_logits - fp16_logits).abs().max().item()
    argmax_match = (fp8_logits.argmax(-1) == fp16_logits.argmax(-1)).float().mean().item()
    return True, f"smoke OK: max|FP8 - FP16|={diff:.2e}, argmax_match={argmax_match*100:.0f}%"


def main():
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GPT(CFG).to(device).eval().half()
    head_dim = CFG.n_embd // CFG.n_head
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"built custom GPT: n_layer={CFG.n_layer}, n_head={CFG.n_head}, "
          f"head_dim={head_dim}, block_size={CFG.block_size}")
    print(f"  ({n_params:.0f}M params, weights FP16, [B, S, H, D] cache) on {device}")

    ok, msg = smoke_test(model, device)
    print(f"FP8 smoke test: {msg}")
    if not ok:
        print("Aborting bench: FP8 path unusable in this environment.")
        return

    print("warming up ...")
    for name in ['fa3', 'fa3_fp8']:
        set_backend(name)
        for S in PREFILL_LENS:
            _ = time_prefill(model, S, device)
    sync()

    print()
    print(f"=== custom 1.4B | FP16 weights | PREFILL | median of {NUM_TRIALS} ===")
    print(f"{'S':>6}  {'FA3 FP16(ms)':>14}  {'FA3 FP8(ms)':>13}  {'FP16/FP8':>9}")
    for S in PREFILL_LENS:
        set_backend('fa3')
        t_f16 = median_of(lambda S=S: time_prefill(model, S, device))
        set_backend('fa3_fp8')
        t_f8  = median_of(lambda S=S: time_prefill(model, S, device))
        print(f"{S:>6}  {t_f16:>14.2f}  {t_f8:>13.2f}  {t_f16 / t_f8:>8.2f}x")


if __name__ == '__main__':
    main()
