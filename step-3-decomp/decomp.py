"""decomp.py — separate prefill cost from per-decode-step cost (FP16, gpt2-xl).

For each S in {128, 256, 512} we measure three things:
  prefill         full forward on S tokens (fresh cache).
  cached_dec      one forward on 1 new token, cache already at S.
  uncached_dec    full forward on S+1 tokens (no cache; what recompute is).

What this reveals:
  cached_dec is constant w.r.t. S      -> the cache decouples decode from sequence length
  uncached_dec ALSO ~constant in FP16  -> tensor cores chew through (S+1) so fast that
                                          both paths bottom out at the same per-step floor.
                                          That floor is mostly per-layer kernel launches,
                                          which neither path can avoid.

Run: python3 decomp.py
"""

import time
import statistics
import torch
from model import GPT
from inference import KVCache

MODEL_TYPE = 'gpt2-xl'
SEQ_LENS   = [128, 256, 512]
N_TRIALS   = 5


def time_op(op, n_trials=N_TRIALS):
    """Median wall-clock (ms) of n_trials calls to op()."""
    times = []
    for _ in range(n_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        op()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000


def main():
    torch.manual_seed(42)
    device = 'cuda'

    model = GPT.from_pretrained(MODEL_TYPE).to(device).eval()
    model.half()
    cfg = model.config
    print(f"loaded {MODEL_TYPE} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M, FP16)")

    def fresh_cache():
        return KVCache(
            n_layer=cfg.n_layer, B=1, max_seq_len=1024,
            n_head=cfg.n_head, head_dim=cfg.n_embd // cfg.n_head,
            device=device, dtype=torch.float16,
        )

    new_token = torch.tensor([[42]], dtype=torch.long, device=device)

    # Warmup so cuDNN's first-call kernel selection doesn't pollute timings
    warm = torch.tensor([[15496] * 32], dtype=torch.long, device=device)
    with torch.no_grad():
        _ = model(warm, cache=fresh_cache())
    torch.cuda.synchronize()

    print()
    print(f"=== {MODEL_TYPE} | FP16 | per-step decomposition (median of {N_TRIALS} trials) ===")
    print(f"{'S':>6} {'prefill(ms)':>13} {'cached_dec(ms)':>16} {'uncached_dec(ms)':>18} {'cache/unc':>11}")

    for S in SEQ_LENS:
        prompt   = torch.tensor([[15496] * S], dtype=torch.long, device=device)
        extended = torch.cat([prompt, new_token], dim=1)
        cache = fresh_cache()

        # 1. Prefill: full forward on S tokens, fresh cache state each trial
        def prefill_op():
            cache.seq_len = 0
            with torch.no_grad():
                _ = model(prompt, cache=cache)
        prefill_t = time_op(prefill_op)

        # Populate cache to S for the cached_dec measurement
        cache.seq_len = 0
        with torch.no_grad():
            _ = model(prompt, cache=cache)   # cache.seq_len -> S

        # 2. Cached decode: one new token, cache state reset to S each trial
        def cached_dec_op():
            cache.seq_len = S
            with torch.no_grad():
                _ = model(new_token, cache=cache)
        cached_dec_t = time_op(cached_dec_op)

        # 3. Uncached decode: full forward on S+1 tokens, no cache
        def uncached_dec_op():
            with torch.no_grad():
                _ = model(extended, cache=None)
        uncached_dec_t = time_op(uncached_dec_op)

        ratio = cached_dec_t / uncached_dec_t
        print(f"{S:>6} {prefill_t:>13.1f} {cached_dec_t:>16.2f} {uncached_dec_t:>18.1f} {ratio:>10.2f}x")


if __name__ == '__main__':
    main()
