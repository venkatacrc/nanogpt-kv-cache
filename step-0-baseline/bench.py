"""bench.py — TTFT / TPOT / total time / peak memory for the no-cache baseline.

We split the timing so TTFT (time to first token) is reported separately from
TPOT (steady-state time per output token). Single trial; for a quick read.
Run: python3 bench.py
"""

import time
import torch
from model import GPT

MODEL_TYPE   = 'gpt2'   # one of: gpt2, gpt2-medium, gpt2-large, gpt2-xl
N_NEW_TOKENS = 50
PROMPT_IDS   = [15496]  # 'Hello' in GPT-2 BPE


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def bench_uncached(model, prompt, N):
    """Generate N tokens with no cache (full forward every step).

    Returns dict with ttft_ms, total_ms, tpot_ms, peak_mem_gb, output_ids.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    idx = prompt

    sync()
    t0 = time.perf_counter()

    # ---- first step (defines TTFT) ----
    logits = model(idx)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    idx = torch.cat([idx, next_token], dim=1)
    sync()
    ttft = time.perf_counter() - t0

    # ---- remaining N-1 steps (define TPOT) ----
    for _ in range(N - 1):
        logits = model(idx)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_token], dim=1)
    sync()
    total = time.perf_counter() - t0

    tpot = (total - ttft) / (N - 1) if N > 1 else float('nan')
    mem_gb = (torch.cuda.max_memory_allocated() / (1024**3)
              if torch.cuda.is_available() else 0.0)

    return {
        'ttft_ms':     ttft * 1000,
        'total_ms':    total * 1000,
        'tpot_ms':     tpot * 1000,
        'peak_mem_gb': mem_gb,
        'output_ids':  idx[0].tolist(),
    }


def main():
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GPT.from_pretrained(MODEL_TYPE).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"loaded {MODEL_TYPE} ({n_params:.1f}M params) on {device}")

    prompt = torch.tensor([PROMPT_IDS], dtype=torch.long, device=device)

    # one warmup pass (small, just to wake up the GPU / cuDNN)
    with torch.no_grad():
        _ = model(prompt)
    sync()

    r = bench_uncached(model, prompt, N_NEW_TOKENS)

    print()
    print(f"=== {MODEL_TYPE} | {device.upper()} | no cache | N={N_NEW_TOKENS} ===")
    print(f"  TTFT (time to first token):  {r['ttft_ms']:8.1f} ms")
    print(f"  TPOT (time per output token):{r['tpot_ms']:8.1f} ms/token")
    print(f"  total                        {r['total_ms']:8.1f} ms")
    print(f"  peak GPU memory              {r['peak_mem_gb']:8.2f} GB")

    try:
        import tiktoken
        text = tiktoken.get_encoding('gpt2').decode(r['output_ids'])
        print(f"  output:                      {text!r}")
    except ImportError:
        pass


if __name__ == '__main__':
    main()
