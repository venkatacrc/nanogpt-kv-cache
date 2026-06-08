"""test_gqa.py - quick correctness check across MHA and GQA configurations.

For each n_kv_head in {n_head (MHA), 4, 2}:
  - run a small random-init model
  - verify all three backends (manual, SDPA, FA3) produce nearly identical
    logits in FP16 (small noise floor expected)

This is a sanity gate before the GQA bench, not a precision study.
"""

import torch
from model import GPT, GPTConfig, CausalSelfAttention

SMALL_CFG = dict(
    block_size=512,
    vocab_size=50257,
    n_layer=4,
    n_head=12,
    n_embd=768,           # head_dim = 64
    bias=True,
)

KV_HEAD_CASES = [12, 4, 2]


def set_backend(name):
    CausalSelfAttention.USE_SDPA = (name == 'sdpa')
    CausalSelfAttention.USE_FA3  = (name == 'fa3')


@torch.no_grad()
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    for n_kv in KV_HEAD_CASES:
        cfg = GPTConfig(**SMALL_CFG, n_kv_head=n_kv)
        torch.manual_seed(42)
        model = GPT(cfg).to(device).eval().half()

        idx = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long, device=device)

        outs = {}
        for backend in ['manual', 'sdpa', 'fa3']:
            set_backend(backend)
            try:
                outs[backend] = model(idx).float()
            except Exception as e:
                outs[backend] = f"ERROR: {type(e).__name__}: {e}"

        label = "MHA" if n_kv == cfg.n_head else f"GQA n_kv_head={n_kv}"
        print(f"\n=== {label} (n_head={cfg.n_head}, head_dim={cfg.n_embd // cfg.n_head}) ===")
        ref = outs['manual']
        if not torch.is_tensor(ref):
            print(f"  manual: {ref}"); continue
        print(f"  manual logits shape: {tuple(ref.shape)}")
        for backend in ['sdpa', 'fa3']:
            out = outs[backend]
            if not torch.is_tensor(out):
                print(f"  {backend}: {out}")
            else:
                diff = (out - ref).abs().max().item()
                argmax_eq = (out.argmax(-1) == ref.argmax(-1)).all().item()
                print(f"  {backend}: max|diff|={diff:.3e}  argmax_match={argmax_eq}")


if __name__ == '__main__':
    main()
