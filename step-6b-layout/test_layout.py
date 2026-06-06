"""test_layout.py - sanity check that the [B, S, H, D] cache layout is
mathematically equivalent to step-6's [B, H, S, D] layout.

The model is randomly initialized (we can't load pretrained weights at these
shapes), so we check INTERNAL consistency: all three backends must agree with
each other on the same custom model. If SDPA and FA3 both match the manual
path, the layout refactor is sound.

We compare at a small shape (S=64) so manual is feasible.

Run: python3 test_layout.py
"""

import torch
from model import GPT, GPTConfig, CausalSelfAttention
from inference import KVCache

CFG = GPTConfig(
    block_size=128,         # small, for the manual path's S^2 score matrix
    vocab_size=50257,
    n_layer=4,              # small for speed
    n_head=12,
    n_embd=1536,            # head_dim = 128 (same as bench config)
    bias=True, dropout=0.0,
)
PROMPT = [15496] * 16        # 16-token prompt


def set_backend(name):
    CausalSelfAttention.USE_SDPA = (name == 'SDPA')
    CausalSelfAttention.USE_FA3  = (name == 'FA3')


@torch.no_grad()
def forward_uncached(model, idx):
    return model(idx)


@torch.no_grad()
def forward_cached(model, cfg, idx, dtype):
    cache = KVCache(
        n_layer=cfg.n_layer, B=1, max_seq_len=cfg.block_size,
        n_head=cfg.n_head, head_dim=cfg.n_embd // cfg.n_head,
        device=idx.device, dtype=dtype,
    )
    # Prefill on the whole prompt
    return model(idx, cache)


def main():
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT(CFG).to(device).eval()
    model.half()
    cfg = model.config
    idx = torch.tensor([PROMPT], dtype=torch.long, device=device)
    print(f"custom GPT: n_layer={cfg.n_layer}, head_dim={cfg.n_embd//cfg.n_head}, "
          f"S={len(PROMPT)}, FP16, [B,S,H,D] cache layout")
    print()

    # Reference: manual + uncached
    set_backend('manual')
    ref_unc = forward_uncached(model, idx)
    ref_cac = forward_cached(model, cfg, idx, torch.float16)

    print(f"{'comparison':>30}  {'max |diff|':>12}  {'argmax agree?':>15}")
    print(f"{'manual unc vs manual cac':>30}  "
          f"{(ref_unc - ref_cac).abs().max().item():>12.3e}  "
          f"{(ref_unc.argmax(-1) == ref_cac.argmax(-1)).all().item()!s:>15}")

    for backend in ['SDPA', 'FA3']:
        set_backend(backend)
        try:
            unc = forward_uncached(model, idx)
            cac = forward_cached(model, cfg, idx, torch.float16)
            print(f"{f'{backend} unc vs manual unc':>30}  "
                  f"{(unc - ref_unc).abs().max().item():>12.3e}  "
                  f"{(unc.argmax(-1) == ref_unc.argmax(-1)).all().item()!s:>15}")
            print(f"{f'{backend} cac vs manual unc':>30}  "
                  f"{(cac - ref_unc).abs().max().item():>12.3e}  "
                  f"{(cac.argmax(-1) == ref_unc.argmax(-1)).all().item()!s:>15}")
        except Exception as e:
            print(f"{f'{backend}':>30}  ERR: {str(e)[:60]}")

    set_backend('manual')


if __name__ == '__main__':
    main()
