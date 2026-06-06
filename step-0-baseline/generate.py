"""generate.py — naive greedy generation, no KV cache.

Every step does a full forward pass on the entire sequence-so-far. Slow.
Run: python3 generate.py
"""

import time
import torch
from model import GPT


@torch.no_grad()
def generate(model, idx, max_new_tokens):
    """Greedy: recompute the full forward every step."""
    for _ in range(max_new_tokens):
        logits = model(idx)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
        idx = torch.cat([idx, next_token], dim=1)
    return idx


def main():
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GPT.from_pretrained('gpt2').to(device).eval()
    print(f"loaded gpt2 ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params) on {device}")

    # GPT-2 BPE id for 'Hello' is 15496
    prompt = torch.tensor([[15496]], dtype=torch.long, device=device)
    N = 20

    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = generate(model, prompt, max_new_tokens=N)
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    ids = out[0].tolist()
    print(f"\ngenerated {N} tokens in {elapsed*1000:.0f} ms ({elapsed*1000/N:.1f} ms/token)")
    print(f"token ids: {ids}")

    try:
        import tiktoken
        enc = tiktoken.get_encoding('gpt2')
        print(f"text:      {enc.decode(ids)!r}")
    except ImportError:
        print("(install tiktoken to see decoded text)")


if __name__ == '__main__':
    main()
