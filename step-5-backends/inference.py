"""inference.py — KVCache + greedy generation.

Two paths:
  generate(model, idx, N, cache=None)   uncached: full forward every step
  generate(model, idx, N, cache=cache)  cached:   prefill once, then decode N-1 times

Keeping both makes it easy to correctness-check the cached path against
the uncached reference (token IDs must match in FP32).
"""

import torch


class KVCache:
    """Pre-allocated per-layer KV cache for GPT-2 inference."""

    def __init__(self, n_layer, B, max_seq_len, n_head, head_dim,
                 device='cpu', dtype=torch.float32):
        self.n_layer     = n_layer
        self.max_seq_len = max_seq_len
        self.k = [torch.zeros(B, n_head, max_seq_len, head_dim,
                              device=device, dtype=dtype) for _ in range(n_layer)]
        self.v = [torch.zeros(B, n_head, max_seq_len, head_dim,
                              device=device, dtype=dtype) for _ in range(n_layer)]
        self.seq_len = 0

    def update(self, layer_idx, k_new, v_new):
        """Write k_new, v_new at positions [seq_len, seq_len + T_new) for layer_idx."""
        T_new = k_new.shape[2]
        assert self.seq_len + T_new <= self.max_seq_len, \
            f"cache overflow: {self.seq_len} + {T_new} > {self.max_seq_len}"
        self.k[layer_idx][:, :, self.seq_len:self.seq_len + T_new, :] = k_new
        self.v[layer_idx][:, :, self.seq_len:self.seq_len + T_new, :] = v_new

    def advance(self, T_new):
        """Bump seq_len once per forward pass, after all layers have updated."""
        self.seq_len += T_new

    def reset(self):
        """Logically clear (doesn't zero tensors)."""
        self.seq_len = 0


@torch.no_grad()
def generate(model, idx, max_new_tokens, cache=None):
    """Greedy sampling. Returns idx concatenated with the generated tokens."""
    if cache is None:
        # Reference path: full forward every step
        for _ in range(max_new_tokens):
            logits = model(idx)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

    # Cached path: one prefill, then max_new_tokens-1 single-token decodes
    logits = model(idx, cache)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tokens = [idx, next_token]
    for _ in range(max_new_tokens - 1):
        logits = model(next_token, cache)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens.append(next_token)
    return torch.cat(tokens, dim=1)
