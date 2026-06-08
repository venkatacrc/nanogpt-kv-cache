"""inference.py — KVCache + greedy generation.

This is the step-6b 'FA3-native layout' variant:
  cache.k[layer] : [B, max_seq_len, n_head, head_dim]   (was [B, n_head, max_seq_len, head_dim])
  cache.v[layer] : same shape

Why: FA3's torch.ops.flash_attn_3.fwd requires [B, S, H, D] input. With the
standard PyTorch [B, H, S, D] cache layout, every FA3 call has to .contiguous()-
transpose K and V (per layer per step), which we measured as the dominant cost
in step-6 decode. Storing the cache directly in FA3's layout removes that tax.

SDPA still needs [B, H, S, D] internally, so the SDPA path now pays its own
transpose-and-back tax. The empirical question step-6b answers: which is faster
overall — the cache-side transpose (old) or the SDPA-side transpose (new)?
"""

import torch


class KVCache:
    """Pre-allocated per-layer KV cache, stored in [B, S, H, D] layout (FA3-native)."""

    def __init__(self, n_layer, B, max_seq_len, n_head, head_dim,
                 device='cpu', dtype=torch.float32):
        self.n_layer     = n_layer
        self.max_seq_len = max_seq_len
        # NOTE: shape is [B, max_seq_len, n_head, head_dim] — second axis is the
        # token position, third is the head. This matches FA3's expected input.
        self.k = [torch.zeros(B, max_seq_len, n_head, head_dim,
                              device=device, dtype=dtype) for _ in range(n_layer)]
        self.v = [torch.zeros(B, max_seq_len, n_head, head_dim,
                              device=device, dtype=dtype) for _ in range(n_layer)]
        self.seq_len = 0

    def update(self, layer_idx, k_new, v_new):
        """Write k_new, v_new at positions [seq_len, seq_len + T_new) for layer_idx.
        k_new and v_new are expected to be in [B, T_new, n_head, head_dim] layout."""
        T_new = k_new.shape[1]                    # T is dim 1 in the new layout
        assert self.seq_len + T_new <= self.max_seq_len, \
            f"cache overflow: {self.seq_len} + {T_new} > {self.max_seq_len}"
        self.k[layer_idx][:, self.seq_len:self.seq_len + T_new, :, :] = k_new
        self.v[layer_idx][:, self.seq_len:self.seq_len + T_new, :, :] = v_new

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
