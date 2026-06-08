"""inference.py — KVCache for the TP model.

Each rank holds the cache for its OWN n_kv_head_local heads. Cache layout is
[B, S, n_kv_head_local, D] (FA3-native), same as step-6b/step-8.
"""

import torch


class KVCache:
    def __init__(self, n_layer, B, max_seq_len, n_kv_head_local, head_dim,
                 device='cpu', dtype=torch.float32):
        self.n_layer     = n_layer
        self.max_seq_len = max_seq_len
        self.k = [torch.zeros(B, max_seq_len, n_kv_head_local, head_dim,
                              device=device, dtype=dtype) for _ in range(n_layer)]
        self.v = [torch.zeros(B, max_seq_len, n_kv_head_local, head_dim,
                              device=device, dtype=dtype) for _ in range(n_layer)]
        self.seq_len = 0

    def update(self, layer_idx, k_new, v_new):
        T_new = k_new.shape[1]
        assert self.seq_len + T_new <= self.max_seq_len
        self.k[layer_idx][:, self.seq_len:self.seq_len + T_new, :, :] = k_new
        self.v[layer_idx][:, self.seq_len:self.seq_len + T_new, :, :] = v_new

    def advance(self, T_new):
        self.seq_len += T_new

    def reset(self):
        self.seq_len = 0
