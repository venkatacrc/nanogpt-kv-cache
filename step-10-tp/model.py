"""model.py — Llama-3 8B-class custom GPT with Tensor Parallel + GQA support.

Megatron-style sharding:
  - c_attn (QKV projection):  COLUMN-parallel  (output dim sharded across ranks)
  - c_proj (attn output):     ROW-parallel + all-reduce
  - c_fc   (MLP up):          COLUMN-parallel
  - c_proj (MLP down):        ROW-parallel + all-reduce

Embeddings, LayerNorms, and LM head are replicated across ranks (cheap).
Each rank holds n_head/world_size attention heads and n_kv_head/world_size
KV heads. The KV cache is sharded along the head dimension.

Set tp.world_size=1 for the single-GPU baseline (no NCCL calls).

Architecture (matches Llama-3 8B layout, GPT-style MLP for simplicity):
  n_layer=32, n_head=32, n_kv_head=8 (GQA 4:1), n_embd=4096, head_dim=128
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

try:
    import flash_attn_3._C  # noqa: F401
    _HAS_FA3 = True
except ImportError:
    _HAS_FA3 = False


@dataclass
class GPTConfig:
    block_size:  int = 8192
    vocab_size:  int = 50257
    n_layer:     int = 32
    n_head:      int = 32
    n_kv_head:   int = 8       # GQA 4:1, Llama-3 style
    n_embd:      int = 4096
    bias:       bool = False   # Llama doesn't use bias on linears
    dropout:   float = 0.0

    def __post_init__(self):
        assert self.n_head % self.n_kv_head == 0


@dataclass
class TPState:
    rank: int = 0
    world_size: int = 1

    def all_reduce(self, t):
        if self.world_size > 1:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, eps=1e-5)


class CausalSelfAttention(nn.Module):
    USE_SDPA = False
    USE_FA3  = False

    def __init__(self, config, tp):
        super().__init__()
        self.tp = tp
        assert config.n_head    % tp.world_size == 0, "n_head not divisible by world_size"
        assert config.n_kv_head % tp.world_size == 0, "n_kv_head not divisible by world_size"

        self.n_head_local    = config.n_head    // tp.world_size
        self.n_kv_head_local = config.n_kv_head // tp.world_size
        self.head_dim        = config.n_embd // config.n_head

        local_q_dim  = self.n_head_local    * self.head_dim
        local_kv_dim = self.n_kv_head_local * self.head_dim
        # Column-parallel QKV: each rank produces its slice of Q, K, V.
        self.c_attn = nn.Linear(config.n_embd, local_q_dim + 2 * local_kv_dim, bias=config.bias)
        # Row-parallel output: input dim is sharded, output is full n_embd
        # (each rank computes a partial sum that we'll all-reduce).
        self.c_proj = nn.Linear(local_q_dim, config.n_embd, bias=config.bias)

    def forward(self, x, cache=None, layer_idx=None):
        B, T, _ = x.size()
        head_dim = self.head_dim

        qkv = self.c_attn(x)
        q_dim  = self.n_head_local    * head_dim
        kv_dim = self.n_kv_head_local * head_dim
        q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=2)

        # [B, T, H_local, D] layout
        q = q.view(B, T, self.n_head_local,    head_dim)
        k = k.view(B, T, self.n_kv_head_local, head_dim)
        v = v.view(B, T, self.n_kv_head_local, head_dim)

        if cache is not None:
            assert layer_idx is not None
            S = cache.seq_len
            cache.update(layer_idx, k, v)
            k = cache.k[layer_idx][:, :S + T, :, :]
            v = cache.v[layer_idx][:, :S + T, :, :]

        is_causal = (q.size(1) == k.size(1)) and q.size(1) > 1
        gqa_active = (self.n_kv_head_local != self.n_head_local)

        if CausalSelfAttention.USE_FA3:
            assert _HAS_FA3, "USE_FA3=True but flash_attn_3 not importable"
            out = torch.ops.flash_attn_3.fwd(
                q, k, v,
                softmax_scale=1.0 / math.sqrt(head_dim),
                is_causal=is_causal,
            )[0]                                            # [B, T, H_local, D]
        elif CausalSelfAttention.USE_SDPA:
            q_h = q.transpose(1, 2)
            k_h = k.transpose(1, 2)
            v_h = v.transpose(1, 2)
            out_h = F.scaled_dot_product_attention(
                q_h, k_h, v_h, is_causal=is_causal, enable_gqa=gqa_active,
            )
            out = out_h.transpose(1, 2)
        else:
            raise RuntimeError("set USE_SDPA or USE_FA3; manual attention "
                               "is too memory-heavy at this scale.")

        # [B, T, H_local, D] -> [B, T, local_q_dim]
        out = out.contiguous().view(B, T, q_dim)
        # Row-parallel: each rank's c_proj output is a partial sum; reduce.
        out = self.c_proj(out)
        out = self.tp.all_reduce(out)
        return out


class MLP(nn.Module):
    def __init__(self, config, tp):
        super().__init__()
        self.tp = tp
        intermediate = 4 * config.n_embd
        assert intermediate % tp.world_size == 0, "intermediate not divisible by world_size"
        local_intermediate = intermediate // tp.world_size
        # Column-parallel up-projection: each rank produces its slice.
        self.c_fc   = nn.Linear(config.n_embd, local_intermediate, bias=config.bias)
        # Row-parallel down-projection: input sharded, all-reduce after.
        self.c_proj = nn.Linear(local_intermediate, config.n_embd, bias=config.bias)

    def forward(self, x):
        h = self.c_fc(x)
        h = F.gelu(h, approximate='tanh')
        out = self.c_proj(h)
        out = self.tp.all_reduce(out)
        return out


class Block(nn.Module):
    def __init__(self, config, tp):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, tp)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config, tp)

    def forward(self, x, cache=None, layer_idx=None):
        x = x + self.attn(self.ln_1(x), cache=cache, layer_idx=layer_idx)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, tp):
        super().__init__()
        self.config = config
        self.tp = tp
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            h    = nn.ModuleList([Block(config, tp) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, cache=None):
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f"context {T} > block_size {self.config.block_size}"
        pos_start = cache.seq_len if cache is not None else 0
        pos = torch.arange(pos_start, pos_start + T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for layer_idx, block in enumerate(self.transformer.h):
            x = block(x, cache=cache, layer_idx=layer_idx)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if cache is not None:
            cache.advance(T)
        return logits


if __name__ == '__main__':
    cfg = GPTConfig()
    tp = TPState()
    model = GPT(cfg, tp)
    n = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"GPT(cfg) @ TP=1: {n:.2f}B params  (Llama-3 8B class)")
