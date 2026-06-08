"""model.py - GPT-2 with Grouped-Query Attention (GQA).

Same FA3-native [B, S, H, D] layout as step-6b. The new piece is GQA:
  - Q has n_head     heads of width head_dim
  - K, V have n_kv_head heads of width head_dim   (with n_head % n_kv_head == 0)
  - Each KV head is shared by n_head / n_kv_head Q heads (the "group")

Why GQA matters:
  - KV cache is n_head / n_kv_head times SMALLER per layer
  - Decode is bandwidth-bound on K/V cache loads; less bandwidth -> faster
  - This is the architecture every modern LLM uses (Llama-2/3, Mistral, ...)

Backend selection (manual / SDPA / FA3) is unchanged. SDPA gets GQA via
enable_gqa=True (PyTorch 2.5+); FA3 just takes K, V with fewer heads
and the kernel handles grouping internally.

Setting n_kv_head == n_head gives the original Multi-Head Attention (MHA).
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import flash_attn_3._C  # noqa: F401  (registers torch.ops.flash_attn_3.*)
    _HAS_FA3 = True
except ImportError:
    _HAS_FA3 = False


@dataclass
class GPTConfig:
    block_size:  int = 1024   # max context length
    vocab_size:  int = 50257  # GPT-2 BPE vocabulary
    n_layer:     int = 12
    n_head:      int = 12
    n_kv_head:   int = None   # GQA: # of KV heads. None -> MHA (n_kv_head = n_head)
    n_embd:      int = 768
    bias:       bool = True   # GPT-2 uses bias on Linear/LayerNorm
    dropout:   float = 0.0    # we're inference-only

    def __post_init__(self):
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        assert self.n_head % self.n_kv_head == 0, \
            f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})"


class LayerNorm(nn.Module):
    """LayerNorm with optional bias (PyTorch's didn't always support bias=False)."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, eps=1e-5)


class CausalSelfAttention(nn.Module):
    # Backend selection (class-level flags; flip at runtime).
    # Default (both False) is the manual path, identical to step-4.
    # USE_FA3 takes precedence over USE_SDPA when both are True.
    USE_SDPA = False
    USE_FA3  = False

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head    = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd    = config.n_embd
        self.head_dim  = config.n_embd // config.n_head
        self.block_size = config.block_size
        # GQA: Q has n_head * head_dim out features; K and V each have
        # n_kv_head * head_dim. We keep ONE fused projection for cache-friendliness
        # with the existing HF GPT-2 weights (which are MHA — when n_kv_head ==
        # n_head this is identical to step-6b's c_attn).
        q_out  = self.n_head    * self.head_dim
        kv_out = self.n_kv_head * self.head_dim
        self.c_attn = nn.Linear(config.n_embd, q_out + 2 * kv_out, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Lazy causal mask: only allocated when the manual attention path needs it.
        self.register_buffer("bias", None, persistent=False)

    def _causal_mask(self, device):
        if self.bias is None:
            m = torch.tril(torch.ones(self.block_size, self.block_size, device=device))
            self.bias = m.view(1, 1, self.block_size, self.block_size)
        return self.bias

    def forward(self, x, cache=None, layer_idx=None):
        B, T, C = x.size()
        head_dim = self.head_dim

        qkv = self.c_attn(x)                                # [B, T, q_out + 2 kv_out]
        q_out  = self.n_head    * head_dim
        kv_out = self.n_kv_head * head_dim
        q, k, v = qkv.split([q_out, kv_out, kv_out], dim=2)

        # KEEP [B, T, H, D] layout (no initial transpose to [B, H, T, D]).
        # Q has n_head heads; K and V have n_kv_head heads (GQA).
        q = q.view(B, T, self.n_head,    head_dim)
        k = k.view(B, T, self.n_kv_head, head_dim)
        v = v.view(B, T, self.n_kv_head, head_dim)

        if cache is not None:
            assert layer_idx is not None, "layer_idx required when cache is provided"
            S = cache.seq_len
            cache.update(layer_idx, k, v)                   # write [B, T, H, D] directly
            k = cache.k[layer_idx][:, :S + T, :, :]         # read [B, S+T, H, D]
            v = cache.v[layer_idx][:, :S + T, :, :]

        # Causal mask needed only when q_len == k_len > 1 (prefill / no-cache).
        is_causal = (q.size(1) == k.size(1)) and q.size(1) > 1

        gqa_active = (self.n_kv_head != self.n_head)

        if CausalSelfAttention.USE_FA3:
            assert _HAS_FA3, "USE_FA3=True but flash_attn_3 is not importable"
            # FA3 handles GQA natively: Q has H heads, K/V have G heads, the
            # kernel reads H/G mapping from tensor shapes. No repeat needed.
            out_fa = torch.ops.flash_attn_3.fwd(
                q, k, v,
                softmax_scale=1.0 / math.sqrt(head_dim),
                is_causal=is_causal,
            )[0]                                            # [B, S, H, D]
            y = out_fa
        elif CausalSelfAttention.USE_SDPA:
            # SDPA wants [B, H, T, D]. enable_gqa=True tells SDPA that Q's head
            # count exceeds K/V's; on supported backends (FlashAttention-2) this
            # is handled kernel-side, otherwise it falls back to repeat_interleave.
            q_h = q.transpose(1, 2)
            k_h = k.transpose(1, 2)
            v_h = v.transpose(1, 2)
            y_h = F.scaled_dot_product_attention(
                q_h, k_h, v_h, is_causal=is_causal, enable_gqa=gqa_active,
            )
            y = y_h.transpose(1, 2)
        else:
            # Manual path: expand K, V across the group axis (small models only).
            q_h = q.transpose(1, 2)                         # [B, H_q,  T,   D]
            k_h = k.transpose(1, 2)                         # [B, H_kv, S+T, D]
            v_h = v.transpose(1, 2)
            if gqa_active:
                group = self.n_head // self.n_kv_head
                k_h = k_h.repeat_interleave(group, dim=1)   # [B, H_q, S+T, D]
                v_h = v_h.repeat_interleave(group, dim=1)
            bias = self._causal_mask(x.device)
            if cache is not None:
                mask = bias[:, :, S:S + T, :S + T]
            else:
                mask = bias[:, :, :T, :T]
            att = (q_h @ k_h.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
            att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y_h = att @ v_h
            y = y_h.transpose(1, 2)

        # y is [B, T, H, D] regardless of backend; merge heads.
        y = y.contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x, approximate='tanh')
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x, cache=None, layer_idx=None):
        x = x + self.attn(self.ln_1(x), cache=cache, layer_idx=layer_idx)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying: lm_head shares wte's weight matrix
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, cache=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"context length {T} > block_size {self.config.block_size}"

        # When the cache holds S past tokens, the new T tokens are at positions [S, S+T)
        pos_start = cache.seq_len if cache is not None else 0
        pos = torch.arange(pos_start, pos_start + T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)               # [B, T, n_embd]
        pos_emb = self.transformer.wpe(pos)               # [T, n_embd]
        x = tok_emb + pos_emb

        for layer_idx, block in enumerate(self.transformer.h):
            x = block(x, cache=cache, layer_idx=layer_idx)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                          # [B, T, vocab_size]

        if cache is not None:
            cache.advance(T)                              # bump seq_len AFTER all layers updated
        return logits

    @classmethod
    def from_pretrained(cls, model_type='gpt2'):
        """Load GPT-2 weights from HF transformers (MHA only; GQA needs custom init)."""
        from transformers import GPT2LMHeadModel

        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),   # 124M
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 355M
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),  # 774M
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias']       = True
        config = GPTConfig(**config_args)

        model = cls(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf
                      if not k.endswith('.attn.masked_bias')
                      and not k.endswith('.attn.bias')]

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                      'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), "key count mismatch"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model


if __name__ == '__main__':
    cfg = GPTConfig()
    model = GPT(cfg)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"GPT(cfg) param count: {n_params:.2f}M  (expect 124.44M)")
