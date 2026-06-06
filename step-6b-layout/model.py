"""model.py - GPT-2 with FA3-native cache layout.

Same backends as step-5/6 (manual / SDPA / FA3) but the internal data layout
is now [B, T, H, D] throughout, matching the [B, S, H, D] KVCache in this step.

Per-backend cost of this layout choice (vs step-6's [B, H, S, D] layout):
  - manual: +1 transpose per Q,K,V trio per layer  (small)
  - SDPA:   +1 transpose per Q,K,V trio per layer  (small; SDPA may handle non-contig)
  - FA3:    -3 transposes + -3 .contiguous() copies per layer  (BIG win in decode)

Backend selection unchanged; flip via class-level flags.
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
    block_size: int = 1024   # max context length
    vocab_size: int = 50257  # GPT-2 BPE vocabulary
    n_layer:    int = 12
    n_head:     int = 12
    n_embd:     int = 768
    bias:      bool = True   # GPT-2 uses bias on Linear/LayerNorm
    dropout:  float = 0.0    # we're inference-only


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
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        # Lazy causal mask: only allocated when the manual attention path needs it.
        # SDPA / FA3 handle masking via is_causal, so for those backends we never
        # materialize this (block_size^2 FP32 is huge at long contexts: 16K -> 1 GB
        # PER LAYER, which would OOM the GPU at 48 layers).
        self.register_buffer("bias", None, persistent=False)

    def _causal_mask(self, device):
        if self.bias is None:
            m = torch.tril(torch.ones(self.block_size, self.block_size, device=device))
            self.bias = m.view(1, 1, self.block_size, self.block_size)
        return self.bias

    def forward(self, x, cache=None, layer_idx=None):
        B, T, C = x.size()

        qkv = self.c_attn(x)                                # [B, T, 3C]
        q, k, v = qkv.split(self.n_embd, dim=2)             # each [B, T, C]

        head_dim = C // self.n_head
        # KEEP [B, T, H, D] layout (no initial transpose to [B, H, T, D]).
        # This matches the cache layout and FA3's expected input.
        q = q.view(B, T, self.n_head, head_dim)
        k = k.view(B, T, self.n_head, head_dim)
        v = v.view(B, T, self.n_head, head_dim)

        if cache is not None:
            assert layer_idx is not None, "layer_idx required when cache is provided"
            S = cache.seq_len
            cache.update(layer_idx, k, v)                   # write [B, T, H, D] directly
            k = cache.k[layer_idx][:, :S + T, :, :]         # read [B, S+T, H, D]
            v = cache.v[layer_idx][:, :S + T, :, :]

        # Causal mask needed only when q_len == k_len > 1 (prefill / no-cache).
        is_causal = (q.size(1) == k.size(1)) and q.size(1) > 1

        if CausalSelfAttention.USE_FA3:
            assert _HAS_FA3, "USE_FA3=True but flash_attn_3 is not importable"
            # Q, K, V already in [B, S, H, D] — FA3's native layout. The cache slice
            # is contiguous in (H, D) because we only sliced the LEADING S dim.
            out_fa = torch.ops.flash_attn_3.fwd(
                q, k, v,
                softmax_scale=1.0 / math.sqrt(head_dim),
                is_causal=is_causal,
            )[0]                                            # [B, S, H, D]
            y = out_fa                                      # stays [B, S, H, D]
        elif CausalSelfAttention.USE_SDPA:
            # SDPA wants [N, ..., L, E] with heads as a batch-style axis at position 1.
            # Transpose Q to [B, H, T, D] and K, V to [B, H, S+T, D]. These views are
            # non-contiguous; SDPA handles that internally (no .contiguous() needed).
            q_h = q.transpose(1, 2)
            k_h = k.transpose(1, 2)
            v_h = v.transpose(1, 2)
            y_h = F.scaled_dot_product_attention(q_h, k_h, v_h, is_causal=is_causal)
            y = y_h.transpose(1, 2)                         # back to [B, T, H, D]
        else:
            q_h = q.transpose(1, 2)
            k_h = k.transpose(1, 2)
            v_h = v.transpose(1, 2)
            bias = self._causal_mask(x.device)
            if cache is not None:
                mask = bias[:, :, S:S + T, :S + T]
            else:
                mask = bias[:, :, :T, :T]
            att = (q_h @ k_h.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
            att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y_h = att @ v_h
            y = y_h.transpose(1, 2)                         # back to [B, T, H, D]

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
        """Load GPT-2 weights from HF transformers."""
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
