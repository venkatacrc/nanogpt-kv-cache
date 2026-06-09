---
layout: default
title: "nanoGPT Inference: An Empirical Step-by-Step Tutorial"

description: "KV Cache → FP16 → Rescues → FlashAttention → GQA → FP8 → Tensor Parallelism → Throughput Pareto"
---
*Benchmarked on a single NVIDIA H100; Tensor Parallelism runs on 2×H100.*

<!--
  POST OUTLINE for the section-by-section coaching notes
  (Goal / Show / Claim / Key points for each section).
  Replace the *[draft]* placeholders with prose.
-->

## 1. Why this post

*[draft section 1 here §1]*

## 2. Background: nanoGPT in 60 seconds

*[draft section 2 here §2]*

## 3. Adding the KV cache

*[draft section 3 here §3]*

![FP32 cache speedup, gpt2-xl](plots/cache_speedup_fp32.png)

## 4. The FP16 surprise

*[draft section 4 here §4]*

![FP32 vs FP16 cache speedup](plots/cache_speedup_fp32_vs_fp16.png)

## 5. Where does the time go?

*[draft section 5 here §5]*

![Per-step time decomposition](plots/time_decomposition_fp16.png)

![Cache speedup is independent of model size](plots/model_size.png)

## 6. Three rescues for the FP16 cache

### 6.1 Long prompts

*[draft §6.1]*

![Cache speedup vs prompt length](plots/long_prompt.png)

### 6.2 Batching

*[draft §6.2]*

![Cache speedup vs batch size](plots/batch.png)

### 6.3 torch.compile

*[draft §6.3]*

![torch.compile attacks the launch floor](plots/compile.png)

### 6.4 Stacking the rescues

*[draft §6.4]*

![Stacked rescues, predicted vs measured](plots/stack.png)

### 6.5 Long generation

*[draft §6.5]*

![Cache wins at long N](plots/long_generation.png)

## 7. Picking an attention backend

*[draft section 7 here §7]*

![Attention backends, prefill](plots/backends_prefill.png)

![Attention backends, decode](plots/backends_decode.png)

## 8. When does FlashAttention-3 actually win?

### 8.1 Llama-like shapes

*[draft §8.1]*

![FA3 prefill scaling with head_dim=128](plots/fa3_prefill_scaling.png)

### 8.2 The KV cache layout question

*[draft §8.2]*

![Cache layout closes the FA3 decode gap](plots/layout_decode.png)

![Batching does not close the residual decode gap](plots/batched_decode.png)

### 8.3 Does GQA finally make FA3 win decode?

*[draft §8.3]*

![GQA decode TPOT and KV cache memory](plots/gqa_decode.png)

![GQA preserves FA3's prefill advantage](plots/gqa_prefill.png)

### 8.4 What about FP8? (A cautionary tale)

*[draft §8.4]*

![Naive FP8 attention is slower AND wrong](plots/fp8.png)

## 9. What we learned

*[draft section 9 here §9]*

### 9.1 TTFT vs batch size

*TTFT is reported as `prefill + first decode step` (first-step approximated with TPOT from the same run).*

![TTFT vs batch size (H100, 1.25B GQA, FA3)](plots/fig_ttft_vs_batch.png)

### 9.2 Throughput vs per-user latency (Pareto)

![Pareto: throughput vs per-user latency](plots/fig_pareto.png)

## 10. References

- **Andrej Karpathy** — [nanoGPT](https://github.com/karpathy/nanoGPT). The
  architecture, the weight-loading-from-HF pattern, and the tasteful 200-line
  forward pass are all his.
- **Tri Dao et al.** — [FlashAttention](https://github.com/Dao-AILab/flash-attention)
  papers and the prebuilt FA3 wheel on H100 used in step-5/6.
- **PyTorch SDPA team** — the dispatcher that "just works" and picks Flash /
  cuDNN / memory-efficient under the hood without us asking.

---

*Code, logs, and plot-generation scripts are at [github.com/venkatacrc/nanogpt-kv-cache](https://github.com/venkatacrc/nanogpt-kv-cache).*
