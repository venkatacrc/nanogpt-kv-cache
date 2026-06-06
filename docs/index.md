---
layout: default
title: Adding a KV Cache to nanoGPT — An Empirical Tutorial
---

# Adding a KV Cache to nanoGPT

*An empirical tutorial with measurements from an NVIDIA H100.*

---

<!--
  POST OUTLINE — fill in each section below. Each section has:
    - Goal:   one sentence describing what this section accomplishes
    - Show:   the code snippet, table, or plot to include
    - Claim:  the headline finding/insight the reader should take away

  Replace this comment block with the actual prose as we go.
-->

## 1. Why this post

*[draft section 1 here]*

## 2. Background: nanoGPT in 60 seconds

*[draft section 2 here]*

## 3. Adding the KV cache

*[draft section 3 here]*

## 4. The FP16 surprise

*[draft section 4 here]*

![FP32 vs FP16 cache speedup](plots/cache_speedup_fp32_vs_fp16.png)

## 5. Where does the time go?

*[draft section 5 here]*

## 6. Three rescues for the FP16 cache

### 6.1 Long prompts
*[draft]*

### 6.2 Batching
*[draft]*

### 6.3 torch.compile
*[draft]*

### 6.4 Stacking the rescues
*[draft]*

## 7. Picking an attention backend

*[draft section 7 here]*

## 8. When does FlashAttention-3 actually win?

### 8.1 Llama-like shapes
*[draft]*

### 8.2 The KV cache layout question
*[draft]*

## 9. What we learned

*[draft section 9 here]*

## 10. References

*[draft section 10 here]*

---

*Code, logs, and plot-generation scripts are at [github.com/venkatacrc/nanogpt-kv-cache](https://github.com/venkatacrc/nanogpt-kv-cache).*
