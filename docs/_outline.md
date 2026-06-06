# Writing Guide — section-by-section coaching notes

> This file is a working doc for the author. Jekyll excludes underscore-prefixed
> files from the built site, so it won't appear on the published page.

For each section: **Goal** (what it accomplishes), **Show** (code/table/plot
to include), **Claim** (the headline takeaway), **Key points** (bullet outline
of what to mention).

---

## Section 1: Why this post

- **Goal**: hook the reader in 200-300 words.
- **Show**: prose only.
- **Claim**: this is not the typical "KV cache makes inference faster" post.
  The journey reveals real surprises that more polished writeups skip over.
- **Key points**:
  1. Most KV-cache writeups present one polished result. They skip the
     experiments that failed.
  2. This post builds the cache step-by-step into nanoGPT, measures on H100
     at every step, and includes the surprises.
  3. Tease 2-3 specific surprises (no spoilers on the mechanism):
     - FP16 KV cache can be *slower* than recomputing at small batch / short prompt.
     - FlashAttention-3 can be *slower* than PyTorch's SDPA in many shapes.
     - The biggest single rescue for the FP16 cache is `torch.compile`.
  4. End with: "after reading, you'll understand when the cache helps, when
     it hurts, and which optimizations actually work."

---

## Section 2: Background — nanoGPT in 60 seconds

- **Goal**: get the reader to the starting line. Anyone who's seen GPT-2 should
  skim this; anyone who hasn't should walk away knowing what a forward pass and
  a generation loop are.
- **Show**:
  - Tiny code snippet from `step-0-baseline/generate.py` of the naive loop:
    ```python
    for _ in range(N):
        logits = model(idx)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_token], dim=1)
    ```
- **Claim**: each generation step re-runs the *entire* forward pass on the full
  growing sequence. That's wasteful — most of the work is recomputing
  attention over tokens we already processed.
- **Key points**:
  1. nanoGPT is Andrej Karpathy's minimal GPT-2 implementation. We start
     from his model code (link).
  2. Greedy decoding loop: forward pass returns logits for every position;
     we keep the last position's argmax, append, repeat.
  3. After N steps the sequence is N tokens longer. Each forward pass redoes
     all the work for tokens 0..N-1. This is the problem the KV cache solves.
  4. Set up: "in step-1 we add a cache that stores K and V tensors per layer
     so each decode step only computes attention for the new token."

---

## Section 3: Adding the KV cache

- **Goal**: walk through the actual code changes that introduce the cache.
- **Show**:
  - The `KVCache` class from `step-1-kvcache/inference.py` (whole class, ~20
    lines). Highlight: pre-allocated buffers, `seq_len` tracker, `update()`
    and `advance()`.
  - The 5-line change in `CausalSelfAttention.forward` that uses the cache:
    ```python
    if cache is not None:
        S = cache.seq_len
        cache.update(layer_idx, k, v)
        k = cache.k[layer_idx][:, :, :S + T, :]
        v = cache.v[layer_idx][:, :, :S + T, :]
        mask = self.bias[:, :, S:S + T, :S + T]
    ```
  - Correctness check output from `step-1-kvcache/test_correctness.log`:
    cached and uncached generate IDENTICAL token IDs in FP32.
  - **PLOT 1**: `cache_speedup_fp32.png` — bar chart of TPOT uncached vs TPOT
    cached on gpt2-xl, with the speedup ratio annotated.
- **Claim**: in FP32, the KV cache delivers a clean ~3.7x speedup on gpt2-xl
  with N=500 tokens, and is bit-exact (every generated token matches).
- **Key points**:
  1. Explain `update()` (write K/V at the right slot) and `advance()` (bump
     seq_len once per forward pass, after all layers update).
  2. Mention the *one subtle thing*: the mask shifts. For a single-token decode,
     row 0 of the new mask attends to columns 0..S (all past + current).
  3. Show the FP32 numbers from `step-1-kvcache/bench_gpt2xl.log`:
     - uncached TPOT 35.41 ms, cached TPOT 9.50 ms (~3.7x faster per token)
     - total speedup 3.72x
  4. End with: "FP32 is the easy case. Now let's halve the precision and see
     what happens."

---

## Section 4: The FP16 surprise

- **Goal**: deliver the first counterintuitive finding. The same code, switched
  to FP16, behaves differently than expected.
- **Show**:
  - The one-line change to go FP16: `model.half()` + KVCache(dtype=torch.float16).
  - **PLOT 2**: `cache_speedup_fp32_vs_fp16.png` — side-by-side bars showing
    FP32 (3.72x) vs FP16 (1.44x) cache speedup on gpt2-xl. Same model, same
    N=500, same prompt — only precision differs.
  - Output from `step-2-fp16/bench.log`:
    - uncached TPOT 15.78 ms, cached TPOT 10.94 ms
    - cache speedup only 1.44x (vs FP32's 3.72x)
- **Claim**: FP16 cuts both paths' time, but it cuts the *uncached* path more
  than the cached path. So the cache's relative advantage shrinks. Worth
  investigating: is this real, or are we measuring wrong?
- **Key points**:
  1. State the observed numbers plainly.
  2. Hypothesize: the cached path is *less* affected by precision because it's
     not doing as much FP-heavy compute per step. Most of its time is spent in
     *fixed overhead* (kernel launches, Python dispatcher) regardless of
     precision. Foreshadow the launch-floor concept.
  3. Note: the FP16 result here used SINGLE TRIAL with light warmup. Later
     we'll see that careful re-measurement gives an even more dramatic result
     (the cache actually LOSES at B=1, P=1). For now, take 1.44x at face value
     and ask: where is the cache spending its time?
  4. End with: "to understand why FP16 changes the picture, we need to look
     INSIDE a decode step."

---

## Section 5: Where does the time go?

- **Goal**: decompose a generation step into prefill, cached decode, and
  uncached decode, and identify the launch floor.
- **Show**:
  - Snippet of `step-3-decomp/decomp.py` showing the per-operation timing
    pattern with `time_op` and median.
  - **PLOT 3**: `time_decomposition_fp16.png` — line plot, three lines for
    prefill / cached_decode / uncached_decode, y=ms, x=seq_len. Shows the
    ~10ms floor for cached_decode while uncached_decode grows.
  - Maybe a table of the actual numbers from `step-3-decomp/decomp.log`.
- **Claim**: the cached decode step has a **wall-clock floor of ~10 ms** on
  gpt2-xl FP16, set by Python/dispatcher overhead and per-layer kernel
  launches, not by compute. The uncached path, in contrast, scales with
  seq_len because it actually has to recompute attention.
- **Key points**:
  1. Define the three operations:
     - PREFILL: one forward at full prompt length (with cache update)
     - CACHED DECODE: one forward of 1 token against existing cache
     - UNCACHED DECODE: one forward of seq_len+1 tokens against no cache
       (equivalent to what the no-cache path does for the next step)
  2. Show the data: at seq_len=128, all three are ~10 ms (everything's at floor).
     At seq_len=512, cached_decode stays ~10 ms but uncached_decode climbs.
  3. Define "launch floor": each kernel launch on a GPU has fixed Python +
     CUDA-dispatch overhead of ~5-10 microseconds. nanoGPT's 48 layers × a few
     kernels each = ~200-300 kernel launches per forward. That's ~2-3 ms of
     pure overhead. Plus PyTorch's dispatcher / autograd checks / etc adds
     more. Total: ~10 ms per forward.
  4. The insight: the FP16 cache loses some of its advantage because the
     cached path was ALREADY at the launch floor. There's nothing left for
     precision to optimize. The uncached path, doing more compute, benefits
     more from going to FP16.
  5. Optional sidebar: this also explains step-4's surprise — single-trial
     uncached measurements suffer JIT compilation overhead because the GPU
     hasn't seen all the sequence-length shapes yet. With proper warmup, the
     uncached path is even faster, and the FP16 cache actually LOSES (0.92x).

---

## Section 6: Three rescues for the FP16 cache

- **Goal**: if the cache loses or just barely wins in FP16, what fixes it?
  Show three independent levers and how they stack.

### 6.1 Long prompts (`long_prompt.py`)

- **Show**:
  - Quick mention of the script's structure (sweeps prompt length P).
  - **PLOT 4**: `long_prompt.png` — cache speedup (y) vs prompt length P (x).
    Highlight crossover where cache stops losing.
  - Use numbers from `step-4-rescues/long_prompt.log` (median-of-3):
    - P=1: 0.91x  (cache loses)
    - P=128: 0.98x (break-even)
    - P=512: 1.67x (cache wins)
- **Claim**: longer prompts make the uncached path do more work per step
  (it has to re-attend the whole prefix), so the cache's relative advantage
  grows. Crossover around P=128.

### 6.2 Batching (`batch.py`)

- **Show**:
  - **PLOT 5**: `batch.png` — cache speedup (y) vs batch size B (x).
  - Numbers from `step-4-rescues/batch.log`:
    - B=1: 0.91x  (loss)
    - B=4: 1.70x
    - B=8: 2.91x
- **Claim**: batching amortizes the per-step launch floor across B sequences.
  Same wall-clock per step processes B× more work. Cache speedup nearly
  triples just by batching.

### 6.3 torch.compile (`compile.py`)

- **Show**:
  - **PLOT 6**: `compile.png` — TPOT bar chart comparing
    {uncached_eager, cached_eager, cached_default, cached_reduceoh}.
  - Numbers from `step-4-rescues/compile.log`:
    - uncached_eager TPOT 9.92 ms
    - cached_eager TPOT 10.93 ms  (cache loses)
    - cached_compile (default) TPOT 5.89 ms  (cache wins 1.68x)
    - cached_compile (reduce-overhead) TPOT 6.02 ms
- **Claim**: `torch.compile` lowers the launch floor itself by fusing kernels
  and cutting dispatcher overhead. TPOT drops by 43% at B=1. The cache flips
  from loss to win. Note: `reduce-overhead` mode is slightly slower than
  `default` here because our dynamic-shape K-slice triggers per-step graph
  recapture — a real teachable failure mode.

### 6.4 Stacking the rescues (`stack.py`)

- **Show**:
  - **PLOT 7**: `stack.png` — bar chart comparing measured vs predicted
    speedup for compile + batch B=8.
  - Numbers from `step-4-rescues/stack.log`:
    - batch alone (B=8): 3.06x
    - compile factor: 1.74x
    - predicted product: 5.34x
    - measured combined: 4.31x
    - efficiency: 81%
- **Claim**: rescues stack but **sub-multiplicatively**. The 19% loss is the
  shared bottleneck: both compile and batching attack the launch floor, so
  applying both has diminishing returns. Headline: 4.3× cache speedup at the
  worst-case setting, where eager FP16 was a 0.91× loss.

---

## Section 7: Picking an attention backend

- **Goal**: introduce the three attention backends (manual, SDPA, FA3) and
  show which wins under what conditions.
- **Show**:
  - Code diff: the class-flag pattern in `step-5-backends/model.py` that
    lets us switch backends at runtime.
  - **PLOT 8**: `backends_prefill.png` — manual vs SDPA vs FA3 prefill latency
    at S=128/256/512/1024 on gpt2-xl FP16. Numbers from `bench_backends.log`.
  - **PLOT 9**: `backends_decode.png` — same backends, cached decode TPOT
    sweep.
  - Quick mention of the non-determinism finding from `test_backends.log`:
    same script, same seed, different runs sometimes give different FP16
    SDPA tokens. Production teams pin `torch.backends.cudnn.deterministic`
    for this reason.
- **Claim**: SDPA always beats manual (1.3-2.2x). FA3, surprisingly, *loses*
  to SDPA on gpt2-xl by 5-24%. The culprit: gpt2-xl's `[B, H, S, D]` cache
  layout forces FA3 to `.contiguous()`-transpose every forward, and at
  gpt2-xl's modest shapes that tax outweighs the kernel savings.

---

## Section 8: When does FlashAttention-3 actually win?

### 8.1 Llama-like shapes (`step-6-shapes/`)

- **Show**:
  - Quick note that we build a 1.4B-param model from scratch with
    head_dim=128 and block_size=16K. Random init is fine for timing studies.
  - **PLOT 10**: `fa3_prefill_scaling.png` — FA3/SDPA prefill ratio across
    S=128 to 16K. Shows the crossover and the growing FA3 advantage.
  - Numbers from `step-6-shapes/bench_shapes.log`.
- **Claim**: with `head_dim=128` (FA3's sweet spot) and enough sequence
  length, FA3 *does* beat SDPA in prefill: 1.11x at S=2K, 1.37x at S=16K.
  But decode is unchanged — FA3 still loses ~12%.

### 8.2 The KV cache layout question (`step-6b-layout/`)

- **Show**:
  - The 1-line change in `KVCache.__init__`: cache shape goes from
    `[B, H, max_seq, D]` to `[B, max_seq, H, D]`.
  - The corresponding `CausalSelfAttention.forward` diff: Q/K/V kept in
    `[B, T, H, D]` throughout, SDPA path now does the transpose.
  - **PLOT 11**: `layout_decode.png` — bar chart of FA3 decode TPOT before
    vs after layout refactor at S_pre=4096, B=1.
  - **PLOT 12**: `batched_decode.png` — FA3/SDPA decode ratio across B=1/4/8
    in the new layout. Show that batching does NOT close the residual gap.
- **Claim**: the layout refactor closes most of the FA3 decode gap (from
  0.87x to 0.96x). The remaining 4-5% is intrinsic kernel quality at
  q_len=1 — SDPA has a specialized decode fast-path that FA3 doesn't.
  Batching doesn't fix it either. To make FA3 win decode you need
  paged attention, GQA, or both.

---

## Section 9: What we learned

- **Goal**: synthesis. 4-5 takeaways the reader walks away with.
- **Show**: just prose. Maybe a small summary table.
- **Claim**: the empirical journey has crystallized into a small set of
  rules-of-thumb for KV caching and attention backends.
- **Key points** (each ~30-50 words):
  1. **The KV cache is conditional.** It wins in FP32 always. In FP16 it
     needs at least one of: long prompts, batching, or `torch.compile` to
     beat the uncached path on small models.
  2. **The launch floor is real.** ~10 ms per cached decode on gpt2-xl is
     not attention; it's Python/dispatcher overhead. Optimizations that don't
     attack the floor (e.g. better FP precision) have limited effect on the
     cached path.
  3. **Backend choice is shape-dependent.** SDPA wins for small/short
     workloads. FA3 wins for long prefill at head_dim=128. Decode is
     SDPA's territory unless you go to large batch + paged attention.
  4. **Layout matters more than people think.** A 9-percentage-point swing
     in FA3 decode performance came from changing only the cache tensor's
     dimension order, not the kernel.
  5. **Single-trial measurements lie.** First-shape JIT overhead and FP16
     non-determinism make naive benchmarks misleading. Always use median of
     multiple trials with a beefed-up warmup that touches every shape you
     intend to measure.

---

## Section 10: References

- **Goal**: cite the work this builds on.
- **Key references**:
  1. Karpathy, A. nanoGPT. https://github.com/karpathy/nanoGPT
  2. Dao, T., Fu, D., Ermon, S., Rudra, A., Ré, C. FlashAttention. NeurIPS 2022.
  3. Dao, T. FlashAttention-2. https://arxiv.org/abs/2307.08691
  4. Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., Dao, T.
     FlashAttention-3. https://arxiv.org/abs/2407.08608
  5. PyTorch scaled_dot_product_attention documentation:
     https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
  6. vLLM (paged attention): Kwon et al. SOSP 2023. https://arxiv.org/abs/2309.06180
  7. Pope, R. et al. Efficiently Scaling Transformer Inference. 2022.
     https://arxiv.org/abs/2211.05102
  8. (Add any specific blog posts you read while learning — Jay Alammar's
     Illustrated GPT-2, Pieter Delobelle's KV cache post, etc.)

---

## Plot list (12 total)

| #   | File                              | Section | Source log                                          |
|-----|-----------------------------------|---------|------------------------------------------------------|
| 1   | cache_speedup_fp32.png            | 3       | step-1-kvcache/bench_gpt2xl.log                      |
| 2   | cache_speedup_fp32_vs_fp16.png    | 4       | step-1-kvcache/bench_gpt2xl.log + step-2-fp16/bench.log |
| 3   | time_decomposition_fp16.png       | 5       | step-3-decomp/decomp.log                             |
| 4   | long_prompt.png                   | 6.1     | step-4-rescues/long_prompt.log                       |
| 5   | batch.png                         | 6.2     | step-4-rescues/batch.log                             |
| 6   | compile.png                       | 6.3     | step-4-rescues/compile.log                           |
| 7   | stack.png                         | 6.4     | step-4-rescues/stack.log                             |
| 8   | backends_prefill.png              | 7       | step-5-backends/bench_backends.log                   |
| 9   | backends_decode.png               | 7       | step-5-backends/bench_backends.log                   |
| 10  | fa3_prefill_scaling.png           | 8.1     | step-6-shapes/bench_shapes.log                       |
| 11  | layout_decode.png                 | 8.2     | step-6/+step-6b decode numbers                       |
| 12  | batched_decode.png                | 8.2     | step-6b-layout/bench_batch_decode.log                |

Plot generation scripts live in `tools/plots/`. One file per plot, named
`fig_<name>.py`. Each script hardcodes the numbers (cited via comment from
the source log) and writes to `docs/plots/<name>.png` using `plot_style.py`.
