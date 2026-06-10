---
layout: default
title: "nanoGPT Inference: An Empirical Step-by-Step Tutorial"
description: "KV Cache → FP16 → Rescues → FlashAttention → GQA → FP8 → Tensor Parallelism → Throughput Pareto"
---

<style>
.guide-layout { display: grid; grid-template-columns: 280px minmax(0, 1fr); gap: 2rem; align-items: start; }
.guide-sidebar { position: sticky; top: 1rem; max-height: calc(100vh - 2rem); overflow: auto; border: 1px solid #d0d7de; border-radius: 8px; padding: 1rem; background: #f6f8fa; }
.guide-sidebar h3 { margin-top: 0; }
.guide-sidebar ul { margin: 0; padding-left: 1.2rem; }
.guide-main section { scroll-margin-top: 1rem; margin-bottom: 2.5rem; padding-bottom: 1.5rem; border-bottom: 1px solid #d8dee4; }
@media (max-width: 960px) {
  .guide-layout { grid-template-columns: 1fr; }
  .guide-sidebar { position: static; max-height: none; }
}
</style>

*Benchmarked on a single NVIDIA H100; Tensor Parallelism runs on 2×H100.*

<div class="guide-layout">
<aside class="guide-sidebar">
<h3>Guide Navigation</h3>
<ul>
  <li><a href="#step-0-baseline">Step 0 - Baseline</a></li>
  <li><a href="#step-1-kv-cache">Step 1 - KV Cache</a></li>
  <li><a href="#step-2-fp16">Step 2 - FP16</a></li>
  <li><a href="#step-3-decomposition">Step 3 - Decomposition</a></li>
  <li><a href="#step-4-rescues">Step 4 - Rescues</a></li>
  <li><a href="#step-5-backends">Step 5 - Backends</a></li>
  <li><a href="#step-6-shapes">Step 6 - Shapes</a></li>
  <li><a href="#step-6b-layout">Step 6b - Layout</a></li>
  <li><a href="#step-7-extras">Step 7 - Extras</a></li>
  <li><a href="#step-8-gqa">Step 8 - GQA</a></li>
  <li><a href="#step-9-fp8">Step 9 - FP8</a></li>
  <li><a href="#step-10-tp">Step 10 - TP</a></li>
  <li><a href="#step-11-pareto">Step 11 - Pareto</a></li>
</ul>
</aside>

<main class="guide-main">

<section id="step-0-baseline">

## Step 0 — Baseline greedy decode

### What we're building
One sentence: implement full-recompute greedy generation in `step-0-baseline/model.py`.

### The idea
Concept explanation in plain English. No code yet.  
What problem does this solve? What does it change?

### The implementation
Key code change only - not the full file.

```python
# TODO: add annotated snippet from step-0-baseline/model.py
```

[Full code: step-0-baseline/model.py](https://github.com/venkatacrc/nanogpt-kv-cache/blob/main/step-0-baseline/model.py)

### Running it

```bash
cd step-0-baseline/
python bench.py
```

### Results

### What the numbers mean
Walk through the output line by line.  
Be explicit about what was expected vs. what happened.

### Baseline decode timing
![Step 0 plot](plots/cache_speedup_fp32.png)
*What to look at: establish the no-cache reference before any optimization.*

### What we learned
One or two sentences. What does this step add to the overall picture? What question does it open for the next step?

</section>

<section id="step-1-kv-cache">

## Step 1 — Add KV cache

### What we're building
One sentence: add `KVCache` and cache-aware attention in `step-1-kvcache/model.py`.

### The idea
Concept explanation in plain English. No code yet.  
What problem does this solve? What does it change?

### The implementation
Key code change only - not the full file.

```python
# TODO: add annotated KV cache snippet
```

[Full code: step-1-kvcache/model.py](https://github.com/venkatacrc/nanogpt-kv-cache/blob/main/step-1-kvcache/model.py)

### Running it

```bash
cd step-1-kvcache/
python bench.py
```

### Results

### What the numbers mean
Walk through the output line by line.  
Be explicit about what was expected vs. what happened.

### FP32 cache speedup
![Step 1 plot](plots/cache_speedup_fp32.png)
*What to look at: verify cache decode approaches Nx improvement in FP32.*

### What we learned
One or two sentences. What does this step add to the overall picture? What question does it open for the next step?

</section>

<section id="step-2-fp16">

## Step 2 — Move to FP16

### What we're building
One sentence: run the same cache path in half precision in `step-2-fp16/model.py`.

### The idea
Concept explanation in plain English. No code yet.  
What problem does this solve? What does it change?

### The implementation
Key code change only - not the full file.

```python
# TODO: add annotated FP16 change snippet
```

[Full code: step-2-fp16/model.py](https://github.com/venkatacrc/nanogpt-kv-cache/blob/main/step-2-fp16/model.py)

### Running it

```bash
cd step-2-fp16/
python bench.py
```

### Results

### What the numbers mean
Walk through the output line by line.  
Be explicit about what was expected vs. what happened.

### FP32 vs FP16 cache speedup
![Step 2 plot](plots/cache_speedup_fp32_vs_fp16.png)
*What to look at: identify the surprising small-batch FP16 regression.*

### What we learned
One or two sentences. What does this step add to the overall picture? What question does it open for the next step?

</section>

<section id="step-3-decomposition">

## Step 3 — Time decomposition

### What we're building
One sentence: isolate per-step latency components in `step-3-decomp/decomp.py`.

### The idea
Concept explanation in plain English. No code yet.  
What problem does this solve? What does it change?

### The implementation
Key code change only - not the full file.

```python
# TODO: add annotated timing-decomposition snippet
```

[Full code: step-3-decomp/decomp.py](https://github.com/venkatacrc/nanogpt-kv-cache/blob/main/step-3-decomp/decomp.py)

### Running it

```bash
cd step-3-decomp/
python decomp.py
```

### Results

### What the numbers mean
Walk through the output line by line.  
Be explicit about what was expected vs. what happened.

### Per-step time decomposition
![Step 3 plot](plots/time_decomposition_fp16.png)
*What to look at: separate launch overhead from true math work.*

### What we learned
One or two sentences. What does this step add to the overall picture? What question does it open for the next step?

</section>

<section id="step-4-rescues">

## Step 4 — Rescue strategies

### What we're building
One sentence: test long prompts, batching, and compile rescues in `step-4-rescues/`.

### The idea
Concept explanation in plain English. No code yet.  
What problem does this solve? What does it change?

### The implementation
Key code change only - not the full file.

```python
# TODO: add one representative rescue snippet
```

[Full code: step-4-rescues/model.py](https://github.com/venkatacrc/nanogpt-kv-cache/blob/main/step-4-rescues/model.py)

### Running it

```bash
cd step-4-rescues/
python compile.py
```

### Results

### What the numbers mean
Walk through the output line by line.  
Be explicit about what was expected vs. what happened.

### Stacked rescues
![Step 4 plot](plots/stack.png)
*What to look at: compare measured stacked gains vs naive multiplied expectations.*

### What we learned
One or two sentences. What does this step add to the overall picture? What question does it open for the next step?

</section>

<section id="step-5-backends">

## Step 5 — Attention backend choices

### What we're building
One sentence: add backend selection (manual/SDPA/FA3) in `step-5-backends/model.py`.

### The idea
Concept explanation in plain English. No code yet.  
What problem does this solve? What does it change?

### The implementation
Key code change only - not the full file.

```python
# TODO: add annotated backend-dispatch snippet
```

[Full code: step-5-backends/model.py](https://github.com/venkatacrc/nanogpt-kv-cache/blob/main/step-5-backends/model.py)

### Running it

```bash
cd step-5-backends/
python bench_backends.py
```

### Results

### What the numbers mean
Walk through the output line by line.  
Be explicit about what was expected vs. what happened.

### Attention backends (decode)
![Step 5 plot](plots/backends_decode.png)
*What to look at: contrast prefill winners with decode behavior.*

### What we learned
One or two sentences. What does this step add to the overall picture? What question does it open for the next step?

</section>

<section id="step-6-shapes">

## Step 6 — Llama-like shapes

### What we're building
One sentence: test Hopper-friendly dimensions in `step-6-shapes/model.py`.

### The idea
Concept explanation in plain English. No code yet.  
What problem does this solve? What does it change?

### The implementation
Key code change only - not the full file.

```python
# TODO: add annotated shape/config snippet
```

[Full code: step-6-shapes/model.py](https://github.com/venkatacrc/nanogpt-kv-cache/blob/main/step-6-shapes/model.py)

### Running it

```bash
cd step-6-shapes/
python bench_shapes.py
```

### Results

### What the numbers mean
Walk through the output line by line.  
Be explicit about what was expected vs. what happened.

### FA3 prefill scaling
![Step 6 plot](plots/fa3_prefill_scaling.png)
*What to look at: find the context length where FA3 overtakes SDPA for prefill.*

### What we learned
One or two sentences. What does this step add to the overall picture? What question does it open for the next step?

</section>

<section id="step-6b-layout">

## Step 6b — KV layout refactor

### What we're building
One sentence: align cache memory layout with FA3 expectations in `step-6b-layout/model.py`.

### The idea
Concept explanation in plain English. No code yet.  
What problem does this solve? What does it change?

### The implementation
Key code change only - not the full file.

```python
# TODO: add annotated layout-change snippet
```

[Full code: step-6b-layout/model.py](https://github.com/venkatacrc/nanogpt-kv-cache/blob/main/step-6b-layout/model.py)

### Running it

```bash
cd step-6b-layout/
python bench_layout.py
```

### Results

### What the numbers mean
Walk through the output line by line.  
Be explicit about what was expected vs. what happened.

### Layout impact on decode
![Step 6b plot](plots/layout_decode.png)
*What to look at: estimate how much of the FA3 decode gap came from layout alone.*

### What we learned
One or two sentences. What does this step add to the overall picture? What question does it open for the next step?

</section>

<section id="step-7-extras">

## Step 7 — Extra stress tests

### What we're building
One sentence: add model-size and long-generation sweeps in `step-7-extras/`.

### The idea
Concept explanation in plain English. No code yet.  
What problem does this solve? What does it change?

### The implementation
Key code change only - not the full file.

```python
# TODO: add annotated sweep-loop snippet
```

[Full code: step-7-extras/model.py](https://github.com/venkatacrc/nanogpt-kv-cache/blob/main/step-7-extras/model.py)

### Running it

```bash
cd step-7-extras/
python long_generation.py
```

### Results

### What the numbers mean
Walk through the output line by line.  
Be explicit about what was expected vs. what happened.

### Cache speedup vs generation length
![Step 7 plot](plots/long_generation.png)
*What to look at: verify cache benefits grow as generated sequence length increases.*

### What we learned
One or two sentences. What does this step add to the overall picture? What question does it open for the next step?

</section>

<section id="step-8-gqa">

## Step 8 — Grouped Query Attention (GQA)

### What we're building
One sentence: add GQA-capable attention path in `step-8-gqa/model.py`.

### The idea
Concept explanation in plain English. No code yet.  
What problem does this solve? What does it change?

### The implementation
Key code change only - not the full file.

```python
# TODO: add annotated GQA head-group snippet
```

[Full code: step-8-gqa/model.py](https://github.com/venkatacrc/nanogpt-kv-cache/blob/main/step-8-gqa/model.py)

### Running it

```bash
cd step-8-gqa/
python bench_gqa.py
```

### Results

### What the numbers mean
Walk through the output line by line.  
Be explicit about what was expected vs. what happened.

### GQA decode and KV memory
![Step 8 plot](plots/gqa_decode.png)
*What to look at: compare latency shifts against KV-memory savings as `n_kv_head` changes.*

### What we learned
One or two sentences. What does this step add to the overall picture? What question does it open for the next step?

</section>

<section id="step-9-fp8">

## Step 9 — FP8 experiment

### What we're building
One sentence: test a naive FP8 attention path in `step-9-fp8/model.py`.

### The idea
Concept explanation in plain English. No code yet.  
What problem does this solve? What does it change?

### The implementation
Key code change only - not the full file.

```python
# TODO: add annotated FP8 cast/scaling snippet
```

[Full code: step-9-fp8/model.py](https://github.com/venkatacrc/nanogpt-kv-cache/blob/main/step-9-fp8/model.py)

### Running it

```bash
cd step-9-fp8/
python bench_fp8.py
```

### Results

### What the numbers mean
Walk through the output line by line.  
Be explicit about what was expected vs. what happened.

### Naive FP8 cautionary result
![Step 9 plot](plots/fp8.png)
*What to look at: separate theoretical FP8 promise from practical implementation pitfalls.*

### What we learned
One or two sentences. What does this step add to the overall picture? What question does it open for the next step?

</section>

<section id="step-10-tp">

## Step 10 — Tensor Parallelism (TP-2)

### What we're building
One sentence: add tensor-parallel inference path in `step-10-tp/model.py`.

### The idea
Concept explanation in plain English. No code yet.  
What problem does this solve? What does it change?

### The implementation
Key code change only - not the full file.

```python
# TODO: add annotated TP partition/collective snippet
```

[Full code: step-10-tp/model.py](https://github.com/venkatacrc/nanogpt-kv-cache/blob/main/step-10-tp/model.py)

### Running it

```bash
cd step-10-tp/
python bench_tp.py
```

### Results

### What the numbers mean
Walk through the output line by line.  
Be explicit about what was expected vs. what happened.

### TP scaling tradeoff
![Step 10 plot](plots/fig_tp.png)
*What to look at: compare prefill gains against decode communication overhead.*

### What we learned
One or two sentences. What does this step add to the overall picture? What question does it open for the next step?

</section>

<section id="step-11-pareto">

## Step 11 — Throughput/latency Pareto

### What we're building
One sentence: measure batch-size tradeoffs in `step-11-pareto/bench_pareto.py`.

### The idea
Concept explanation in plain English. No code yet.  
What problem does this solve? What does it change?

### The implementation
Key code change only - not the full file.

```python
# TODO: add annotated throughput-sweep snippet
```

[Full code: step-11-pareto/bench_pareto.py](https://github.com/venkatacrc/nanogpt-kv-cache/blob/main/step-11-pareto/bench_pareto.py)

### Running it

```bash
cd step-11-pareto/
python bench_pareto.py
```

### Results

### What the numbers mean
Walk through the output line by line.  
Be explicit about what was expected vs. what happened.

### Throughput vs per-user latency
![Step 11 plot](plots/fig_pareto.png)
*What to look at: locate operating points where throughput rises with acceptable per-user latency.*

### What we learned
One or two sentences. What does this step add to the overall picture? What question does it open for the next step?

</section>

</main>
</div>
