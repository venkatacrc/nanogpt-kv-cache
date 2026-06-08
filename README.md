# nanogpt-kv-cache

A step-by-step tutorial that adds a **KV cache** to Andrej Karpathy's nanoGPT,
then empirically dissects when the cache helps, when it hurts, and which rescues
actually work — all on a single H100, with code small enough to read in an
afternoon.

Each numbered folder is a self-contained step. Read the code, run the script,
read the log, move on.

---

## TL;DR — what we measured (gpt2-xl unless noted, FP16 unless noted, H100)

The empirical findings, in one table:

| #  | Question                                                  | Headline result                                            |
|----|-----------------------------------------------------------|------------------------------------------------------------|
| 1  | Does a KV cache speed up generation?                      | Yes in FP32: ~Nx speedup vs full recompute                 |
| 2  | Does the FP16 cache also speed up generation?             | **Not at B=1, P=1**: 0.91x — the cache *loses*             |
| 3  | Why does the FP16 cache lose?                             | Cached decode hits a **~10 ms TPOT launch floor**          |
| 4  | Can we rescue the FP16 cache?                             | Yes, via 3 levers (see "Rescues" below) — stacks to **4.3x** |
| 5  | Is SDPA / FlashAttention always faster than manual attn?  | Yes, always 1.3–2.2x. Bigger win at long prefill.          |
| 5  | Is FlashAttention-3 always faster than SDPA?              | **No**: at head_dim=64 and short contexts, FA3 *loses* to SDPA up to 24% |
| 6  | Does FA3 win at Hopper-friendly shapes (head_dim=128)?    | Prefill yes (1.11x at S=2K, **1.37x at S=16K**). Decode no. |
| 6b | Can a `[B, S, H, D]` cache layout rescue FA3 decode?      | Closes ~9 pp of the gap. Last ~5 pp is intrinsic kernel quality. |
| 6b | Does batching close the remaining FA3 decode gap?         | **No**: FA3/SDPA ratio is unmoved by B=1, 4, 8.            |
| 7  | How does cache speedup scale with model size and N?        | Speedup grows with generated length N; model-size trend is flatter than expected. |
| 8  | Does GQA make FA3 decode win?                              | Decode still near-tied with SDPA; biggest GQA win is KV-memory reduction. |
| 9  | Is naive FP8 attention a free speedup on H100?             | **No**: slower at tested lengths and numerically unstable without proper scaling. |
| 10 | Does TP-2 help on a 6B-class model that fits on 1 GPU?     | Prefill improves (up to ~1.47x), decode TPOT regresses (~20%) from NCCL overhead. |
| 11 | What does throughput vs latency look like at scale?         | TPOT stays ~flat from B=2..256 while total throughput scales near-linearly. |

The big surprises:

- **FP16 KV cache can be slower than recomputing** at small batch size and short prompts. Mechanism: the cached path is launch-floor-dominated; the uncached path is compute-dominated; when there's little compute (short P, small B), recompute is cheaper.
- **FlashAttention-3 is not a universal upgrade**. At single-stream decode of a model with `[B, H, S, D]`-layout KV cache, FA3 loses to PyTorch's SDPA — even at long context, even at head_dim=128, even with batching, even after refactoring the cache to FA3's native layout. The "FA3 is faster" headline is a prefill-and-large-batch story; decode is its own world.
- **torch.compile rescues the cache** at the worst-case setting (B=1, P=1) by lowering the launch floor itself, turning a 0.91x loss into a 1.68x win.
- **Rescues stack at 81% efficiency** — compile + batching together = 4.3x cache speedup, not the 5.4x you'd predict from multiplying single-rescue measurements. They share part of the same bottleneck.

---

## Step-by-step map

Each folder is a sealed unit: model.py + inference.py + a small script + a canonical `.log`. Skim the script, run it, compare to the log. Move on.

```
step-0-baseline/             gpt2 forward pass + naive greedy generation (full recompute every step).
                             scripts: bench.py, generate.py
                             log:     bench.log

step-1-kvcache/              Add KVCache class + cache-aware attention + cached generate.
                             Bit-exact vs uncached in FP32. Cache wins on gpt2-xl by ~N×.
                             scripts: bench.py, test_correctness.py
                             logs:    bench_gpt2.log, bench_gpt2xl.log, test_correctness.log

step-2-fp16/                 Half() the model + FP16 cache. Surprise: cache LOSES at gpt2-xl B=1 P=1 (0.91x).
                             scripts: bench.py, test_correctness.py
                             logs:    bench.log, test_correctness.log

step-3-decomp/               Decompose per-step where the time goes. Reveals ~10 ms TPOT launch floor.
                             script:  decomp.py
                             log:     decomp.log

step-4-rescues/              Three independent rescues that turn the cache from loss to win.
                             scripts: long_prompt.py     (P=1 -> 512:  0.91x -> 1.67x)
                                      batch.py           (B=1 -> 8:    0.91x -> 2.91x)
                                      compile.py         (eager -> torch.compile: 0.91x -> 1.68x)
                                      stack.py           (compile + batch B=8:   4.31x, 81% of predicted)

step-5-backends/             Make attention backend selectable: manual / SDPA / FA3.
                             SDPA wins always. FA3 LOSES on gpt2-xl (head_dim=64) by 5-24%.
                             scripts: test_backends.py, bench_backends.py

step-6-shapes/               Rebuild model at Llama-ish shapes (head_dim=128, block_size up to 16K).
                             FA3 prefill flips and beats SDPA past S=2K, growing to 1.37x at S=16K.
                             Decode unchanged: FA3 still loses by ~12%.
                             script:  bench_shapes.py

step-6b-layout/              Refactor KV cache to FA3-native [B, S, H, D] layout.
                             FA3 decode gap closes from 0.87x to 0.96x — most of the penalty
                             was layout, not the kernel. The remaining ~5% is intrinsic.
                             scripts: test_layout.py, bench_layout.py, bench_batch_decode.py

step-7-extras/               Extra sweeps to stress-test the baseline narrative.
                             (a) model_size.py: cache speedup vs model size
                             (b) long_generation.py: cache speedup vs generated length N
                             logs: model_size.log, long_generation.log

step-8-gqa/                  Add grouped-query attention (GQA) support end-to-end.
                             Compare SDPA vs FA3 for prefill/decode under n_kv_head sweeps.
                             scripts: test_gqa.py, bench_gqa.py
                             logs: test_gqa.log, bench_gqa.log

step-9-fp8/                  Experimental FA3 FP8 attention path (E4M3) with naive scaling.
                             Useful negative result: shows why production FP8 needs proper
                             calibration / block-wise scaling, not just dtype casts.
                             script:  bench_fp8.py
                             log:     bench_fp8.log

step-10-tp/                  Tensor Parallelism (TP) on a Llama-3-like 6B-class model.
                             Compare 1-GPU vs TP-2 for prefill and decode (SDPA/FA3).
                             scripts: bench_tp.py, run_tp2.py
                             logs:    bench_tp.log, run_tp2.log

step-11-pareto/              Throughput-vs-latency sweep over batch size B (FA3 path).
                             Adds production-facing metrics view: TPOT, total tok/s, and TTFT proxy.
                             script:  bench_pareto.py
                             logs:    bench_pareto.log, bench_pareto_extended.log
```

---

## Setup

```bash
git clone https://github.com/venkatacrc/nanogpt-kv-cache.git
cd nanogpt-kv-cache
pip install torch transformers tiktoken
# Optional, for step-5/6/6b FA3 experiments (Hopper / SM 9.0+ required):
pip install flash-attn-3
```

Hardware:
- All measurements in this repo are from a single **NVIDIA H100 80GB**.
- Step-0 through step-4 run on any modern NVIDIA GPU. Step-5 SDPA path works
  on Ampere+ (A100, H100). Step-5/6/6b/8/9/11 FA3 paths require Hopper (H100, H200).
- Step-10 TP experiments require **2 GPUs** for TP-2 runs (`run_tp2.py`).

Software:
- Python 3.10+
- PyTorch 2.5+ (tested with 2.10)
- transformers, tiktoken
- flash_attn_3 (Hopper-only, optional for FA3 paths)

---

## How to run a step

Every step is self-contained. Example:

```bash
cd step-4-rescues/
python3 compile.py     # ~5-8 min on H100; saves nothing, prints a table
```

Each script is independent. There's no shared state, no setup beyond `pip install`.
Many of the long-running scripts use `median of N trials` and a beefed-up warmup
to avoid first-call JIT pollution — see the docstring at the top of each script.

To compare against my numbers, look at the matching `*.log` file in the same
folder. Hardware variation is small but real; FP16 reduction order is also
non-deterministic in SDPA, so don't expect bit-identical numbers.

---

## What this repo deliberately leaves out

This is a tutorial, not a production inference framework. We now include
intro-level GQA, FP8 experiments, TP-2, and throughput/latency sweeps, but
if you're building something real you'll still need:

- **Paged attention** (vLLM-style page tables for the KV cache)
- **Speculative / multi-token decoding** (turns decode into mini-prefill)
- **Continuous batching runtime** (request queueing + scheduler; we only benchmark static B)
- **Production quantization stack** (INT8/AWQ/GPTQ, or production-grade FP8 with calibration)
- **RoPE positional embeddings** (Llama-style; nanoGPT uses learned WPE)

Each of these is a 10-100x lever in real serving systems, on top of everything
measured here.

---

## Credits

- **Andrej Karpathy** — [nanoGPT](https://github.com/karpathy/nanoGPT). The
  architecture, the weight-loading-from-HF pattern, and the tasteful 200-line
  forward pass are all his.
- **Tri Dao et al.** — [FlashAttention](https://github.com/Dao-AILab/flash-attention)
  papers and the prebuilt FA3 wheel on H100 used in step-5/6.
- **PyTorch SDPA team** — the dispatcher that "just works" and picks Flash /
  cuDNN / memory-efficient under the hood without us asking.

---

## License

MIT. See `LICENSE`.
