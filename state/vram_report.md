# VRAM report — real encoders on the actual GTX 1650

Measured by `python scripts/probe_vram.py --amp on`; raw data in
`state/vram_probe.json`. Every row is a genuine forward + backward + optimiser
step with a real pretrained checkpoint, not the tiny test encoder.

| | |
|---|---|
| GPU | NVIDIA GeForce GTX 1650 with Max-Q Design |
| Total VRAM | 4096 MiB (3938 MiB free at idle — ~158 MiB is driver overhead) |
| Compute capability | sm_75 (Turing TU117): **no tensor cores, no bf16** |
| Graphs | 32 nodes each (our `segmentation.max_nodes`), 96-dim features |
| "Safe" threshold | 80% of 4096 MiB = 3277 MiB reserved |

**Judged on `max_memory_reserved`, not `max_memory_allocated`.** Reserved is what
the caching allocator actually holds and what collides with the 4 GB wall;
allocated understates it by 100–450 MB here.

---

## Results

| Config | batch | seq | AMP | grad ckpt | alloc MB | **reserved MB** | fits 4 GB? |
|---|---|---|---|---|---|---|---|
| Task 1 distilbert | 8 | 128 | on | off | 1301 | 1378 | yes — local |
| Task 1 distilbert | 16 | 128 | on | off | 1311 | 1382 | yes — local |
| Task 1 distilbert | 32 | 128 | on | off | 1303 | 1574 | yes — local |
| Task 1 distilbert | 32 | 128 | on | **on** | 1300 | 1376 | yes — local |
| Task 1 bert-base | 8 | 128 | on | off | 2131 | 2336 | yes — local |
| Task 1 bert-base | 16 | 128 | on | off | 2141 | 2286 | yes — local |
| Task 1 bert-base | 16 | 128 | on | **on** | 2134 | 2294 | yes — local |
| Task 1 bert-base | 32 | 128 | on | off | 2219 | 2620 | yes — local |
| Task 1 bert-base | 32 | 128 | on | **on** | 2122 | 2274 | yes — local |
| Task 1 bert-base | **64** | 128 | on | off | 3805 | **4254** | **NO — Kaggle** |
| Task 1 bert-base | 64 | 128 | on | **on** | 2116 | 2234 | yes — local |
| Task 1 bert-base | 32 | **256** | on | off | 3805 | **4254** | **NO — Kaggle** |
| Task 1 bert-base | 32 | 256 | on | **on** | 2116 | 2234 | yes — local |
| Task 2 GNN | 8 | — | on | — | 25 | 50 | yes — trivial |
| Task 2 GNN | 32 | — | on | — | 33 | 56 | yes — trivial |
| Task 2 GNN | 128 | — | on | — | 78 | 130 | yes — trivial |
| Task 3 fusion distilbert | 8 | 128 | on | off | 1326 | 1418 | yes — local |
| Task 3 fusion distilbert | 8 | 128 | on | **on** | 1315 | 1398 | yes — local |
| Task 3 fusion distilbert | 16 | 128 | on | off | 1336 | 1406 | yes — local |
| Task 3 fusion bert-base | 8 | 128 | on | off | 2164 | 2350 | yes — local |
| Task 3 fusion bert-base | 8 | 128 | on | **on** | 2136 | 2298 | yes — local |
| Task 3 fusion bert-base | 16 | 128 | on | off | 2166 | 2310 | yes — local |
| Task 3 fusion bert-base | 32 | 128 | on | off | 2234 | 2642 | yes — local |
| Task 3 fusion bert-base | 32 | 128 | on | **on** | 2146 | 2298 | yes — local |
| Task 3 fusion bert-base | 64 | 128 | on | **on** | 2144 | 2268 | yes — local |
| Task 4 frozen + precomputed text | 256 | — | on | — | 393 | 484 | yes — local |
| Task 4 frozen + precomputed text | 512 | — | on | — | 510 | 692 | yes — local |
| Task 4 frozen + precomputed text | 1024 | — | on | — | 744 | 1072 | yes — local |
| Task 4 frozen + precomputed text | 2048 | — | on | — | 1213 | 1802 | yes — local |

---

## What this actually says

**1. The Phase-A expectation of "2.5–4 GB for real fusion" does not hold, and the
reason is structural.** Task 3 fusion with bert-base at batch 8 costs 2350 MB
reserved — only ~14 MB more than Task 1 bert-base alone. The GNN branch is
~430k parameters over 32-node graphs, and cross-attention is a single query token
over 128 keys. Both are rounding errors next to the text encoder. **Fusion is not
the expensive part; the text encoder is.**

**2. Peak memory is dominated by optimiser state, not activations.** bert-base is
109M parameters: fp32 weights 418 MB + gradients 418 MB + AdamW `m` and `v`
836 MB ≈ 1.7 GB before a single activation exists. That is why:

- AMP saves almost nothing at small batch (1301 vs 1293 MB at batch 8) but does
  help at batch 32 (1303 vs 1501 MB) — it shrinks activations, and activations
  only matter once they are the larger term.
- Gradient checkpointing is a *no-op* below batch 32 and a **45% cut** at batch
  64 (3805 → 2116 MB alloc), exactly where activations finally dominate.

**3. Gradient checkpointing is the lever that keeps bert-base local.** Both
configs that overflow — batch 64, and seq 256 at batch 32 — drop to 2234 MB
reserved with checkpointing on. The cost is roughly a 30% slowdown from
recomputing the forward pass.

**4. `reserved` at batch 64 without checkpointing reads 4254 MB on a 4096 MB
card.** That is not a measurement error: Windows WDDM lets an over-committed
allocation spill into shared system memory rather than hard-failing, so it
"works" while thrashing across PCIe. Treat it as failed. This is why the verdict
column is computed on reserved memory against an 80% threshold and not on
whether the probe threw `OutOfMemoryError`.

**5. Task 2 and Task 4 are nowhere near the limit.** The GNN uses 130 MB at batch
128, and Task 4 with precomputed text uses 1802 MB at batch **2048** — four times
the configured `contrastive.batch_size: 512`. If Task 4 retrieval numbers look
weak, batch size is free headroom to spend.

---

## Routing decision

| Run | Where | Settings |
|---|---|---|
| Task 1 distilbert (dev) | **local** | batch 32, seq 128, AMP |
| Task 1 bert-base (headline) | **local** | batch 32 seq 128 AMP; add grad-ckpt for batch 64 |
| Task 1 bert-base × 3 freeze modes × 3 seeds | **Kaggle** | not a memory limit — 9 runs × ~1 h is a *time* limit on one laptop GPU |
| Task 2 GNN, all sweeps | **local** | batch up to 128; memory is irrelevant here |
| Task 3 fusion distilbert | **local** | batch 8–16, AMP |
| Task 3 fusion bert-base | **local** | batch 32 AMP; grad-ckpt only above batch 32 |
| Task 4 contrastive | **local** | frozen text + `--precompute-text-embeddings`, batch 512 (2048 also fits) |
| B2 CNN baseline | **local** | trivially small |

The constraint that actually pushes work to Kaggle is **wall-clock, not VRAM**.
Everything in this project fits on the 1650; what does not fit is running three
freeze modes across three seeds sequentially on one laptop.

## Caveats

- Measured with a **single step** on synthetic tensors. A real run adds dataloader
  workers with `pin_memory` (host RAM, not VRAM) and gradient accumulation, which
  does not raise the peak — accumulation trades steps for memory, it does not add
  any.
- Graphs were fixed at 32 nodes, our configured maximum, so these are worst-case
  for the GNN branch.
- Windows keeps a compositor surface on the same GPU. Close other GPU consumers
  before a long run; the 158 MiB idle overhead is already excluded from the
  3938 MiB free figure but a browser can take several hundred MB more.
