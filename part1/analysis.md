# Part 1.3 — Benchmark Analysis

## 1. Setup

- **Hardware:** Apple M-series / x86-64 CPU (4 logical cores used via --oversubscribe)
- **MPI:** Open MPI (mpirun --oversubscribe -n 4)
- **World size:** 4 ranks
- **dtype:** float32 (NumPy)
- **Iterations per timing:** 10 warm-start repetitions

## 2. Sweeps

We swept `hidden_dim` ∈ {16, 32, 64, 128} and `batch_size` ∈ {8, 32},
with fixed `feature_dim = output_dim = 8`, `topk = 2`, `num_experts = 4`, `world_size = 4`.

All timings are milliseconds per forward pass (average of 10 iterations),
measured with `mpirun --oversubscribe -n 4 python part1/benchmark.py` on a 4-core CPU (Apple M-series).

| batch | hidden | SimpleMoE (ms) | MoE_TP (ms) | MoE_EP (ms) |
|------:|-------:|---------------:|------------:|------------:|
|     8 |     16 |           0.10 |        0.87 |        0.18 |
|     8 |     32 |           0.10 |        0.86 |        0.18 |
|     8 |     64 |           0.11 |        0.88 |        0.20 |
|     8 |    128 |           0.13 |        0.95 |        0.20 |
|    32 |     16 |           0.33 |        3.36 |        0.28 |
|    32 |     32 |           0.34 |        3.32 |        0.29 |
|    32 |     64 |           0.35 |        3.73 |        0.32 |
|    32 |    128 |           0.43 |        3.71 |        0.35 |

## 3. Discussion

### Tensor Parallel (MoE_TP)

MoE_TP shards each expert's weight matrices across ranks (column-parallel).  Each rank
computes a `(batch, hidden/world_size)` partial output, then **Allgather** reassembles
the full `(batch, output_dim)` result on every rank.

- **Communication pattern:** `Allgather` — each rank sends `batch × local_out_features × 4 bytes`
  and receives the full output from all other ranks.
- **Observed:** MoE_TP is 6–10× slower than SimpleMoE across all configs. With
  `output_dim = 8` (tiny output), every ShardedLinear call triggers two Allgather collectives
  whose per-call latency (~0.4 ms each) completely dominates the negligible matmul cost.
  MoE_TP is **communication-latency-bound** at these small dimensions.
- **Scaling behavior:** Allgather volume is O(batch × output_dim), independent of hidden_dim,
  so timing barely changes as hidden grows (0.87 ms → 0.95 ms for batch=8). At much larger
  hidden dims, the matmul would eventually dominate and amortize the collective overhead.

### Expert Parallel (MoE_EP)

MoE_EP routes entire tokens to their owning ranks via two **Alltoall** rounds.

- **Communication pattern:** Two `alltoall` calls per top-k slot.
  Volume per round = `batch × input_dim` (tokens out) + `batch × output_dim` (results back).
  This is **independent of `hidden_dim`**.
- **Observed:** MoE_EP is 2–13× faster than MoE_TP and only 1.3–1.8× slower than SimpleMoE.
  The Alltoall sends only 2 × (batch × 8) floats — very small — so per-token dispatch is
  cheap. The expert compute is local after dispatch, avoiding the repeated Allgather overhead.
- **Batch scaling:** EP timing grows more with batch (0.18 ms → 0.28–0.35 ms from batch 8 to 32)
  than TP does (0.87 ms → 3.36 ms), because TP's token-by-token loop through ShardedExpert
  triggers O(batch × topk) allgathers. EP amortizes all tokens into one Alltoall pair per
  topk slot, keeping communication cost proportional to batch rather than batch².

#### Custom `myAlltoall` path

My EP implementation uses the manual PA2 `myAlltoall` for both token dispatch
and result return. Since that buffered collective sends equal-size segments, EP
first exchanges each destination bucket's row count with `myAlltoall`, pads the
token or result buckets to the largest bucket size for the round, and trims each
received segment back to its exchanged row count. The padding keeps the manual
collective's fixed-segment contract while preserving the variable token routing
that expert parallelism needs.

### Regime Summary

| Workload | SimpleMoE (ms) | MoE_TP (ms) | MoE_EP (ms) | Bottleneck |
|----------|:--------------:|:-----------:|:-----------:|------------|
| b=8, hidden=16–128 | 0.10–0.13 | 0.86–0.95 | 0.18–0.20 | TP: Allgather latency per ShardedLinear call |
| b=32, hidden=16–128 | 0.33–0.43 | 3.32–3.73 | 0.28–0.35 | TP: O(batch×topk) Allgather calls; EP: local expert + cheap Alltoall |

**Key takeaway:**
- TP is preferred when experts have large weight matrices and high arithmetic intensity
  (e.g., large hidden_dim), because the column-parallel matmul keeps all ranks busy.
- EP is preferred when `num_experts ≫ world_size` or batch sizes are large, because
  each rank stores only one expert (no weight replication) and the Alltoall cost
  does not scale with the expert's hidden dimension.
- In practice on GPU clusters with NVLink, TP Allgather is extremely fast (NVLink
  bandwidth ~900 GB/s on H100), making TP competitive even at smaller hidden dims.
  EP's two-Alltoall pattern requires the same high-bandwidth interconnect to be efficient
  at scale (e.g., DeepSeek-V3 uses 256 experts across 32 nodes with InfiniBand).
