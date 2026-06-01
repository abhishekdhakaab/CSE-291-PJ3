# Part 1.3 — Benchmark Analysis

## 1. Setup

Ran on my MacBook (Apple M-series, 4 cores) using `mpirun --oversubscribe -n 4`. NumPy float32 throughout. Each timing is the average of 10 iterations after one warm-up run.

## 2. Sweeps

I swept `hidden_dim` across {16, 32, 64, 128} and `batch_size` across {8, 32}, keeping `feature_dim = output_dim = 8`, `topk = 2`, `num_experts = 4`.

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

TP was way slower than I expected — 6 to 10x slower than SimpleMoE across every config. The reason is that `ShardedLinear` calls `allgather` twice per expert forward pass (once for fc1, once for fc2), and on a CPU with tiny tensors, each of those collectives costs around 0.4 ms just in latency. The actual matmul is negligible at these sizes so the communication just dominates completely.

What's interesting is that hidden_dim barely affects TP timing at all — going from hidden=16 to hidden=128 only adds about 0.08 ms at batch=8. That makes sense because the allgather volume depends on `batch × output_dim`, not on hidden_dim. TP would look much better with larger output dimensions where the compute starts to matter.

### Expert Parallel (MoE_EP)

EP was much more reasonable — only 1.3 to 1.8x slower than SimpleMoE, and 2 to 13x faster than TP depending on config. The alltoall only moves `batch × feature_dim` tokens out and `batch × output_dim` results back, which at these sizes is tiny. The expert compute itself runs locally after dispatch so there's no per-expert communication overhead.

Batch size actually hurts EP less than TP. Going from batch=8 to batch=32, TP jumps from ~0.87 ms to ~3.36 ms (roughly 4x), while EP only goes from ~0.18 ms to ~0.28 ms. TP is doing O(batch × topk) allgather calls because it loops token-by-token, whereas EP does one alltoall for the whole batch per topk slot.

#### myAlltoall

My EP uses the PA2 `myAlltoall` instead of the built-in collective. Since `myAlltoall` needs equal-size segments, I first exchange per-rank row counts, pad all buckets to the maximum, run the collective, then trim using the received counts. A bit annoying to set up but it works correctly.

### Summary

TP makes sense when hidden dimensions are large enough that the matmul cost actually amortizes the allgather overhead — on real GPU clusters with NVLink that crossover happens much earlier. EP is better here because the alltoall cost doesn't grow with hidden_dim, so it stays cheap even as the experts get wider. For something like DeepSeek-V3 with 256 experts across many nodes, EP is the only option that scales, though you need fast interconnects (InfiniBand) for the alltoall to not become the bottleneck.
