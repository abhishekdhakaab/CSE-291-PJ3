# Part 1.3 — Benchmark Analysis

## Setup

I ran the benchmark on my MacBook (Apple M-series, 4 cores) using `mpirun --oversubscribe -n 4`. Everything is NumPy float32. Each timing is the average of 10 iterations after one warm-up run so the numbers are reasonably stable.

## Results

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

## What I observed

### Tensor Parallel is surprisingly slow here

MoE_TP came out 6–10x slower than SimpleMoE across every config, which was more than I expected. The reason makes sense once you think about it: ShardedLinear calls Allgather twice per expert forward pass (once after fc1, once after fc2), and on a CPU with tiny tensors, each of those collectives costs around 0.4 ms just in latency. The actual matrix multiply at these sizes is basically free, so the communication just dominates completely.

What's interesting is that hidden_dim barely matters for TP timing at all — going from hidden=16 to hidden=128 only adds about 0.08 ms at batch=8. That's because Allgather volume depends on `batch × output_dim`, not on hidden_dim. The bottleneck is the fixed per-call overhead of the collective, not the data volume. If I had much larger hidden dims, the matmul would eventually start to cost something and the communication overhead would get amortized — but at these sizes we're nowhere near that crossover.

Batch size hurts TP a lot. Going from batch=8 to batch=32, TP jumps from ~0.87 ms to ~3.36 ms (roughly 4x slower). This is because my TP forward loop runs ShardedExpert token-by-token, so it's doing O(batch × topk) Allgather calls. That's just a lot of small collectives piling up.

### Expert Parallel is much more reasonable

EP stayed within 1.3–1.8x of SimpleMoE, which is a very different story. The Alltoall sends only `batch × feature_dim` floats out and `batch × output_dim` floats back — at these tiny dimensions that's a small amount of data, so the dispatch is cheap. The expert compute itself runs locally after the first Alltoall, so there's no per-expert communication overhead.

EP also scales with batch much more gracefully than TP. From batch=8 to batch=32, EP only goes from ~0.18 ms to ~0.28 ms. That makes sense — EP does one Alltoall pair per topk slot regardless of how many tokens there are, whereas TP is doing repeated collectives for every single token.

The one place EP doesn't shine is when hidden_dim is very large — because then the Alltoall for sending tokens becomes relatively cheaper compared to the expert compute, but my ShardedLinear in TP would also benefit more from the large matmul. At small dimensions EP wins pretty clearly; at large dimensions on fast interconnects (like NVLink on H100s), TP would likely close the gap.

### myAlltoall

My EP uses the manual `myAlltoall` from PA2 for both the token dispatch and the result return. Since myAlltoall needs equal-size segments, I first exchange per-rank row counts (also via myAlltoall), pad all the token buckets to the maximum bucket size, run the collective, then trim using the received counts. It's a bit annoying to set up but it works correctly and lets me reuse the PA2 implementation directly.

### Which one to use and when

For these small workloads EP is clearly better — cheaper communication, simpler scaling with batch size. TP only makes sense when the expert weights are large enough that the column-parallel matmul actually keeps all ranks busy and amortizes the Allgather cost. On a real GPU cluster with NVLink (900 GB/s on H100), TP Allgather is extremely fast and the crossover happens much earlier. But on CPU with these tiny dimensions, EP wins every time.

For something like DeepSeek-V3 with 256 experts across 32 nodes, EP is really the only option that scales — you can't replicate 256 experts on every rank. The Alltoall cost becomes the bottleneck there too, which is why they need InfiniBand to make it work.
