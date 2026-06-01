# Part 2.3 — Why MoE? A Cost-Benefit Analysis

## Parameter counts

DeepSeek-V3 has **~671B total parameters** but only **~37B activated per token**
(8 routed experts + 1 shared expert × 3-layer MLP, plus MLA attention across 58 MoE layers).
Llama-3 8B activates all 8B params on every token. For the same inference
compute budget (~37B activated FLOPs), a dense model would have 37B total params;
DeepSeek-V3 uses ~18× more capacity at the same per-token cost.

## Training FLOPs and memory

Because the training FLOP accounting (`6·N_activated·D`) uses **activated** params
rather than total params, training DeepSeek-V3 to D tokens costs roughly the same
FLOPs as training a ~37B dense model to D tokens — yet the model benefits from a
671B-parameter routing vocabulary. Peak training memory per GPU scales with
*activated* weights (held in optimizer state), not total weight count — so
activation memory matches a ~37B dense model. However, all 671B params must be
*sharded* across devices (stored but not all in optimizer state simultaneously),
so total storage is 671B × 2 bytes ≈ ~1.3 TB in bf16.

## Communication costs

In Expert Parallel (EP) routing, every MoE forward pass requires two all-to-all
collectives proportional to `batch × hidden × num_active_experts`. With
`n_routed_experts=256` across 8 GPUs (32 experts/rank), the all-to-all volume per
MoE layer per token is `8 × 7168 × 2 bytes ≈ 112 KB`. At 58 MoE layers this is
~6.5 MB per token batch, compared to zero extra communication in a dense model.
Communication cost scales as O(`num_experts_per_tok × h`), not O(`n_routed_experts`).

## Inference economics

**Low load:** MoE serves requests cheaply — only activated weights need to be in
hot GPU memory for the routed experts, matching dense-37B cost.
**High load:** Token-bucket imbalance (hot experts receiving many tokens) causes
uneven GPU utilization and stalls. Dense models have no such bottleneck.

## One concrete advantage

MoE allows **scaling model capacity ~18× beyond what fits in optimizer memory**
at the same per-token training FLOP budget. The scaling-law loss improvement
from having 671B total params (higher model capacity and memorization) exceeds
what a same-compute dense model (~37B) achieves, because larger N reduces the
first term `406.4/N^0.34` in the Chinchilla-style loss.

## One concrete disadvantage

MoE introduces **all-to-all communication on every layer forward pass**, which
requires high-bandwidth interconnects (NVLink / InfiniBand). On commodity GPU
clusters without such interconnects, the communication overhead can eliminate the
efficiency advantage entirely, making dense models more practical at the same budget.
