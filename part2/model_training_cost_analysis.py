"""Model training cost analysis for Part 2."""
import argparse
import json
import math


def model_training_cost_analysis_llama(model_config_path):
    """Returns (total_params, flops_layer_TF, peak_memory_GB) for a Llama-3 model.

    Convention: batch=1, seq_len from config, bf16, rematerialization on.
    """
    with open(model_config_path) as f:
        cfg = json.load(f)

    h    = cfg["hidden_size"]
    i    = cfg["intermediate_size"]
    L    = cfg["num_hidden_layers"]
    n_q  = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    V    = cfg["vocab_size"]
    S    = cfg["max_position_embeddings"]
    d    = h // n_q  # head dim
    tie  = cfg.get("tie_word_embeddings", False)

    # ── Parameter count ──────────────────────────────────────────────────────
    embed   = V * h
    lm_head = 0 if tie else V * h
    norm    = h  # final RMSNorm

    # Per-layer: QKV projections, output projection, SwiGLU MLP, 2x RMSNorm
    layer = (h * n_q * d        # Q
           + h * n_kv * d       # K
           + h * n_kv * d       # V
           + n_q * d * h        # O
           + h * i + h * i + i * h  # gate, up, down
           + h + h)             # input_norm, post_attn_norm

    total_params = embed + lm_head + norm + L * layer

    # ── Forward FLOPs for one layer (batch=1, seq=S) ─────────────────────────
    # Attention projections
    f  = 2 * S * h * (n_q * d)   # Q
    f += 2 * S * h * (n_kv * d)  # K
    f += 2 * S * h * (n_kv * d)  # V
    f += 2 * S * (n_q * d) * h   # O
    # Attention score and context
    f += 2 * n_q * S * S * d     # QK^T
    f += 2 * n_q * S * S * d     # Attn @ V
    # SwiGLU MLP
    f += 2 * S * h * i           # gate
    f += 2 * S * h * i           # up
    f += 2 * S * i * h           # down
    flops_layer_TF = f / 1e12

    # ── Peak memory for one layer (bf16, rematerialization) ──────────────────
    BPE = 2  # bytes per bf16 element
    peak = (layer * BPE             # layer weights
          + S * h * BPE             # input activation
          + S * n_q * d * BPE       # Q
          + 2 * S * n_kv * d * BPE  # K + V
          + n_q * S * S * BPE)      # attention score matrix (peak)
    peak_memory_GB = peak / (1024 ** 3)

    return total_params, flops_layer_TF, peak_memory_GB


def get_optimal_N_D_from_cost(cost_budget):
    """Returns (N, D, training_budget_flops, best_gpu) for a dollar training budget.

    Uses Chinchilla-style scaling law: L = 406.4/N^0.34 + 410.7/D^0.29 + 1.69
    MFU = 40%.
    """
    gpus = {
        "H100": {"price_per_hour": 3.0,  "peak_tflops": 989.0},
        "H200": {"price_per_hour": 4.0,  "peak_tflops": 989.0},
        "B200": {"price_per_hour": 6.0,  "peak_tflops": 2250.0},
    }
    MFU = 0.40

    # Pick GPU that gives the most FLOPs per dollar.
    best_gpu, best_flops = None, 0.0
    for name, spec in gpus.items():
        flops = (cost_budget / spec["price_per_hour"]) * 3600 * spec["peak_tflops"] * 1e12 * MFU
        if flops > best_flops:
            best_flops, best_gpu = flops, name

    F = best_flops
    # Lagrangian optimum of L under 6*N*D = F:
    #   N = [(a*alpha) / (b*beta) * (F/6)^beta] ^ (1/(alpha+beta))
    a, alpha = 406.4, 0.34
    b, beta  = 410.7, 0.29
    N = int(round(((a * alpha) / (b * beta) * (F / 6) ** beta) ** (1 / (alpha + beta))))
    D = int(round(F / (6 * N)))

    return N, D, F, best_gpu


def model_training_cost_analysis_deepseek(model_config_path):
    """Returns (total_params, flops_layer_TF, peak_memory_GB) for DeepSeek-V3."""
    with open(model_config_path) as f:
        cfg = json.load(f)

    h           = cfg["hidden_size"]
    L           = cfg["num_hidden_layers"]
    n_q         = cfg["num_attention_heads"]
    V           = cfg["vocab_size"]
    S           = cfg["max_position_embeddings"]
    tie         = cfg.get("tie_word_embeddings", False)
    q_lora      = cfg["q_lora_rank"]
    kv_lora     = cfg["kv_lora_rank"]
    qk_nope     = cfg["qk_nope_head_dim"]
    qk_rope     = cfg["qk_rope_head_dim"]
    v_head      = cfg["v_head_dim"]
    k_dense     = cfg["first_k_dense_replace"]
    dense_i     = cfg["intermediate_size"]
    n_routed    = cfg["n_routed_experts"]
    n_shared    = cfg["n_shared_experts"]
    n_active    = cfg["num_experts_per_tok"]
    moe_i       = cfg["moe_intermediate_size"]
    n_kv        = n_q  # MLA uses full head count for kv

    # ── MLA attention params ─────────────────────────────────────────────────
    attn = (h * q_lora + q_lora * n_q * (qk_nope + qk_rope)          # Q down+up
          + h * (kv_lora + qk_rope) + kv_lora * n_kv * (qk_nope + v_head)  # KV down+up
          + n_q * v_head * h                                            # O proj
          + h + h)                                                      # 2x RMSNorm

    # ── MLP params ───────────────────────────────────────────────────────────
    dense_mlp  = 3 * h * dense_i
    moe_routed = n_routed * 3 * h * moe_i
    moe_shared = n_shared * 3 * h * moe_i
    moe_mlp    = moe_routed + moe_shared

    # ── Total params ─────────────────────────────────────────────────────────
    total_params = (V * h + (0 if tie else V * h) + h
                  + k_dense * (attn + dense_mlp)
                  + (L - k_dense) * (attn + moe_mlp))

    activated = (k_dense * (attn + dense_mlp)
               + (L - k_dense) * (attn + n_active * 3 * h * moe_i + moe_shared))

    print(f"[DeepSeek-V3] Total params:     {total_params:,}  (~{total_params/1e9:.1f}B)")
    print(f"[DeepSeek-V3] Activated/token:  {activated:,}  (~{activated/1e9:.1f}B)")

    # ── FLOPs for one MoE layer (batch=1, seq=S) ─────────────────────────────
    d_eff = qk_nope + qk_rope
    f  = (2 * S * h * q_lora + 2 * S * q_lora * n_q * (qk_nope + qk_rope))  # Q
    f += (2 * S * h * (kv_lora + qk_rope) + 2 * S * kv_lora * n_kv * (qk_nope + v_head))  # KV
    f += 2 * S * n_q * v_head * h                                              # O
    f += 2 * n_q * S * S * d_eff                                               # QK^T
    f += 2 * n_q * S * S * v_head                                              # Attn@V
    f += 2 * S * h * moe_i * (n_active + n_shared) * 3                        # MoE MLP
    flops_layer_TF = f / 1e12

    # ── Peak memory for one MoE layer (bf16) ─────────────────────────────────
    BPE = 2
    moe_layer_params = attn + moe_mlp
    peak = (moe_layer_params * BPE
          + S * h * BPE
          + S * n_q * (qk_nope + qk_rope) * BPE
          + 2 * S * kv_lora * BPE
          + n_q * S * S * BPE)
    peak_memory_GB = peak / (1024 ** 3)

    return total_params, flops_layer_TF, peak_memory_GB


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--training_budget", type=float, default=None)
    args = parser.parse_args()

    if args.model_config:
        if "deepseek" in args.model_config.lower():
            params, flops, mem = model_training_cost_analysis_deepseek(args.model_config)
        else:
            params, flops, mem = model_training_cost_analysis_llama(args.model_config)
        print(f"Number of parameters: {params}")
        print(f"Number of TFLOPs: {flops}")
        print(f"Peak memory cost: {mem} GBs")

    if args.training_budget:
        N, D, F, gpu = get_optimal_N_D_from_cost(args.training_budget)
        print(f"best_gpu: {gpu}")
        print(f"training_budget_flops: {F:.2e}")
        print(f"Optimal N: {N}")
        print(f"Optimal D: {D}")
