"""Model training cost analysis for Part 2."""
import argparse
import json
import math


def model_training_cost_analysis_llama(model_config_path):
    """Analyze training cost of a dense Llama-style model.

    Returns:
        total_params:   total trainable parameter count (int)
        flops_layer_TF: forward FLOPs of a single transformer layer (TFLOPs)
        peak_memory_GB: peak forward memory of a single transformer layer (GB)

    See the Part 2.1 writeup for the sequence-length / batch convention.
    """
    with open(model_config_path) as f:
        cfg = json.load(f)

    hidden_size = cfg["hidden_size"]
    intermediate_size = cfg["intermediate_size"]
    num_layers = cfg["num_hidden_layers"]
    num_heads = cfg["num_attention_heads"]
    num_kv_heads = cfg["num_key_value_heads"]
    vocab_size = cfg["vocab_size"]
    seq_len = cfg["max_position_embeddings"]
    head_dim = hidden_size // num_heads
    tie_embeddings = cfg.get("tie_word_embeddings", False)

    embedding_params = vocab_size * hidden_size
    lm_head_params = 0 if tie_embeddings else vocab_size * hidden_size
    final_norm_params = hidden_size

    attention_params = (
        hidden_size * num_heads * head_dim
        + hidden_size * num_kv_heads * head_dim
        + hidden_size * num_kv_heads * head_dim
        + num_heads * head_dim * hidden_size
    )

    mlp_params = 3 * hidden_size * intermediate_size
    norm_params = 2 * hidden_size
    layer_params = attention_params + mlp_params + norm_params

    total_params = embedding_params + lm_head_params + final_norm_params + num_layers * layer_params

    flops = 0
    flops += 2 * seq_len * hidden_size * (num_heads * head_dim)
    flops += 2 * seq_len * hidden_size * (num_kv_heads * head_dim)
    flops += 2 * seq_len * hidden_size * (num_kv_heads * head_dim)
    flops += 2 * seq_len * (num_heads * head_dim) * hidden_size
    flops += 2 * num_heads * seq_len * seq_len * head_dim
    flops += 2 * num_heads * seq_len * seq_len * head_dim
    flops += 2 * seq_len * hidden_size * intermediate_size
    flops += 2 * seq_len * hidden_size * intermediate_size
    flops += 2 * seq_len * intermediate_size * hidden_size

    flops_layer_TF = flops / 1e12

    bytes_per_element = 2
    peak_memory = (
        layer_params * bytes_per_element
        + seq_len * hidden_size * bytes_per_element
        + seq_len * num_heads * head_dim * bytes_per_element
        + 2 * seq_len * num_kv_heads * head_dim * bytes_per_element
        + num_heads * seq_len * seq_len * bytes_per_element
    )
    peak_memory_GB = peak_memory / (1024 ** 3)

    return total_params, flops_layer_TF, peak_memory_GB


def model_training_cost_analysis_deepseek(model_config_path):
    """Analyze training cost of a DeepSeek-V3-style MoE model.

    Same return signature as the Llama version. See the Part 2.3 writeup
    for the MLA attention and the dense-vs-MoE layer breakdown.
    """
    with open(model_config_path) as f:
        cfg = json.load(f)

    hidden_size = cfg["hidden_size"]
    num_layers = cfg["num_hidden_layers"]
    num_heads = cfg["num_attention_heads"]
    vocab_size = cfg["vocab_size"]
    seq_len = cfg["max_position_embeddings"]
    tie_embeddings = cfg.get("tie_word_embeddings", False)

    q_lora_rank = cfg["q_lora_rank"]
    kv_lora_rank = cfg["kv_lora_rank"]
    qk_nope_dim = cfg["qk_nope_head_dim"]
    qk_rope_dim = cfg["qk_rope_head_dim"]
    v_head_dim = cfg["v_head_dim"]

    num_dense_layers = cfg["first_k_dense_replace"]
    dense_intermediate_size = cfg["intermediate_size"]
    num_routed_experts = cfg["n_routed_experts"]
    num_shared_experts = cfg["n_shared_experts"]
    num_active_experts = cfg["num_experts_per_tok"]
    moe_intermediate_size = cfg["moe_intermediate_size"]

    attention_params = (
        hidden_size * q_lora_rank
        + q_lora_rank * num_heads * (qk_nope_dim + qk_rope_dim)
        + hidden_size * (kv_lora_rank + qk_rope_dim)
        + kv_lora_rank * num_heads * (qk_nope_dim + v_head_dim)
        + num_heads * v_head_dim * hidden_size
        + 2 * hidden_size
    )

    dense_mlp_params = 3 * hidden_size * dense_intermediate_size
    routed_expert_params = num_routed_experts * 3 * hidden_size * moe_intermediate_size
    shared_expert_params = num_shared_experts * 3 * hidden_size * moe_intermediate_size
    moe_mlp_params = routed_expert_params + shared_expert_params

    embedding_params = vocab_size * hidden_size
    lm_head_params = 0 if tie_embeddings else vocab_size * hidden_size
    final_norm_params = hidden_size

    total_params = (
        embedding_params
        + lm_head_params
        + final_norm_params
        + num_dense_layers * (attention_params + dense_mlp_params)
        + (num_layers - num_dense_layers) * (attention_params + moe_mlp_params)
    )

    effective_qk_dim = qk_nope_dim + qk_rope_dim

    flops = 0
    flops += 2 * seq_len * hidden_size * q_lora_rank
    flops += 2 * seq_len * q_lora_rank * num_heads * effective_qk_dim
    flops += 2 * seq_len * hidden_size * (kv_lora_rank + qk_rope_dim)
    flops += 2 * seq_len * kv_lora_rank * num_heads * (qk_nope_dim + v_head_dim)
    flops += 2 * seq_len * num_heads * v_head_dim * hidden_size
    flops += 2 * num_heads * seq_len * seq_len * effective_qk_dim
    flops += 2 * num_heads * seq_len * seq_len * v_head_dim
    flops += 2 * seq_len * hidden_size * moe_intermediate_size * 3 * (
        num_active_experts + num_shared_experts
    )

    flops_layer_TF = flops / 1e12

    bytes_per_element = 2
    moe_layer_params = attention_params + moe_mlp_params
    peak_memory = (
        moe_layer_params * bytes_per_element
        + seq_len * hidden_size * bytes_per_element
        + seq_len * num_heads * effective_qk_dim * bytes_per_element
        + 2 * seq_len * kv_lora_rank * bytes_per_element
        + num_heads * seq_len * seq_len * bytes_per_element
    )
    peak_memory_GB = peak_memory / (1024 ** 3)

    return total_params, flops_layer_TF, peak_memory_GB


def get_optimal_N_D_from_cost(cost_budget):
    """Pick the GPU and (N, D) that minimize loss under a $ training budget.

    cost_budget: a monetary training budget (in dollars)
    Returns:
        N: optimal model parameter count (absolute number)
        D: optimal training token count (absolute number)
        training_budget_flops: effective total training FLOPs
        best_gpu: name of the selected GPU, one of {'H100', 'H200', 'B200'}

    See the Part 2.2 writeup for the scaling law, the GPU price / TFLOPs
    table, and the MFU assumption.
    """
    gpus = {
        "H100": {"price_per_hour": 3.0, "peak_tflops": 989.0},
        "H200": {"price_per_hour": 4.0, "peak_tflops": 989.0},
        "B200": {"price_per_hour": 6.0, "peak_tflops": 2250.0},
    }

    mfu = 0.40
    best_gpu = None
    training_budget_flops = 0.0

    for gpu_name, gpu_spec in gpus.items():
        gpu_hours = cost_budget / gpu_spec["price_per_hour"]
        available_flops = gpu_hours * 3600 * gpu_spec["peak_tflops"] * 1e12 * mfu

        if available_flops > training_budget_flops:
            training_budget_flops = available_flops
            best_gpu = gpu_name

    a = 406.4
    alpha = 0.34
    b = 410.7
    beta = 0.29

    N = int(round(((a * alpha) / (b * beta) * (training_budget_flops / 6) ** beta) ** (1 / (alpha + beta))))
    D = int(round(training_budget_flops / (6 * N)))

    return N, D, training_budget_flops, best_gpu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training cost analysis")
    parser.add_argument("--model_config", type=str, help="Path to model config")
    parser.add_argument(
        "--training_budget",
        type=float,
        default=None,
        help="Training budget in dollars",
    )
    args = parser.parse_args()

    if args.model_config:
        if "deepseek" in args.model_config:
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_deepseek(
                args.model_config
            )
        elif "llama" in args.model_config:
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_llama(
                args.model_config
            )
        else:
            print("Unknown model type — name your config llama*.json or deepseek*.json")
            raise SystemExit(1)

        print(f"Number of parameters: {num_parameters}")
        print(f"Number of TFLOPs: {num_flops}")
        print(f"Peak memory cost: {memory_cost} GBs")

    if args.training_budget:
        N, D, training_budget_flops, best_gpu = get_optimal_N_D_from_cost(
            args.training_budget
        )
        print(f"best_gpu: {best_gpu}")
        print(f"training_budget_flops: {training_budget_flops}")
        print(f"Optimal N: {N}")
        print(f"Optimal D: {D}")