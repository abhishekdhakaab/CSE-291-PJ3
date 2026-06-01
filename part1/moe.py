"""Mixture-of-Experts: reference, tensor-parallel, and expert-parallel variants."""
import numpy as np

from mpi_wrapper import mpi
from rng import get_rng, rng_context


class Linear:
    def __init__(self, in_features, out_features):
        self.weight = get_rng().randn(in_features, out_features) * 0.01
        self.bias = np.zeros(out_features)

    def __call__(self, x):
        return np.dot(x, self.weight) + self.bias


class Expert:
    def __init__(self, input_dim, hidden_dim, output_dim):
        with rng_context("expert"):
            self.fc1 = Linear(input_dim, hidden_dim)
            self.fc2 = Linear(hidden_dim, output_dim)

    def __call__(self, x):
        return self.fc2(np.maximum(0, self.fc1(x)))


class Router:
    """Softmax-gated top-k router (replicated across ranks)."""

    def __init__(self, input_dim, num_experts):
        self.linear = Linear(input_dim, num_experts)

    def __call__(self, x, topk=1):
        logits = self.linear(x)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        indices = np.argsort(-probs, axis=1)[:, :topk]
        gates = np.take_along_axis(probs, indices, axis=1)
        gates = gates / np.sum(gates, axis=1, keepdims=True)
        return indices, gates


# ---------------------------------------------------------------------------
# Reference implementation (provided — do not modify).
# ---------------------------------------------------------------------------
class SimpleMoE:
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1):
        self.output_dim = output_dim
        self.topk = min(topk, num_experts)
        with rng_context("router"):
            self.router = Router(input_dim, num_experts)
        with rng_context("expert"):
            self.experts = [Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)]

    def forward(self, x):
        batch_size = x.shape[0]
        indices, gates = self.router(x, self.topk)
        outputs = np.zeros((batch_size, self.output_dim))
        for k in range(self.topk):
            for i in range(batch_size):
                outputs[i] += gates[i, k] * self.experts[indices[i, k]](x[i:i+1])[0]
        return outputs

    def __call__(self, x):
        return self.forward(x)


# ---------------------------------------------------------------------------
# Part 1.1 — Tensor Parallel MoE
# ---------------------------------------------------------------------------
class ShardedLinear:
    """Column-sharded linear layer. Each rank owns out_features // world_size columns.
    A single allgather reassembles the full output on every rank.
    """

    def __init__(self, in_features, out_features):
        self.world_size = mpi.Get_size()
        assert out_features % self.world_size == 0
        self.out_features_global = out_features
        local_out = out_features // self.world_size
        self.weight = get_rng().randn(in_features, local_out) * 0.01
        self.bias = get_rng().randn(local_out)

    def __call__(self, x):
        # Compute local columns, allgather across ranks, concatenate.
        local_out = (np.dot(x, self.weight) + self.bias).astype(np.float32)
        return np.concatenate(mpi.allgather(local_out), axis=1)


class ShardedExpert:
    def __init__(self, input_dim, hidden_dim, output_dim):
        with rng_context("expert"):
            self.fc1 = ShardedLinear(input_dim, hidden_dim)
            self.fc2 = ShardedLinear(hidden_dim, output_dim)

    def __call__(self, x):
        return self.fc2(np.maximum(0, self.fc1(x)))


class MoE_TP:
    """Tensor-parallel MoE: every rank holds a slice of every expert's weights."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1):
        self.output_dim = output_dim
        self.topk = min(topk, num_experts)
        if mpi.Get_rank() == 0:
            print(f"[MoE_TP] world_size={mpi.Get_size()}, num_experts={num_experts}, topk={self.topk}")
        with rng_context("router"):
            self.router = Router(input_dim, num_experts)
        with rng_context("expert"):
            self.experts = [ShardedExpert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)]

    def forward(self, x):
        batch_size = x.shape[0]
        indices, gates = self.router(x, self.topk)
        outputs = np.zeros((batch_size, self.output_dim))
        for k in range(self.topk):
            for i in range(batch_size):
                outputs[i] += gates[i, k] * self.experts[indices[i, k]](x[i:i+1])[0]
        return outputs

    def __call__(self, x):
        return self.forward(x)


# ---------------------------------------------------------------------------
# Part 1.2 — Expert Parallel MoE
# ---------------------------------------------------------------------------
class MoE_EP:
    """Expert-parallel MoE: each rank owns exactly one expert.
    Two alltoall rounds dispatch tokens to expert-owning ranks and return results.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.topk = min(topk, num_experts)
        self.world_size = mpi.Get_size()
        assert num_experts == self.world_size
        with rng_context("router"):
            self.router = Router(input_dim, num_experts)
        with rng_context("expert_with_rank"):
            self.expert = Expert(input_dim, hidden_dim, output_dim)

    def _alltoall(self, send_buckets):
        """Exchange variable-row arrays via alltoall with padding."""
        # Share row counts so every rank knows how many rows to expect.
        send_counts = np.array([b.shape[0] for b in send_buckets], dtype=np.int64)
        recv_counts = np.empty_like(send_counts)
        mpi.myAlltoall(send_counts, recv_counts)

        # Pad each bucket to the global max row count so alltoall segments are equal.
        max_rows = max(1, max(mpi.allgather(int(send_counts.max()))))
        width = send_buckets[0].shape[1]
        send_buf = np.zeros((self.world_size, max_rows, width), dtype=send_buckets[0].dtype)
        for r, b in enumerate(send_buckets):
            send_buf[r, :b.shape[0]] = b

        recv_buf = np.empty_like(send_buf)
        mpi.myAlltoall(send_buf, recv_buf)

        return [recv_buf[r, :int(recv_counts[r])].copy() for r in range(self.world_size)]

    def forward(self, x):
        batch_size = x.shape[0]
        indices, gates = self.router(x, self.topk)
        outputs = np.zeros((batch_size, self.output_dim), dtype=np.float64)

        for k in range(self.topk):
            # Build per-rank token buckets and remember which batch position they came from.
            buckets = [[] for _ in range(self.world_size)]
            positions = [[] for _ in range(self.world_size)]
            for i in range(batch_size):
                r = int(indices[i, k])
                buckets[r].append(x[i])
                positions[r].append(i)

            send = [
                np.array(buckets[r], dtype=x.dtype) if buckets[r]
                else np.zeros((0, self.input_dim), dtype=x.dtype)
                for r in range(self.world_size)
            ]

            # Round 1: send tokens to expert owners.
            recv_tokens = self._alltoall(send)

            # Run local expert on all received tokens.
            all_recv = np.concatenate([t for t in recv_tokens if t.shape[0] > 0], axis=0) \
                if any(t.shape[0] > 0 for t in recv_tokens) \
                else np.zeros((0, self.input_dim), dtype=x.dtype)
            all_results = self.expert(all_recv) if all_recv.shape[0] > 0 \
                else np.zeros((0, self.output_dim), dtype=x.dtype)

            # Split results back by source rank.
            result_buckets, offset = [], 0
            for r in range(self.world_size):
                n = recv_tokens[r].shape[0]
                result_buckets.append(all_results[offset:offset + n])
                offset += n

            # Round 2: send results back to originating ranks.
            recv_results = self._alltoall(result_buckets)

            # Scatter into output.
            for r in range(self.world_size):
                for j, tok_idx in enumerate(positions[r]):
                    if j < recv_results[r].shape[0]:
                        outputs[tok_idx] += gates[tok_idx, k] * recv_results[r][j]

        return outputs.astype(np.float32)

    def __call__(self, x):
        return self.forward(x)
