# PA3 Part 3 — Speculative Decoding Report

## Setup

- **Target model:** `EleutherAI/pythia-1.4b-deduped` (~1.4B parameters)
- **Draft model:**  `EleutherAI/pythia-160m-deduped` (~160M parameters)
- **Device:** CUDA (T4 GPU via Google Colab)
- **Dtype:** fp16 for both models
- **Verification strategy:** Greedy (argmax), vectorized single target forward pass
- **Draft strategy:** Greedy (`do_sample=False`), maximizes draft acceptance rate

## Implementation Summary

### `initialize_target_model` / `initialize_draft_model`

Both models are loaded in fp16 via `AutoModelForCausalLM.from_pretrained(..., dtype=torch.float16)`,
placed on `self.device`, set to `eval()` mode, and have `use_cache=True` to enable KV caching.

### `generate_draft_tokens`

Uses the draft model's `.generate()` with `do_sample=False` (greedy) and `max_new_tokens=k`.
Draft tokens are extracted by slicing off the input prefix: `output[:, input_ids.shape[1]:]`.

### `verify_tokens_vectorized`

Concatenates `[input_ids | draft_tokens]` into a single sequence of length `L+k`, runs one
target forward pass, then reads logits at positions `[L-1 : L+k]` (k+1 positions) to:
- Verify draft tokens `[0..k-1]` against the target's greedy predictions
- If all k draft tokens are accepted, collect one free **bonus token** from `logits[L+k-1]`

### `speculative_decode` main loop

Each iteration: (1) draft model proposes `k` tokens greedily, (2) target verifies in one
batched forward pass, (3) accepted tokens plus correction/bonus are appended. Loop stops on
EOS or `max_tokens` reached.

---

## Results — Benchmark across 3 prompts (k=8, num_runs=3)

| Prompt | Speculative (avg) | Baseline (avg) | Speedup | Acceptance Rate |
|--------|:-----------------:|:--------------:|:-------:|:---------------:|
| "The future of artificial intelligence is" | 2.13s / 64.37 tok/s | 1.83s / 54.59 tok/s | 0.86x* | 91.67% |
| "Write a short story about a robot learning to feel emotions:" | 1.42s / 73.13 tok/s | 1.86s / 53.75 tok/s | **1.31x** | 100.00% |
| "Write the lyrics to the song 'Happy Birthday'." | 1.36s / 74.84 tok/s | 2.01s / 50.26 tok/s | **1.48x** | 93.68% |

\* Prompt 1 was the first run after model loading (cold GPU). Run 1 took 3.98s due to CUDA
warm-up; runs 2 and 3 achieved 83.98 and 84.25 tok/s. On a warm GPU (sweep below) the same
prompt reaches 1.71x speedup at k=2. Prompts 2 and 3 both exceed 1.0x.

---

## Sweep: `num_speculative_tokens` ∈ {2, 4, 8, 16}

Prompt: "The future of artificial intelligence is", warm GPU, 3 runs each.
Baseline (warm): 48.49 tok/s, 2.09s.

| k  | Tokens/s | Acceptance Rate | Speedup vs baseline |
|----|:--------:|:---------------:|:-------------------:|
|  2 |   81.67  |     91.67%      |        1.71x        |
|  4 |   81.05  |     91.67%      |        1.69x        |
|  8 |   81.20  |     91.67%      |        1.70x        |
| 16 |   69.21  |     91.67%      |        1.43x        |

---

## Analysis

### Acceptance Rate

The acceptance rate is constant at 91.67% across all k values for the same prompt. With
greedy (deterministic) generation, the text produced is identical for every k, so the
fraction of positions where draft and target agree is fixed. Different prompts yield different
acceptance rates (91.67%, 100%, 93.68%), reflecting how well the draft model's distribution
aligns with the target for each specific text.

### Speedup

k=2 through k=8 all deliver ~1.70x speedup on a warm GPU. k=16 drops to 1.43x because
at larger k the draft model runs longer while the acceptance rate stays fixed — the draft
overhead grows but tokens accepted per iteration does not increase proportionally.

### Optimizations Applied

1. **Greedy draft (`do_sample=False`):** Maximizes acceptance probability.
2. **fp16 throughout:** Halves memory bandwidth, increasing effective GPU throughput.
3. **KV cache (`use_cache=True`):** Skips re-computing attention over the seen prefix.
4. **Vectorized verification:** One target forward pass verifies all k tokens at once.
5. **Bonus token on full acceptance:** When all k drafts are accepted, the target's
   logit at position `L+k-1` yields one extra free token per iteration.

---

## Conclusion

Achieved >1.0x wall-clock speedup on 2 of 3 prompts (1.31x, 1.48x). Prompt 1 shows
0.86x on average due to a cold GPU on run 1; the same prompt achieves 1.71x on a warm GPU
(confirmed in the sweep). Draft-token acceptance rate is 91.67–100% on all prompts (≥75%).

---

## Bonus 3.B — N-gram Lookup Decoding

### Implementation

`NGramLookupDecoder` extends `SpeculativeDecoder`. A `from_decoder(base)` classmethod
reuses already-loaded target and draft models without reloading.

At each step, `_ngram_lookup` scans the running context for the last `ngram_size=4` tokens.
If a match is found, the following tokens are used as draft (no model call). Otherwise it
falls back to the standard draft model. Verification is the same `verify_tokens_vectorized`.

### Results (ngram_size=4, k=8, 3 runs each, warm GPU)

| Prompt | N-gram tok/s | Baseline tok/s | Speedup | Acceptance Rate | N-gram Hit Rate |
|--------|:------------:|:--------------:|:-------:|:---------------:|:---------------:|
| "Write the lyrics to the song 'Happy Birthday'." | **241.18** | 52.19 | **4.58x** | 93.68% | 83.33% |
| "The future of artificial intelligence is" | **237.73** | 49.00 | **4.95x** | 91.67% | 83.33% |
| "Write a short story about a robot learning to feel emotions:" | **204.05** | 50.54 | **4.02x** | 100.00% | 75.00% |
| **Average** | **227.65** | **50.58** | **4.52x** | **~95%** | **~80%** |

### Why the speedup is so large

N-gram draft tokens are exact copies of tokens the target already produced deterministically.
Because the target is greedy, it must agree with its own prior output — acceptance on n-gram
hits is near 100%. For iterations where the cache misses, the decoder falls back to the draft
model at the same ~91–93% acceptance rate. With a 75–83% hit rate, the majority of draft
proposals cost only a context lookup instead of a model forward pass, giving 4–5x overall
speedup vs. ~1.7x for draft-only speculative decoding.
