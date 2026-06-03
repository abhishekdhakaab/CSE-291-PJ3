# PA3 — Speculative Decoding Report

Used NVIDIA CUDA 4090

---

## 3.2 Performance Results (`k=8`, `max_tokens=100`, 3 runs)

| Prompt | Speculative Tok/s | Baseline Tok/s | Speedup | Acceptance Rate |
|--------|-------------------|----------------|---------|-----------------|
| "The future of artificial intelligence is" | 46.34 | 35.43 | 1.26× | 91.67% |
| "Write a short story about a robot learning to feel emotions:" | 48.22 | 31.20 | 1.53× | 100.00% |
| "Write the lyrics to the song 'Happy Birthday'." | 50.31 | 35.53 | 1.40× | 93.68% |

All three prompts exceed the ≥1.0× speedup and ≥75% acceptance rate requirements. The robot story prompt hits 100% acceptance which means the draft predicts every token exactly as the target would. The AI future prompt is the weakest at 1.26× and also has the lowest acceptance rate 91.67%.

---

## 3.3 Sweep: `num_speculative_tokens` ∈ {2, 4, 8, 16}

Prompt: "The future of artificial intelligence is", baseline: 3.06s / 32.7 tok/s.

| k  | Avg Time (s) | Tok/s | Speedup | Acceptance Rate |
|----|-------------|-------|---------|-----------------|
|  2 | 2.36 | 42.5 | 1.30× | 97.06% |
|  4 | 2.06 | 49.1 | 1.48× | 95.24% |
|  8 | 1.98 | 50.5 | 1.55× | 91.67% |
| 16 | 2.15 | 47.1 | 1.42× | 85.45% |

All k values beat the baseline and have >=1.0x speedup. k=8 gives the best speedup at 1.55×. Acceptance rate decreases as k grows. This is because the draft's predictions become less reliable further out, so longer speculations are more likely to contain a mismatch.

---

## Optimizations

Both models are loaded in float16, halving memory usage and speeding up matrix multiplications compared to float32. The draft model uses greedy decoding, which maximises the acceptance rate for a same-distribution model. The KV cache is enabled, so each step processes only one new token rather than recomputing the full context. The biggest optimization though is the vectorized verification, where the full draft is added to the context so that there only needs to be one forward pass over L+k tokens to verify all k tokens.

---
## Bonus 3.B — N-gram Lookup Decoding

### Implementation

`NGramLookupDecoder` extends `SpeculativeDecoder`. A `from_decoder(base)` classmethod
reuses already-loaded target and draft models without reloading.

At each step, `_ngram_lookup` scans the running context for the last `ngram_size=4` tokens.
If a match is found, the following tokens are used as draft (no model call). Otherwise it
falls back to the standard draft model. Verification is the same `verify_tokens_vectorized`.


### Results (ngram_size=4, k=8, 3 runs each)

| Prompt | N-gram tok/s | Baseline tok/s | Speedup | Acceptance Rate | N-gram Hit Rate |
|--------|:------------:|:--------------:|:-------:|:---------------:|:---------------:|
| "Write the lyrics to the song 'Happy Birthday'." | 141.94 | 22.70 | 6.16× | 93.68% | 83.33% |
| "The future of artificial intelligence is" | 126.14 | 22.74 | 5.60× | 91.67% | 83.33% |
| "Write a short story about a robot learning to feel emotions:" | 98.73 | 27.44 | 3.39× | 100.00% | 75.00% |
| Average | 122.27 | 24.29 | 5.05× | ~95% | ~80% |

### Why the speedup is so large

N-gram draft tokens are exact copies of tokens the target already produced deterministically.
Because the target is greedy, it must agree with its own prior output — acceptance on n-gram
hits is near 100%. For iterations where the cache misses, the decoder falls back to the draft
model at the same ~91–93% acceptance rate. With a 75–83% hit rate, the majority of draft
