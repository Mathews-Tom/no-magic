# MicroKV Challenges

Test your understanding of KV cache by predicting what happens in these scenarios. Try to work out the answer before revealing it.

---

### Challenge 1: Multiply Count Growth

**Setup:** `linear_f` (line 210) increments a global `multiply_count` by `len(x) * len(w[0])` for every matrix multiply. `generate_no_cache` re-runs the full sequence through the model at every step (lines 233-308). `generate_with_cache` only processes the new token (lines 318-396). `GEN_LEN = 16` (line 51), `PROMPT = "anna"` (4 tokens, line 57).

**Question:** For a prompt of length 4 and generation of 16 tokens, the no-cache path processes sequences of length 5, 6, 7, ..., 20 (growing by 1 each step). If each forward pass costs multiply_count proportional to sequence length, how does total multiply work scale compared to the cached version?

<details>
<summary>Reveal Answer</summary>

**Answer:** No-cache total work is proportional to `sum(5, 6, ..., 20) = 200` sequence-length units. With cache, each of the 16 generation steps processes exactly 1 new token regardless of history, so total work is `4 (prompt) + 16 * 1 = 20` sequence-length units — a 10x difference here, and the gap grows quadratically as sequence length increases.

**Why:** The no-cache path calls `gpt_forward(tokens[:i+1], ...)` on lines 261-263, where `i` grows from `prompt_len` to `prompt_len + GEN_LEN - 1`. Each call recomputes attention over the full prefix from scratch. Attention cost scales as `O(T^2)` in general (each token attends to all predecessors), but even ignoring the quadratic attention term, the linear layer multiplies alone scale as `O(T)` per step, making the total `O(T^2)` across all generation steps. The cached path on lines 318-396 appends the new token's K and V to stored lists (`kv_cache[layer]['k'].append(...)`) and computes attention using the stored keys/values. The new token attends to all previous tokens in O(T) total attention cost, but the Q, K, V projections only require one new token's worth of computation. Across 16 steps this is `O(T)` total — linear not quadratic in generated length.

**Script reference:** `03-systems/microkv.py`, lines 210-211 (multiply_count increment), lines 233-308 (no-cache generation recomputing full prefix), lines 318-396 (cached generation processing one token per step), lines 498-510 (multiply_count comparison printed after both runs)

</details>

---

### Challenge 2: Cache Memory Formula

**Setup:** The memory estimate at line 393 is `cache_memory = 2 * N_LAYER * N_EMBD * len(kv_cache[0]['k'])`. `N_LAYER = 2` (line 40), `N_EMBD = 16` (line 38). Each cache entry stores both a key vector and a value vector of size `N_EMBD` per layer.

**Question:** After generating 16 tokens from a 4-token prompt, how many floats are stored in the KV cache? If each float were 2 bytes (FP16), how many bytes is that? At this rate, how would cache size grow if you generated 10,000 tokens instead?

<details>
<summary>Reveal Answer</summary>

**Answer:** After 20 total tokens (4 prompt + 16 generated): `2 * 2 * 16 * 20 = 1,280 floats`. At 2 bytes each: `2,560 bytes ≈ 2.5 KB`. At 10,000 tokens: `2 * 2 * 16 * 10000 = 640,000 floats = 1.28 MB` (FP16). The cache grows linearly in sequence length.

**Why:** The formula `2 * N_LAYER * N_EMBD * T` captures: factor of 2 for K and V tensors, `N_LAYER` layers each storing their own K/V, `N_EMBD` values per token per layer, and `T` tokens accumulated. This is the fundamental memory cost of KV caching — trading computation (quadratic recomputation) for memory (linear storage). In production LLMs with N_EMBD=4096, N_LAYER=32, and FP16 storage, the KV cache costs `2 * 32 * 4096 * T * 2 bytes = 524,288 * T bytes ≈ 0.5 MB per token`. At 100K token contexts (GPT-4 class), this is 50 GB of KV cache — which is why memory-efficient attention variants and paged attention systems exist.

**Script reference:** `03-systems/microkv.py`, lines 38-40 (N_EMBD, N_LAYER hyperparameters), lines 369-374 (K/V append operations per layer), lines 390-393 (memory calculation and print), lines 494-497 (memory comparison between methods)

</details>

---

### Challenge 3: Paged Attention Block Boundaries

**Setup:** `simulate_paged_attention` (lines 406-447) divides the KV cache into fixed-size blocks: `block_size = 4` (line 412). Logical position maps to blocks via `logical_block = pos // block_size` (line 419). The comment on lines 409-411 explains why contiguous allocation causes fragmentation.

**Question:** If you run 3 concurrent sequences with lengths 5, 3, and 7 tokens, how many physical blocks does contiguous allocation waste vs paged allocation? A block holds 4 token slots — what happens to the partially-filled last block in each sequence under contiguous allocation?

<details>
<summary>Reveal Answer</summary>

**Answer:** Contiguous allocation: sequence of 5 needs `ceil(5/4) = 2` blocks (1 wasted slot), sequence of 3 needs 1 block (1 wasted slot), sequence of 7 needs 2 blocks (1 wasted slot). Total: 5 physical blocks needed, 3 slots wasted (60% of 1 block). Paged allocation allocates exactly the blocks needed as tokens arrive, and the last block of each sequence is still partially filled — but no more than under contiguous allocation. The difference is that paged allocation can reuse freed blocks from completed sequences without the fragmentation of contiguous reserved regions.

**Why:** Contiguous allocation must reserve a contiguous physical memory region for the maximum possible length of each sequence at allocation time, since extending a contiguous region later requires copying. This means short sequences that terminate early leave their reserved-but-unused region permanently unavailable until explicitly freed and coalesced. Paged allocation (as in vLLM) uses a page table: each block can be physically anywhere in memory, so a new sequence can occupy the first available free block regardless of its physical address. The `physical_block = page_table[seq_id][logical_block]` lookup on line 421 shows this indirection — logical position maps through the page table to a physical block that could be anywhere in the pool. This eliminates fragmentation at the cost of one indirection per cache access.

**Script reference:** `03-systems/microkv.py`, lines 406-447 (simulate_paged_attention), lines 412-413 (block_size and block pool), lines 419-424 (logical-to-physical mapping), lines 409-411 (fragmentation motivation comment)

</details>

---

### Challenge 4: Why Cached Attention Still Needs All Previous K/V

**Setup:** In `generate_with_cache`, the new token's query `q_new` is computed from just the new token (line 356), but attention is computed over all stored keys: `for j, k_vec in enumerate(kv_cache[layer]['k'])` (lines 361-366). The cache grows by appending new K and V on lines 369-374.

**Question:** Could you save even more memory by only storing the last `W` tokens in the cache (a sliding window)? What would break if you did this for the standard causal language model in this script?

<details>
<summary>Reveal Answer</summary>

**Answer:** A sliding window cache would cause the model to "forget" tokens older than `W` steps. For a generative language model, this means the generated text after step W could become inconsistent with the prompt — the model can no longer attend to the original question or early context. Output quality degrades for long sequences.

**Why:** The causal attention mask allows token `t` to attend to all tokens `0..t-1`. When generating token `t`, the correct prediction depends on computing `softmax(q_t @ K[0..t-1].T / sqrt(d)) @ V[0..t-1]`. If the cache only stores tokens `t-W..t-1`, the attention distribution is incorrect — tokens that should receive high attention weight (e.g., a name from the prompt) may be entirely absent. The model was trained with full causal attention, so at inference time the distribution shift from windowed attention causes degraded outputs. Sliding window attention (as in Mistral's Sliding Window Attention) works when the model was also *trained* with the same window size — the model learns to not depend on tokens beyond W steps back. Using a sliding window cache on a model trained with full attention is a post-hoc approximation that causes systematic error. The comment on line 397 notes that the full cache implementation here stores all prior tokens deliberately for correctness.

**Script reference:** `03-systems/microkv.py`, lines 356-374 (new token Q computation and K/V append), lines 361-366 (attention over all cached K), lines 397-404 (cache correctness comment), lines 318-320 (generate_with_cache docstring)

</details>
