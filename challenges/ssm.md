# MicroSSM Challenges

Test your understanding of State Space Models (Mamba-style) by predicting what happens in these scenarios. Try to work out the answer before revealing it.

---

### Challenge 1: Fixed State Size vs Growing KV Cache

**Setup:** The SSM state is a fixed-size tensor `h[n]` of shape `[N_STATE]` per dimension (lines 350-360 in `selective_scan`). `N_STATE = 8` (line 38), `N_EMBD = 16` (line 37). Compare to the KV cache in `microkv.py` which grows by `N_EMBD` values per layer per new token.

**Question:** After processing a 1,000-token sequence, how much memory does the SSM state use (in floats)? How does this compare to a KV cache for the same sequence with `N_LAYER = 2`, `N_EMBD = 16`? What is the fundamental tradeoff this size difference represents?

<details>
<summary>Reveal Answer</summary>

**Answer:** SSM state: `N_STATE * N_EMBD = 8 * 16 = 128 floats` — the same regardless of sequence length (1 token or 1,000 tokens). KV cache after 1,000 tokens: `2 * N_LAYER * N_EMBD * T = 2 * 2 * 16 * 1000 = 64,000 floats`. The SSM uses 500x less memory at T=1,000, and the ratio grows linearly with sequence length.

**Why:** The SSM compresses the entire history of processed tokens into a fixed-size "state" vector `h`. At each step, the state is updated via `h[n] = a_bar[n] * h[n] + b_bar[n] * x_t[d]` (lines 421-423), discarding the raw token and keeping only what the learned `A`, `B`, `C` matrices say is worth remembering. This is fundamentally lossy — information from 1,000 tokens ago may be attenuated or lost depending on the `A` eigenvalues. The KV cache is lossless: every token's key and value vector is stored verbatim, allowing perfect recall of any past token (subject to attention weight). The SSM's O(1) memory per step enables processing arbitrarily long sequences without memory growth, at the cost of potentially forgetting long-range details. Transformers with KV cache guarantee exact recall but require O(T) memory.

**Script reference:** `03-systems/microssm.py`, lines 38 (N_STATE), lines 37 (N_EMBD), lines 350-360 (state initialization in selective_scan), lines 421-423 (state update formula), lines 540-548 (memory comparison in analysis section)

</details>

---

### Challenge 2: The Delta Bias and Input-Dependent Timescales

**Setup:** `delta_bias = -2.0` (line 314). The discretization computes `delta = softplus(delta_raw + delta_bias)` (line 405), where `delta_raw` comes from a linear projection of the input. The comment on lines 315-318 explains that negative bias makes `delta` small by default. `a_bar = 1 + delta_d * A_diag[n]` (line 421) uses Euler discretization (not ZOH).

**Question:** When `delta_raw = 0.0` (the network projects a "neutral" input), what is `delta` after softplus with `delta_bias = -2.0`? What does a small `delta` mean for how much the state changes at this timestep? Why is the default small rather than large?

<details>
<summary>Reveal Answer</summary>

**Answer:** `delta = softplus(0.0 + (-2.0)) = softplus(-2.0) = log(1 + exp(-2.0)) ≈ log(1 + 0.135) ≈ log(1.135) ≈ 0.127`. This is a small positive number. With `delta ≈ 0.127` and `A_diag[n] ≈ -0.5` (typical initialized value), `a_bar = 1 + 0.127 * (-0.5) ≈ 0.937` — close to 1, meaning the state barely changes (high retention of previous state, low incorporation of new input).

**Why:** The `delta` parameter controls the "step size" of the discretization — how aggressively the continuous-time SSM dynamics are applied to one discrete token step. A large `delta` means the state rapidly forgets the past (the continuous-time system is allowed to evolve far) and incorporates the new input strongly. A small `delta` means the state is nearly frozen (the system barely evolves), preserving the existing memory. The default bias of `-2.0` makes the network "conservative by default" — most tokens make small updates to the state. This mirrors how language works: most tokens are background context, only occasionally does a key token (a name, a number, a negation) require a large state update. The input-dependent `delta_raw` projection lets the network learn to open the gate wide for important tokens. This selectivity is what distinguishes Mamba from S4 (which has fixed `delta`).

**Script reference:** `03-systems/microssm.py`, lines 314-318 (delta_bias initialization and comment), lines 403-410 (delta computation via softplus), lines 421-423 (Euler discretization with delta), lines 269-332 (init_ssm_params with all parameter initializations)

</details>

---

### Challenge 3: Selective B and C vs Fixed

**Setup:** In `selective_scan` (lines 337-438), `B` and `C` are computed per-token from the input: `B_t = [sum(...) for ...]` (lines 413-416) and `C_t = [sum(...) for ...]` (lines 417-419). The comment on lines 339-345 contrasts this with the original S4 model where `A`, `B`, `C` are fixed matrices.

**Question:** If `B` and `C` were fixed (not input-dependent), what would the SSM lose compared to the selective version? Give a concrete example of why selectivity matters for language.

<details>
<summary>Reveal Answer</summary>

**Answer:** With fixed `B` and `C`, every token would write the same relative amounts to the state (fixed `B`) and read from the state in the same pattern (fixed `C`). The model could not learn to strongly memorize a specific token (like a name) when it first appears and then precisely retrieve it many tokens later.

**Why:** Fixed `B` means the "input gate" applies the same projection to every token regardless of content. A pronoun "she" and a proper name "Maria" would update the state with identically-scaled projections, making it impossible to give the name a stronger memory trace. Fixed `C` means the "output gate" retrieves from the state in the same pattern for every query — the model cannot learn to "look up" a specific slot in the state when generating a token that depends on a specific earlier word. With selective `B` and `C` computed from the current input `x_t` (via learned linear projections `W_B` and `W_C`), the network can learn: when processing "Maria" (content-dependent), use a large `B` component to write the name's representation strongly into a specific state dimension; when generating a pronoun that should agree with "Maria", use a large `C` component on that same dimension to retrieve it. This input-dependent memory access is the core contribution of Mamba over prior SSMs like S4 and is why the comment on line 341 calls it "the Mamba contribution."

**Script reference:** `03-systems/microssm.py`, lines 337-345 (selective_scan docstring explaining selectivity), lines 413-419 (input-dependent B_t and C_t computation), lines 279-295 (W_B and W_C projection matrix initialization), lines 339-345 (fixed vs selective comparison comment)

</details>

---

### Challenge 4: Euler vs ZOH Discretization

**Setup:** The discretization uses Euler method: `a_bar = 1 + delta_d * A_diag[n]` (line 421), `b_bar = delta_d * B_t[n]` (line 422). The comment on lines 425-431 notes that Zero-Order Hold (ZOH) would use `a_bar = exp(delta * A)` and `b_bar = A^(-1) * (exp(delta*A) - I) * B`. `log_A` is initialized to negative values: `log_A = [math.log(n+1) * -1 for n in range(N_STATE)]` (line 306).

**Question:** The `log_A` initialization gives `A_diag[n] = -log(n+1)`, which means `A_diag[0] = 0`, `A_diag[1] ≈ -0.693`, etc. With Euler discretization and `delta = 0.1`, compute `a_bar` for n=0 and n=1. What does `a_bar[0] = 1.0` mean for state dimension 0?

<details>
<summary>Reveal Answer</summary>

**Answer:** For n=0: `a_bar = 1 + 0.1 * 0 = 1.0`. For n=1: `a_bar = 1 + 0.1 * (-0.693) = 1 - 0.0693 = 0.931`. State dimension 0 has `a_bar = 1.0` — it never decays. State dimension 1 retains 93.1% of its value each step.

**Why:** When `a_bar = 1.0`, the state update is `h[0] = 1.0 * h[0] + b_bar * x`, making dimension 0 a pure integrator: it accumulates all past inputs without any forgetting. This creates a "memory channel" that preserves the running sum of all inputs. Higher state dimensions (n=1, 2, ...) have increasingly negative `A_diag` values (since `log(n+1)` grows with n), giving faster decay rates — they function as short-term memory channels. This structured initialization gives the SSM a range of timescales out of the box: one dimension for long-term integration, others for progressively shorter windows. ZOH would give the correct continuous-time solution `a_bar = exp(delta * A)`, but for `A = 0`, both methods agree: `exp(0) = 1`. The Euler approximation is valid when `delta` is small; the comment on line 425 acknowledges this simplification. The `log_A` parameterization (always negative after negation) ensures `A_diag <= 0`, keeping all `a_bar <= 1` under Euler — the state cannot explode.

**Script reference:** `03-systems/microssm.py`, lines 305-310 (log_A initialization), lines 419-423 (Euler discretization in selective_scan), lines 425-431 (ZOH comparison comment), lines 396-401 (A_diag computation from log_A via exp)

</details>
