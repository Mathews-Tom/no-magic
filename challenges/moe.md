# MicroMoE Challenges

Test your understanding of Mixture of Experts by predicting what happens in these scenarios. Try to work out the answer before revealing it.

---

### Challenge 1: Expert Collapse Without Auxiliary Loss

**Setup:** `AUX_LOSS_COEFF = 0.1` (line 52), which weights the load balancing loss. The auxiliary loss penalizes uneven routing by computing `N_EXPERTS * sum(f_i * P_i)` (lines 408-430). The comment on lines 399-415 explains the "rich get richer" failure mode.

**Question:** If you set `AUX_LOSS_COEFF = 0.0`, which expert(s) would likely receive the most tokens after training? Would the model's language modeling loss be better, worse, or the same compared to `AUX_LOSS_COEFF = 0.1`?

<details>
<summary>Reveal Answer</summary>

**Answer:** Without the auxiliary loss, 1-2 experts would receive nearly all tokens (expert collapse). The LM loss might be slightly lower because the optimizer can overfit the best expert without the constraint of distributing load, but generalization would suffer.

**Why:** Early in training, slight random variation in loss means some experts produce marginally better outputs for certain inputs. The router receives positive gradient signal for routing to these experts, increasing their probability. This creates a positive feedback loop: better experts get more tokens, receive more gradient, improve more, get even more tokens. Unused experts receive zero gradient and stay at random initialization. Without the `f_i * P_i` penalty (which increases when any expert has both high `f_i` usage fraction AND high `P_i` router probability), there's no force breaking this symmetry. The comment on lines 656-667 describes the threshold check: experts below 10% utilization indicate collapse. The lower LM loss comes at the cost of unused capacity — 4 experts minus 1-2 active means 50-75% of expert parameters are wasted.

**Script reference:** `02-alignment/micromoe.py`, lines 52-55 (AUX_LOSS_COEFF with comment), lines 392-430 (auxiliary loss computation and explanation), lines 653-667 (collapse detection)

</details>

---

### Challenge 2: Top-K Renormalization

**Setup:** After selecting the top-2 experts, the script renormalizes their scores: `score_sum = sum(s.data for s in selected_scores)` / `norm_scores = [s / score_sum for s in selected_scores]` (lines 356-360). `N_EXPERTS = 4`, `TOP_K = 2`.

**Question:** Suppose the router outputs probabilities `[0.7, 0.2, 0.07, 0.03]` for the 4 experts. Experts 0 and 1 are selected (top-2). After renormalization, what are their weights in the weighted combination? Without renormalization, would the expert outputs still sum to a reasonable contribution?

<details>
<summary>Reveal Answer</summary>

**Answer:** After renormalization: expert 0 weight = `0.7 / (0.7 + 0.2) = 0.778`, expert 1 weight = `0.2 / (0.7 + 0.2) = 0.222`. Without renormalization, the combined output would be `0.7 * out_0 + 0.2 * out_1 = 0.9 * (weighted average)` — the combined signal would have magnitude ~0.9 of a single expert rather than ~1.0.

**Why:** The router softmax ensures all 4 expert probabilities sum to 1.0. When you discard experts 2 and 3 (0.07 + 0.03 = 0.1 total probability), the retained probabilities only sum to 0.9. Without renormalization, the combined output has 10% less magnitude than a fully-utilized expert, and this magnitude varies per token based on how much probability mass was discarded. Over many layers and training steps, this creates inconsistent scaling that impedes learning. Renormalization ensures the combined output is always a proper convex combination (weights sum to 1), giving consistent output magnitude regardless of the specific top-K probability distribution.

**Script reference:** `02-alignment/micromoe.py`, lines 347-360 (top-K selection and renormalization), lines 351-360 (renormalization logic), lines 370-383 (weighted expert combination)

</details>

---

### Challenge 3: Expert Gradient Bridge

**Setup:** Expert MLPs use plain floats (not autograd `Value` objects), while the router uses `Value` objects. The "gradient bridge" is created by wrapping expert outputs as `Value` objects (lines 381-383): `expert_val = Value(expert_out[j])` and `combined[j] = combined[j] + norm_scores[k_idx] * expert_val`.

**Question:** After `total_loss.backward()` (line 545), do the expert weight matrices `w1` and `w2` get gradient updates? How do expert gradients actually get computed and applied?

<details>
<summary>Reveal Answer</summary>

**Answer:** No, `w1` and `w2` are plain floats — `total_loss.backward()` only updates `Value` objects. Expert weights are updated separately via `expert_backward_float` (line 629), called in a second loop over the sequence (lines 570-631).

**Why:** The `Value` autograd graph only connects through autograd nodes. When expert outputs are wrapped as `Value(expert_out[j])` (line 382), these `Value` objects are leaf nodes in the graph — they have no children linking back to the plain-float expert weights. After `total_loss.backward()`, these leaf `Value` wrappers accumulate `.grad` values representing `d(loss)/d(expert_output[j])`. But the expert weights themselves (`w1`, `w2`) are plain Python floats outside the autograd graph entirely. The second loop (lines 570-631) re-runs the expert forward passes, uses the analytically-derived cross-entropy gradient (lines 615-616), chains it through the LM head (lines 619-624) and the routing scores (lines 626-628), then calls `expert_backward_float` to compute manual chain-rule updates. This hybrid approach is explained in the comment at lines 557-565.

**Script reference:** `02-alignment/micromoe.py`, lines 157-160 (implementation note), lines 374-387 (gradient bridge wrapping), lines 544-555 (autograd backward for router/embeddings), lines 557-631 (separate expert update loop)

</details>

---

### Challenge 4: Why the f_i × P_i Product?

**Setup:** The auxiliary loss formula (line 408) is `L_aux = N_EXPERTS * sum_i(f_i * P_i)` where `f_i` is the fraction of tokens assigned to expert i (binary, computed from hard top-K routing) and `P_i` is the average router probability for expert i (soft, differentiable). The comment on lines 411-415 explains this product.

**Question:** Why use the product `f_i * P_i` rather than simply penalizing variance of `f_i` (e.g., `sum_i (f_i - 1/N)^2`)? Why couldn't the auxiliary loss just use `P_i` alone?

<details>
<summary>Reveal Answer</summary>

**Answer:** `f_i` alone is not differentiable (it comes from a hard argmax via top-K selection, not softmax). `P_i` alone would penalize high-probability experts even when they're necessary. The product `f_i * P_i` creates a differentiable surrogate for the hard routing imbalance.

**Why:** The hard top-K selection (line 347-349) uses `scored.sort()` which is not differentiable — gradients cannot flow through it. `f_i` is a count of how many tokens were hardrouted to expert i, which has zero gradient everywhere. `P_i` (average soft router probability) IS differentiable through the softmax. The product `f_i * P_i` works because: if expert i is overloaded (high `f_i`), the loss increases proportionally to `P_i`, giving the router a differentiable gradient signal to reduce `P_i` for that expert. Penalizing variance of `f_i` directly would require differentiating through the hard routing decision, which is impossible. Penalizing `P_i` alone would push all router probabilities to be uniform, even if the hard routing is already balanced. The product captures the actual imbalance (via `f_i`) while providing the gradient path (via `P_i`).

**Script reference:** `02-alignment/micromoe.py`, lines 392-430 (`compute_aux_loss` with detailed explanation), lines 344-349 (non-differentiable top-K), lines 339-340 (differentiable router_probs), lines 409-415 (why the product formula works)

</details>

