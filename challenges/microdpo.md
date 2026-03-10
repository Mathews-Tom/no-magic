# MicroDPO Challenges

Test your understanding of Direct Preference Optimization by predicting what happens in these scenarios. Try to work out the answer before revealing it.

---

### Challenge 1: Beta Equals Zero

**Setup:** The DPO loss (line 442) computes `delta = beta * (log_ratio_chosen - log_ratio_rejected)` (line 487). The loss is `-log(sigmoid(delta))`, implemented as `log(1 + exp(-delta))` for numerical stability (lines 494-500). The default `DPO_BETA = 0.1` (line 50).

**Question:** If you set `beta = 0`, what does `delta` become? What is the resulting loss value? What happens to the gradients flowing back through the policy model?

<details>
<summary>Reveal Answer</summary>

**Answer:** `delta = 0` for every preference pair. The loss becomes `log(1 + exp(0)) = log(2) = 0.693` for every pair, regardless of model behavior. Gradients through the policy are zero because the loss is a constant with respect to model parameters.

**Why:** The `beta` multiplier on line 487 scales the entire preference signal. At `beta = 0`, the implicit reward difference is always zero: the model cannot distinguish preferred from rejected completions. The loss `log(2)` is the maximum-entropy baseline -- equivalent to a random coin flip between preferred and rejected. Since the loss doesn't depend on any policy log-probabilities (the `beta * ...` term zeroes them out), no gradient flows back through `log_ratio_chosen` or `log_ratio_rejected` to update the model weights. The policy stays frozen at the reference model. This matches the intuition from line 51: "Low beta (0.01) barely moves the policy from the reference."

**Script reference:** `02-alignment/microdpo.py`, lines 50-55 (beta definition and comment), 487 (delta computation), 494-500 (loss computation)

</details>

---

### Challenge 2: Identical Preferred and Rejected Completions

**Setup:** The DPO loss computes log-probability ratios for both the chosen and rejected sequences (lines 472-482). The preference signal depends on `log_ratio_chosen - log_ratio_rejected` (line 487).

**Question:** If the chosen and rejected completions are the exact same token sequence, what happens to `delta`? What does the loss become?

<details>
<summary>Reveal Answer</summary>

**Answer:** `delta = 0`, and the loss is `log(2) = 0.693` -- identical to the `beta = 0` case. No learning occurs.

**Why:** If `chosen_tokens == rejected_tokens`, then `log_pi_chosen == log_pi_rejected` (same sequence through the same model) and `log_ref_chosen == log_ref_rejected` (same sequence through the same reference). Therefore `log_ratio_chosen == log_ratio_rejected`, their difference is zero, `delta = beta * 0 = 0`, and the loss is `log(1 + exp(0)) = log(2)`. The model receives no signal about which direction to move. This makes intuitive sense: if the "preferred" and "rejected" examples are identical, there is no preference to learn from. In production DPO, this is a data quality issue -- identical pairs are filtered out during preprocessing.

**Script reference:** `02-alignment/microdpo.py`, lines 471-487 (log-probability computation and delta), 395-414 (`sequence_log_prob_policy` showing both sequences go through the same computation)

</details>

---

### Challenge 3: Reference Model Already Prefers the Chosen Response

**Setup:** The log-ratio `log(pi/pi_ref)` measures divergence from the reference (line 480-482). The DPO loss pushes this ratio up for chosen sequences and down for rejected ones.

**Question:** If the reference model already assigns 10x higher probability to the chosen completion than the rejected one (i.e., `log_ref_chosen - log_ref_rejected = log(10) = 2.3`), does DPO still learn anything? How does the initial loss compare to a case where the reference assigns equal probability to both?

<details>
<summary>Reveal Answer</summary>

**Answer:** DPO still learns -- the loss starts lower (the policy inherits the reference's preference), but gradients push the policy to amplify this preference further. With equal reference probabilities, the initial loss is higher, meaning there is more room and stronger gradient signal to differentiate the completions.

**Why:** At initialization, the policy equals the reference, so `log_ratio_chosen = log_ratio_rejected = 0`. The initial `delta = 0` and loss = `log(2)` regardless of the reference's preferences -- the reference probabilities cancel out in the log-ratio subtraction on lines 481-482. But after the first gradient step, the policy begins diverging. If the reference already "agrees" with the preference (assigns higher prob to chosen), the policy's task is easier: it doesn't need to fight against a reference that prefers the rejected completion. In contrast, if the reference prefers the rejected completion, the policy must develop a large `log(pi/pi_ref)` gap to overcome this, which the `beta` parameter regulates. This is the KL-regularization effect: beta penalizes divergence from the reference.

**Script reference:** `02-alignment/microdpo.py`, lines 460-466 (log-ratio interpretation), 480-487 (delta computation), 199-211 (`snapshot_weights` creating the frozen reference)

</details>

---

### Challenge 4: Very Large Beta

**Setup:** `DPO_BETA` is the "inverse temperature of the implicit reward model" (line 55). The default is 0.1. The loss is `log(1 + exp(-beta * (log_ratio_chosen - log_ratio_rejected)))`.

**Question:** If you set `DPO_BETA = 100`, what happens to the loss landscape? What behavior would you expect from the trained model?

<details>
<summary>Reveal Answer</summary>

**Answer:** The loss becomes extremely sensitive to small differences in log-ratios. Even a tiny preference for the rejected completion produces a massive loss. The model aggressively reshapes its distribution toward chosen completions, likely collapsing to only produce sequences seen in the chosen set (mode collapse on preferred data).

**Why:** With `beta = 100`, the `delta` term amplifies log-ratio differences by 100x. A log-ratio gap of 0.01 produces `delta = 1.0` instead of `0.001`. The sigmoid curve becomes razor-sharp around zero: sequences are either strongly preferred (low loss) or strongly rejected (high loss), with no middle ground. The gradient signal is intense for any "wrong" prediction, driving the policy far from the reference. The KL regularization that beta provides (keeping the policy close to the reference) breaks down at extreme values -- the model is allowed (forced, even) to diverge arbitrarily far. As the comment on line 53 states: "high beta (1.0) aggressively reshapes the distribution toward preferred completions but risks mode collapse." At `beta = 100`, mode collapse is almost certain.

**Script reference:** `02-alignment/microdpo.py`, lines 50-55 (beta definition and mode collapse warning), 487 (beta scaling in delta), 489-500 (loss computation and stability)

</details>

---

### Challenge 5: The Numerical Stability Guard

**Setup:** The DPO loss uses a stability check on line 496: `if neg_delta.data > 20.0`, it uses `neg_delta` directly instead of `log(1 + exp(neg_delta))`.

**Question:** Why is the threshold 20? What would happen without this guard when `neg_delta = 100`? Does this approximation affect gradients?

<details>
<summary>Reveal Answer</summary>

**Answer:** `exp(100)` is approximately `2.69e43`, and `log(1 + 2.69e43) = 100.0` to float64 precision. The guard avoids computing `exp(100)` (which could overflow to `inf` for values above ~709 in float64) by using the identity `log(1 + exp(z)) ~ z` for large `z`. The gradients are preserved: `d/dz log(1 + exp(z)) = sigmoid(z)`, which approaches 1 for large `z`, and the fallback branch `loss = neg_delta` has gradient 1 with respect to `neg_delta` -- the same limit.

**Why:** For `z > 20`, `exp(z) > 4.85e8`, so `log(1 + exp(z))` and `z` differ by less than `exp(-20) ~ 2e-9`, well below float64 precision. The gradient `sigmoid(z) = 1/(1+exp(-z))` is `1 - exp(-z)` for large `z`, which rounds to 1.0. The autograd graph through `neg_delta` (which is `-delta`, which is `-beta * (log_ratio_chosen - log_ratio_rejected)`) correctly propagates this unit gradient back through the policy log-probabilities. Without the guard, `exp(709)` overflows to `inf`, `log(inf) = inf`, and the loss becomes `inf` -- backpropagation produces `nan` gradients, destroying training.

**Script reference:** `02-alignment/microdpo.py`, lines 489-500 (stability guard with comment explaining the logsigmoid identity)

</details>
