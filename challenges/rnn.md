# MicroRNN Challenges

Test your understanding of vanilla RNNs and GRUs by predicting what happens in these scenarios. Try to work out the answer before revealing it.

---

### Challenge 1: The Vanishing Gradient Measurement

**Setup:** The gradient norm measurement section (lines 457-516) computes loss only at the FINAL timestep after processing `seq_len` tokens: `loss = -safe_log(probs[target])` (line 487). It then calls `loss.backward()` and measures `||dL/dh_t||` for each earlier timestep.

**Question:** Why does computing loss only at the last timestep reveal vanishing gradients more dramatically than the standard training loop (which averages loss across all positions on lines 435-439)?

<details>
<summary>Reveal Answer</summary>

**Answer:** Computing loss only at the last position forces the gradient to traverse the full sequence length before reaching early timesteps. In the training loop, position 0's loss creates a short gradient path (just 1 timestep back), while position 15's loss creates a long path. Averaging these mixes short and long paths, hiding the exponential decay. Single-endpoint loss isolates the worst-case gradient path.

**Why:** Gradient backpropagation through time (BPTT) chains partial derivatives: `dL/dh_0 = (dL/dh_T) * (dh_T/dh_{T-1}) * ... * (dh_1/dh_0)`. For vanilla RNN, each factor is the Jacobian of `tanh(W_hh @ h + ...)` with respect to `h`, which is approximately `W_hh`. The product of T Jacobians makes the gradient magnitude approximately `||W_hh||^T`. If the spectral radius of `W_hh` < 1, this product decays exponentially as T increases. The training loop's position-averaged loss hides this because the loss at each position t contributes a gradient that only needs to backpropagate t steps, not T. The comment on lines 482-485 explains this explicitly.

**Script reference:** `01-foundations/micrornn.py`, lines 435-439 (training loss averaging), lines 482-490 (single-endpoint measurement), lines 493-516 (gradient norm collection and ratio)

</details>

---

### Challenge 2: When z_t ≈ 0 in the GRU

**Setup:** The GRU update gate `z_t = sigmoid(W_xz @ x_t + W_hz @ h_{t-1})` (line 343). The interpolation rule is `h_t = (1 - z_t) * h_{t-1} + z_t * h_candidate` (line 363). The comment on lines 329-332 says "when z_t ≈ 0, h_t = h_{t-1}."

**Question:** If the update gate output is exactly 0 for every timestep in a sequence (z_t = 0 for all t), what is the gradient `dh_T/dh_0`? Why is this the key insight behind gating?

<details>
<summary>Reveal Answer</summary>

**Answer:** `dh_T/dh_0 = 1` (identity). The gradient flows without decay across all timesteps.

**Why:** When z_t = 0, `h_t = h_{t-1}` exactly (no update). Therefore `dh_t/dh_{t-1} = 1`. Chaining T such derivatives: `dh_T/dh_0 = 1^T = 1`. This is the "gradient highway" — the identity connection completely bypasses weight matrices. The vanilla RNN's recurrence is `h_t = tanh(W_hh @ h_{t-1} + ...)`, so `dh_t/dh_{t-1}` always involves `W_hh`, causing exponential decay. The GRU's interpolation structure creates a path where the gradient can bypass `W_hh` entirely when the update gate saturates near zero. In practice, the gate learns to close (z ≈ 0) during "holding" timesteps and open (z ≈ 1) during "updating" timesteps, selectively propagating gradient only when needed.

**Script reference:** `01-foundations/micrornn.py`, lines 341-364 (GRU forward pass), lines 358-364 (interpolation with gradient highway comment), lines 621-627 (explanation in comparison output)

</details>

---

### Challenge 3: The Gradient Norm Ratio Direction

**Setup:** The gradient norm ratio computed at line 512 is `gradient_norms[0] / gradient_norms[-1]`. The comment on line 516 says `< 0.01 = severe vanishing, > 0.1 = gradient highway active`.

**Question:** Why is the ratio computed as `first / last` (early timestep norm divided by late timestep norm) rather than `last / first`? A ratio less than 1 means what?

<details>
<summary>Reveal Answer</summary>

**Answer:** The ratio is `first/last` because `gradient_norms[0]` is the norm at timestep 0 (furthest from the loss, computed last in BPTT), and `gradient_norms[-1]` is the norm at the final timestep (closest to the loss, computed first in BPTT). A ratio less than 1 means gradients are smaller at earlier timesteps — the gradient has decayed traveling backwards through the sequence.

**Why:** Loss is computed at the final timestep. Backward pass flows from last to first. `gradient_norms[-1]` (last hidden state, first computed in backward) has the largest gradient because it's one step from the loss. `gradient_norms[0]` (first hidden state, last computed in backward) has the smallest gradient because it's T steps from the loss. The ratio `norms[0] / norms[-1]` measures how much gradient is left at the beginning relative to the end. A ratio near 0 means almost no gradient signal reaches the early timesteps — the model cannot learn long-range dependencies. For vanilla RNN this ratio is often < 0.01; for GRU it is typically > 0.1 due to gradient highways.

**Script reference:** `01-foundations/micrornn.py`, lines 496-516 (gradient norm computation and ratio), lines 511-516 (ratio interpretation), lines 612-616 (comparison table displaying ratio)

</details>

---

### Challenge 4: Parameter Count Comparison

**Setup:** Vanilla RNN parameters: `W_xh` (N_HIDDEN × VOCAB_SIZE), `W_hh` (N_HIDDEN × N_HIDDEN), `b_h` (N_HIDDEN), `W_hy` (VOCAB_SIZE × N_HIDDEN), `b_y` (VOCAB_SIZE). GRU adds `W_xz`, `W_hz`, `W_xr`, `W_hr` for the gates (lines 215-245). `N_HIDDEN = 32`, `VOCAB_SIZE = len(unique_chars) + 1` (approximately 28).

**Question:** Does the GRU have roughly 2x or 3x the parameters of vanilla RNN? The comment on line 228 says "doubling the parameter count vs vanilla RNN." Is that accurate?

<details>
<summary>Reveal Answer</summary>

**Answer:** The GRU has approximately 3x the parameters of vanilla RNN (counting only hidden-to-hidden and input-to-hidden matrices), not 2x. The comment is inaccurate. Both models share the same output projection (W_hy, b_y).

**Why:** Vanilla RNN has 3 weight matrices for the hidden layer: W_xh, W_hh, and (treating b_h as a matrix) = 3 parameter tensors. GRU has W_xz, W_hz (update gate), W_xr, W_hr (reset gate), W_xh, W_hh (candidate) = 6 parameter tensors for the hidden computation — exactly 3x. Both models share the same W_hy and b_y output projection. However, the comment says "doubling" which refers to the W_hh-equivalent recurrent weight count (1 in vanilla vs 2 in GRU for the gates), not the total count. The actual multiplier depends on which parameters you count. At N_HIDDEN=32, VOCAB_SIZE≈28: vanilla RNN has about 960+1024+32 ≈ 2016 hidden params; GRU has about 5×(960+1024) ≈ 6016 hidden params (≈3x).

**Script reference:** `01-foundations/micrornn.py`, lines 195-212 (vanilla RNN params), lines 215-245 (GRU params), lines 228-229 (inaccurate "doubling" comment), lines 405-406 (parameter count printing)

</details>

