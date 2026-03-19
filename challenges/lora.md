# MicroLoRA Challenges

Test your understanding of Low-Rank Adaptation by predicting what happens in these scenarios. Try to work out the answer before revealing it.

---

### Challenge 1: The B-Matrix Zero Initialization

**Setup:** `make_lora_B` (line 172) initializes LoRA B to all zeros. `make_lora_A` (line 164) initializes A with small random noise `~ N(0, 0.02)`. The adapted output is `base_out + A @ (B @ x)` in `lora_linear` (line 237).

**Question:** At the start of LoRA adaptation (step 0), what does the LoRA adapter contribute to the output? If B were initialized to random noise instead of zeros, what would break?

<details>
<summary>Reveal Answer</summary>

**Answer:** At step 0, the LoRA adapter contributes exactly zero to the output (B is all zeros, so `B @ x = 0`, so `A @ 0 = 0`). If B were initialized randomly, the adapter would immediately perturb the base model output, destroying the pretrained solution before any task-specific signal arrives.

**Why:** The zero-B initialization ensures the model starts LoRA adaptation from the pretrained state. The comment on lines 173-179 explains this precisely: "At init: A @ B = A @ 0 = 0, so the adapted model is identical to the base model." This is critical because the base model has already been trained to solve the A-M names task. If B were random, the initial forward pass would produce garbage predictions, and the loss at step 0 would be high and uninformative — the adaptation would be training from a degraded starting point rather than fine-tuning from a strong one. The random A initialization provides diverse gradient directions once B starts learning (non-zero gradients flow through `dL/dA = (dL/d_lora_out) @ lora_mid.T`), but B's zero init keeps the starting point clean.

**Script reference:** `02-alignment/microlora.py`, lines 164-179 (make_lora_A and make_lora_B initialization with comments), lines 237-264 (lora_linear showing A @ B @ x computation), lines 174-179 (why zero-B comment)

</details>

---

### Challenge 2: Gradient Zeroing for Frozen Weights

**Setup:** During LoRA adaptation (lines 567-573), backward is called on the full computation graph, which includes both frozen base model parameters and LoRA adapter parameters. Then base model gradients are explicitly zeroed: `for p in base_param_list: p.grad = 0.0` (line 572-573).

**Question:** Why is it necessary to zero base model gradients after backward, rather than simply never computing them? Could you skip the backward pass through frozen weights entirely?

<details>
<summary>Reveal Answer</summary>

**Answer:** The gradient computation cannot be selectively stopped at the LoRA boundary — the Value autograd graph is a single connected graph. `loss.backward()` propagates through the entire graph including frozen weights, so their `.grad` fields get populated. Zeroing them after backward is the only way to prevent their update.

**Why:** The LoRA forward pass (line 560) calls `gpt_forward` with `lora=lora_adapters`. Inside, `lora_linear` computes `base_out + lora_out`. The `base_out = linear(x, w_frozen)` operation links `w_frozen` values into the computation graph as parents of `base_out`. When `loss.backward()` runs, it traverses the entire graph and sets `.grad` on every `Value` node that contributed to the loss, including `w_frozen` entries. There's no mechanism to stop traversal at the `w_frozen` boundary. The explicit zeroing at lines 572-573 discards these gradients before the optimizer update, ensuring only LoRA parameters actually change. The comment on lines 568-573 explains this as "the core LoRA mechanism."

**Script reference:** `02-alignment/microlora.py`, lines 558-573 (LoRA training loop backward and gradient zeroing), lines 568-573 (freeze comment), lines 259-264 (lora_linear linking frozen weights into the graph)

</details>

---

### Challenge 3: Rank and Parameter Count

**Setup:** `LORA_RANK = 2` (line 42), `N_EMBD = 16` (line 35). `init_lora_adapters` creates Q and V adapters per layer. A-matrix is `[N_EMBD, LORA_RANK]` (line 219), B-matrix is `[LORA_RANK, N_EMBD]` (line 220). `N_LAYER = 1` (line 37).

**Question:** How many trainable LoRA parameters are there in total? The base model has ~4,200 parameters (reported at line 498). What percentage does LoRA add? If you tripled `LORA_RANK` to 6, how many LoRA parameters would there be?

<details>
<summary>Reveal Answer</summary>

**Answer:** With rank=2, there are `2 * (16*2 + 2*16) = 2 * (32 + 32) = 128` LoRA parameters (Q adapter + V adapter, A+B each). That is about 3% of the base model. With rank=6: `2 * (16*6 + 6*16) = 2 * (96 + 96) = 384` parameters, roughly 9%.

**Why:** Each LoRA adapter consists of matrix A with shape `[N_EMBD, LORA_RANK]` = 16×2 = 32 values, and matrix B with shape `[LORA_RANK, N_EMBD]` = 2×16 = 32 values. One adapter = 64 parameters. There are 2 adapters (Q and V) × 1 layer = 2 adapters × 64 = 128 parameters total. The script reports these at lines 541-543. With rank r, each adapter has `N_EMBD * r + r * N_EMBD = 2 * r * N_EMBD` parameters. The base model has 4,200+ parameters, so rank=2 adds 128/4200 ≈ 3%. Tripling rank multiplies LoRA parameters by 3x (linear relationship), reaching ~9%.

**Script reference:** `02-alignment/microlora.py`, lines 42 (LORA_RANK), lines 219-224 (adapter shapes), lines 541-543 (parameter count reporting), line 593 (pct calculation)

</details>

---

### Challenge 4: Why Q and V, Not K?

**Setup:** The comment at lines 207-211 explains: "Why Q and V, not K or O? The original LoRA paper found that adapting Q and V projections captures the most task-relevant information per parameter. Intuitively: Q controls 'what to look for' and V controls 'what to extract' — both are highly task-specific."

**Question:** The training split is A-M names (base model) vs N-Z names (LoRA adaptation). The base model is tested on both splits after adaptation (lines 616-624). Why should LoRA adapting Q and V improve N-Z performance without severely degrading A-M performance? What's special about adapting through attention rather than the MLP?

<details>
<summary>Reveal Answer</summary>

**Answer:** The frozen W_frozen preserves A-M knowledge. A @ B adds a small rank-2 perturbation that can steer attention queries and value extraction without overwriting the base capability — the additive structure keeps the base signal intact.

**Why:** The adapted output is `W_frozen @ x + A @ (B @ x)`. The base model's A-M representations are encoded in `W_frozen`. When processing an N-Z name, the LoRA term `A @ (B @ x)` can shift the query to find different attention patterns or weight values differently, without destroying the A-M patterns in `W_frozen`. The rank-2 constraint limits how much the adapter can change — it can only modify the output in a 2D subspace, leaving the other 14 dimensions of the 16D output space exactly as the frozen base produces them. Adapting the MLP directly would overwrite feed-forward transformations that are more entangled with the base knowledge. Attention projections are more modular: changing what to query for (Q) or what value to extract (V) is relatively local and reversible.

**Script reference:** `02-alignment/microlora.py`, lines 204-226 (Q and V adapter rationale), lines 247-264 (lora_linear showing additive structure), lines 614-624 (cross-evaluation showing minimal A-M degradation)

</details>

