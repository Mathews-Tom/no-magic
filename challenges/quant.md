# MicroQuant Challenges

Test your understanding of post-training quantization by predicting what happens in these scenarios. Try to work out the answer before revealing it.

---

### Challenge 1: Absmax INT8 With an Outlier

**Setup:** `quantize_absmax_int8` (lines 318-333) computes `scale = max_abs / 127.0` and maps each value to `round(val / scale)`, clipped to `[-127, 127]`. The comment on lines 328-330 explains how a single large value degrades all other values.

**Question:** Suppose a weight matrix has values uniformly distributed in `[-0.5, 0.5]` except for one outlier at `10.0`. What is the absmax scale? What is the quantized representation of a typical value like `0.3`? Compare to quantizing the same matrix without the outlier.

<details>
<summary>Reveal Answer</summary>

**Answer:** With outlier: `scale = 10.0 / 127.0 ≈ 0.0787`. The value `0.3` quantizes to `round(0.3 / 0.0787) = round(3.81) = 4`, dequantizing to `4 * 0.0787 = 0.315` — error of `0.015`. Without outlier: `scale = 0.5 / 127.0 ≈ 0.00394`. The value `0.3` quantizes to `round(0.3 / 0.00394) = round(76.1) = 76`, dequantizing to `76 * 0.00394 = 0.299` — error of `0.001`. The outlier degraded precision on `0.3` by 15x.

**Why:** Absmax forces all 256 integer levels to cover the full range `[-max_abs, +max_abs]`. An outlier at `10.0` stretches this range to `[-10, 10]`, spreading 254 quantization levels across a span 20x wider than necessary for 99.9% of the weights. The typical weight at `0.3` now only has access to the handful of integer levels near `4 * scale ≈ 0.3`, rather than the 76 levels it would use in the outlier-free case. This is the core failure mode of per-tensor absmax quantization on transformer weight matrices, which often have heavy-tailed distributions with occasional large outliers. The comment on lines 323-326 explicitly notes that one large value affects the quantization of every other value in the tensor.

**Script reference:** `03-systems/microquant.py`, lines 318-333 (quantize_absmax_int8), lines 323-326 (outlier sensitivity comment), lines 454-469 (compute_roundtrip_error using max abs error), lines 530-545 (comparison output showing error per scheme)

</details>

---

### Challenge 2: INT4 Asymmetric Range

**Setup:** `quantize_absmax_int4` (lines 336-353) uses `range_val = 8` (line 349) and clips to `[-8, 7]` (line 352). The comment on line 348 notes this is the standard INT4 signed integer range. Compare to INT8 which uses `[-127, 127]` (line 331).

**Question:** INT4 has 16 possible values: why does the range `[-8, 7]` have one more negative value than positive? If you tried to use `[-7, 7]` instead (symmetric, 15 values), what would you lose in practice?

<details>
<summary>Reveal Answer</summary>

**Answer:** Two's complement 4-bit signed integers naturally represent `[-8, 7]` — the bit pattern `1000` represents `-8` and `0111` represents `+7`. Using `[-7, 7]` wastes the `1000` bit pattern (requiring either treating it as `-7` or as undefined), effectively throwing away 1/16 of the representable range. In practice this means using the standard hardware INT4 type without special handling is worth roughly 6% more precision at no cost.

**Why:** In binary two's complement, N-bit signed integers have range `[-(2^(N-1)), 2^(N-1) - 1]`. For N=4: `[-8, 7]`. The asymmetry arises because zero is a positive integer (`0000`), so there's one fewer positive bit pattern than negative. Hardware INT4 arithmetic (as supported on modern accelerators) operates on this range natively. Using `[-7, 7]` would require clipping to a non-standard range, adding overhead and wasting the `1000` bit pattern. The script uses `range_val = 8` on line 349 and clips to `[-range_val, range_val - 1]` on line 352, faithfully implementing the hardware-standard range. This asymmetry also means the quantization is slightly biased: negative values get one more representable point than positive values, which can matter for weights with non-zero mean.

**Script reference:** `03-systems/microquant.py`, lines 336-353 (quantize_absmax_int4), lines 348-352 (range_val=8 and clip to [-8, 7]), lines 340-344 (INT4 comment), lines 530-545 (error comparison INT4 vs INT8)

</details>

---

### Challenge 3: Per-Channel vs Per-Tensor Accuracy

**Setup:** `quantize_per_channel_int8` (lines 385-405) computes a separate scale for each row: `max_abs = max(abs(v) for v in row)` / `scale = max_abs / 127.0` (lines 400-402). `quantize_absmax_int8` uses a single global scale. The comment on lines 387-396 explains why output channels benefit from independent scaling.

**Question:** Suppose a weight matrix has row 0 with values in `[-0.1, 0.1]` and row 1 with values in `[-5.0, 5.0]`. What is the per-tensor scale? What are the per-channel scales? Compute the roundtrip error for a value `0.05` in row 0 under each scheme.

<details>
<summary>Reveal Answer</summary>

**Answer:** Per-tensor scale: `5.0 / 127.0 ≈ 0.03937`. Value `0.05` → `round(0.05 / 0.03937) = round(1.27) = 1` → dequantized `0.03937` → error `|0.05 - 0.03937| = 0.0106`.

Per-channel scale for row 0: `0.1 / 127.0 ≈ 0.000787`. Value `0.05` → `round(0.05 / 0.000787) = round(63.5) = 64` → dequantized `64 * 0.000787 = 0.0504` → error `|0.05 - 0.0504| = 0.0004`.

Per-channel reduces error on row 0 by 26x.

**Why:** Row 0's small values are compressed into just `round(0.1/0.03937) = ±2` integer levels under per-tensor scaling — only 5 distinct quantized values cover the entire row, causing massive quantization noise. Per-channel scaling allocates all 255 levels independently to each row's range. Since rows of weight matrices often correspond to different output neurons with very different activation scales (especially after layer normalization with learned scale parameters), per-channel quantization matches the scale to the actual data distribution. The tradeoff is that each row needs its own stored scale value — `N_rows` scale values instead of 1 — increasing metadata overhead by `N_rows * sizeof(float)` bytes.

**Script reference:** `03-systems/microquant.py`, lines 385-405 (quantize_per_channel_int8 with per-row scales), lines 398-404 (per-row scale computation loop), lines 318-333 (quantize_absmax_int8 for comparison), lines 454-469 (compute_roundtrip_error), lines 387-396 (why per-channel comment)

</details>

---

### Challenge 4: Zero-Point Quantization for Asymmetric Data

**Setup:** `quantize_zeropoint_int8` (lines 356-382) computes `zero_point = round(-min_val / scale)` (line 374) and stores it alongside the scale. The quantization formula is `q = round(val / scale + zero_point)` (line 377), and dequantization is `val = (q - zero_point) * scale` (line 381). The range covers `[0, 255]` (unsigned byte).

**Question:** Suppose a ReLU activation layer produces values in `[0, 4.0]` (all non-negative). What zero-point does `quantize_zeropoint_int8` compute? Why would absmax INT8 (which covers `[-max_abs, max_abs]`) be particularly wasteful for this data?

<details>
<summary>Reveal Answer</summary>

**Answer:** `scale = (4.0 - 0.0) / 255.0 ≈ 0.01569`. `zero_point = round(-0.0 / 0.01569) = round(0) = 0`. The unsigned range `[0, 255]` maps exactly to `[0.0, 4.0]` with zero_point=0. Absmax INT8 would use `scale = 4.0 / 127.0 ≈ 0.03150` and cover `[-4.0, 4.0]`, but half of that range (negative values) is never used by ReLU activations — wasting half the representable range and giving 2x worse precision.

**Why:** Absmax INT8 is designed for symmetric distributions centered near zero (typical for weight matrices after training). Activations after ReLU are strictly non-negative, so the symmetric range `[-max, max]` wastes the 128 negative integer levels on values that never appear. Zero-point quantization (also called affine quantization) maps the data range `[min_val, max_val]` to the full unsigned range `[0, 255]`, using 256 levels instead of 128 to cover the same value range — effectively doubling precision. The `zero_point` parameter shifts the integer grid so that `q=0` corresponds to `val=min_val`, not to `val=0`. This is why quantization-aware training for models with ReLU activations always uses asymmetric/zero-point quantization for activations: it matches the actual data distribution rather than assuming symmetry.

**Script reference:** `03-systems/microquant.py`, lines 356-382 (quantize_zeropoint_int8), lines 367-375 (scale and zero_point computation), lines 377-381 (quantize/dequantize with zero_point), lines 358-366 (motivation comment for asymmetric data)

</details>
