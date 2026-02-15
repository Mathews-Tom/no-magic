## Algorithm

**Name:** <!-- e.g., Mixture of Experts -->  
**Tier:** <!-- 01-foundations / 02-alignment / 03-systems -->  
**File:** <!-- e.g., 03-systems/micromoe.py -->

**One-sentence summary:**

<!-- What does this script prove? This should match the file's thesis docstring. -->

## Change Type

<!-- Check one. -->

- [ ] New script
- [ ] Improvement to existing script (better comments, bug fix, readability)
- [ ] Cross-cutting fix (describe scope below)

## Dataset

**Source:** <!-- URL the script downloads from -->
**Size:** <!-- Must be under 5MB -->
**Fetch method:** `urllib` auto-download with local caching

## Metrics

| Metric          | Value                                  |
| --------------- | -------------------------------------- |
| Total lines     | <!-- e.g., 420 -->                     |
| Comment density | <!-- e.g., ~35% -->                    |
| Runtime         | <!-- e.g., 3m12s on M2 MacBook Air --> |
| CPU model       | <!-- e.g., Apple M2 -->                |

## Sample Output

<!-- Paste a few lines of training progress and inference results. -->

```

```

## Non-Negotiable Constraints

<!-- Every box must be checked or the PR will be closed without review. -->

- [ ] Single `.py` file — no local imports, no companion files
- [ ] Zero external dependencies — Python stdlib only (`os`, `math`, `random`, `json`, `struct`, `urllib`, `collections`, `itertools`, `functools`, `string`, `hashlib`, `time`)
- [ ] `python script.py` runs with zero arguments and exits cleanly
- [ ] `random.seed(42)` at top before any randomness
- [ ] Completes in under 7 minutes on M-series Mac (10 minutes on 2019 Intel i5)
- [ ] Dataset auto-downloads via `urllib` on first run, cached locally, under 5MB
- [ ] No `requirements.txt`, `pyproject.toml`, or build tooling added

## Commenting Standard

<!-- The most common rejection reason. Verify each item. -->

- [ ] File opens with a one-sentence thesis docstring
- [ ] Section headers (`# === SECTION NAME ===`) separate major phases
- [ ] Every non-obvious block has a "why" comment
- [ ] Key equations have math-to-code mapping comments (variable ↔ symbol)
- [ ] At least one intuition comment per core algorithmic concept
- [ ] Simplifying choices flagged with signpost comments (what production does differently)
- [ ] No obvious or redundant comments — every comment earns its place
- [ ] Comment density approximately 30–40%

## Autograd & Numerical Stability

<!-- Check all that apply. Skip this section if the script does not use scalar autograd. -->

- [ ] N/A — script does not use scalar autograd
- [ ] `Value` class implements the canonical interface from `docs/autograd-interface.md`
- [ ] Autograd callout block present after `Value` class
- [ ] Stable softmax: `exp(x - max(x))` pattern with explanatory comment
- [ ] Clipped log-probabilities: `max(p, 1e-10)` before `log()` with comment
- [ ] Adam epsilon: `1e-8` in denominator with comment
- [ ] Test vectors pass (from `docs/autograd-interface.md`)

## Readability

- [ ] Passes the "one sitting" test — a motivated engineer can read top-to-bottom without external references
- [ ] Variable names are descriptive (`learning_rate` not `lr`, `hidden_dim` not `hd`)
- [ ] Functions named for what they compute (`softmax`, `rmsnorm`, `linear`)
- [ ] Consistent section ordering: imports → constants → data loading → model → training → inference
- [ ] No unnecessary complexity or cleverness — explicit loops over dense comprehensions when clearer

## Logistics

- [ ] File placed in the correct tier directory
- [ ] No changes to other scripts (unless fixing a cross-cutting bug — explain below)
- [ ] No extra files (no per-script READMEs, notebooks, or test files)
- [ ] Attribution comments for referenced papers or implementations (immediately after thesis docstring)
- [ ] Fresh-directory test passed — deleted cached data, re-ran, auto-download works

## Additional Context

<!-- Anything reviewers should know: design decisions, deviations from the spec in implementation.md, open questions. -->
