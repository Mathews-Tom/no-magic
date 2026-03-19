# New Script Checklist

Every item here must be completed before merging a new algorithm script. The 04-agents tier was added without updating any supporting systems — this document exists to prevent that.

---

## Before Writing

- [ ] Open a GitHub issue describing the algorithm, what it teaches, and why it belongs here
- [ ] Confirm the algorithm is not already implemented (search `01-foundations/`, `02-alignment/`, `03-systems/`, `04-agents/`)
- [ ] Confirm the algorithm can train and infer on CPU in under 10 minutes
- [ ] Confirm a dataset exists that fits the 5MB auto-download constraint
- [ ] Confirm the script can be a single `.py` file with zero external imports
- [ ] Identify the correct tier and place it in `docs/implementation.md` under the appropriate phase

---

## Script Implementation

- [ ] Single `.py` file — no local imports, no companion files, no `utils.py`
- [ ] Zero external dependencies — only stdlib: `os`, `math`, `random`, `json`, `struct`, `urllib`, `collections`, `itertools`, `functools`, `string`, `hashlib`, `time`
- [ ] `random.seed(42)` is the first executable line after imports
- [ ] Dataset auto-downloads via `urllib` on first run, cached locally, max 5MB
- [ ] `python <tier>/micro<name>.py` with zero arguments runs the full train + inference loop
- [ ] Section ordering: `imports → constants/hyperparameters → data loading → model definition → training loop → inference`
- [ ] Section headers use `# === SECTION NAME ===` format
- [ ] File thesis: one-sentence docstring at the very top stating what the script proves
- [ ] "Why" comments throughout — reasoning, not narration of what the code does
- [ ] Math-to-code mappings: equations shown with explicit variable correspondence
- [ ] Intuition comments: why the technique works, not just how
- [ ] Signpost comments: flag simplifying choices and note what production systems do differently
- [ ] No obvious comments — every comment adds information the code doesn't convey
- [ ] Comment density ~30–40%
- [ ] Descriptive variable names: `learning_rate` not `lr`, `hidden_dim` not `hd`
- [ ] Functions named for what they compute (`rmsnorm`, `softmax`, `linear`)
- [ ] No classes unless the algorithm requires them (e.g., `Value` for autograd)
- [ ] 4-space indentation, 100-character max line length
- [ ] No global mutable state

---

## Testing & Verification

- [ ] Run the script end-to-end: `python <tier>/micro<name>.py`
- [ ] Confirm runtime is under 10 minutes on a laptop CPU (M-series Mac target: under 7 minutes)
- [ ] Run `python scripts/verify.py <tier>/micro<name>.py` and confirm it passes
- [ ] Run twice and confirm `random.seed(42)` produces identical output both times
- [ ] Confirm the script trains and then performs inference (not forward-pass only, unless it is a documented comparison script)

---

## Supporting Artifacts (ALL required)

### Tier README

- [ ] Add a row to `<tier>/README.md` table with: algorithm name, key concept, measured runtime
- [ ] Remove the algorithm from any "Future Candidates" section if it was listed there

### Main README (`README.md`)

- [ ] Increment the algorithm count badge: `algorithms-NN-orange` (line 7)
- [ ] Increment the script count in the correct tier `<summary>` tag (e.g., `"5 scripts" → "6 scripts"`, line ~207)
- [ ] Add a `<td>` entry in the correct tier `<table>` block following the existing format:
  ```html
  <td align="center"><a href="<tier>/micro<name>.py"><b>Algorithm Name</b></a><br/>
  <img src="videos/previews/micro<name>.gif" width="280"/><br/>
  <sub>One-line description</sub></td>
  ```
- [ ] Update the tier description in the "What You'll Find Here" section if the tier's scope changed
- [ ] Update the EPUB script count if it is mentioned

### Mermaid Dependency Graph (in `README.md`)

- [ ] Add node `micro<name>` to the correct subgraph
- [ ] Add prerequisite edges: solid arrows (`-->`) for strong dependencies, dashed arrows (`-.->`) for conceptual dependencies
- [ ] Add `micro<name>` to the `class` styling line at the bottom of the graph block

### Benchmark Runner

- [ ] Confirm `scripts/run_benchmarks.py` SECTIONS list includes the script's tier directory
- [ ] Run the benchmark: `python scripts/run_benchmarks.py micro<name>` and confirm it completes

### Manim Visualization

- [ ] Create scene file: `videos/scenes/scene_micro<name>.py`
- [ ] Inherit from `NoMagicScene` and implement `animate()`
- [ ] Add entry to `SCENE_MAP` in `videos/render_all.py`
- [ ] Render full MP4: `python videos/render_all.py micro<name> --full-only`
- [ ] Render GIF preview: `python videos/render_all.py micro<name> --preview-only`
- [ ] Confirm GIF exists at `videos/previews/micro<name>.gif`
- [ ] Confirm the `<img src="videos/previews/micro<name>.gif">` tag in the README table entry resolves

### Learning Path (`LEARNING_PATH.md`)

- [ ] Add the script to the relevant track with a time estimate and "You'll learn" description
- [ ] Add the script to Track 6 (Full Curriculum) if it is not already covered there

### Flashcards

- [ ] Create or append to `resources/flashcards/<tier-name>.csv`
- [ ] Add 5–10 cards covering the algorithm's core concepts
- [ ] Match the tab-separated format: `question\tanswer\ttags` (see `resources/flashcards/agents.csv` for reference)
- [ ] Tags follow the pattern `<tier-name> micro<name>` (e.g., `agents micromcts`)
- [ ] Update the flashcard count in `README.md` if it is mentioned

### Challenges

- [ ] Create `challenges/<name>.md` with 3–5 "predict the behavior" questions
- [ ] Each question includes: setup with explicit line references, the question, and a `<details>` block with answer and explanation
- [ ] Answer block includes: the answer, the "why", and a "Script reference" citing file and line numbers
- [ ] Add the script to the table in `challenges/README.md`
- [ ] Update the challenge count in `README.md` if it is mentioned

### Quick Start Path

- [ ] If the script is a conceptual milestone (first exposure to a major idea), consider adding it to the Quick Start Path in `README.md`

---

## Commit Structure

- [ ] Script commit: `feat: add micro<name>.py — <one-line description>`
- [ ] Supporting artifacts commit: `docs: add supporting artifacts for micro<name>`
- [ ] Visualization commit: `feat: add Manim scene and GIF preview for micro<name>`
- [ ] Each commit covers one logical unit — do not bundle script + artifacts + visualization into one commit
- [ ] No AI attribution in commit messages (`Co-Authored-By`, "Generated with Claude Code", etc.)

---

## Final Verification

- [ ] `python scripts/verify.py` passes for the full suite (not just the new script)
- [ ] `git diff main -- '*.py'` shows changes only in the new script file
- [ ] Algorithm count badge matches `ls 01-foundations/*.py 02-alignment/*.py 03-systems/*.py 04-agents/*.py | wc -l`
- [ ] Tier script count in the README `<summary>` tag matches `ls <tier>/*.py | wc -l`
- [ ] Flashcard count in README matches `wc -l resources/flashcards/*.csv` (subtract header rows)
- [ ] Challenge count in README matches `ls challenges/*.md | grep -v README | wc -l`
- [ ] All GIF `<img>` tags in the README table resolve to files that exist on disk
