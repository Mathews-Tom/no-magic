# MicroReAct Challenges

Test your understanding of the ReAct agent (Reasoning + Acting) by predicting what happens in these scenarios. Try to work out the answer before revealing it.

---

### Challenge 1: Action Masking Before Both Values Retrieved

**Setup:** `get_action_mask` (lines 262-292) returns a mask where invalid actions have value `0` (blocked) and valid actions have value `1`. The mask is applied in `forward` at line 488: `masked_logits[i] -= 1e9 if not action_mask[i] else 0`. `MAX_STEPS = 3` (line 45). The comment on lines 265-275 explains the two-phase structure.

**Question:** At the start of a new episode (no values retrieved yet), which action types are masked out? If the agent could take the `ANSWER` action immediately without retrieving either operand, what would it be forced to output and why would this be problematic?

<details>
<summary>Reveal Answer</summary>

**Answer:** At step 0, the `ANSWER` action and `COMPUTE` action are masked (set to `-1e9` in logits), leaving only `LOOKUP_A` and `LOOKUP_B` as valid choices. If `ANSWER` were permitted immediately, the agent would have to produce an answer with no retrieved operand values — it could only guess based on the question encoding alone, since the actual numbers haven't been fetched from the "database."

**Why:** The ReAct agent solves arithmetic problems by: (1) looking up operand A, (2) looking up operand B, (3) computing A OP B, (4) answering. The masking enforces this dependency structure. If `ANSWER` were unmasked at step 0, policy gradients could learn to skip retrieval whenever it happened to guess correctly by chance — reinforcing a degenerate shortcut that fails on new problems with different values. The `-1e9` logit suppression makes these actions effectively impossible (softmax of `-1e9` ≈ 0), so gradients through masked actions are near zero. The comment on lines 267-270 explicitly states this prevents the "cheat" of answering before gathering information, mirroring how real tool-using agents must gather observations before acting on them.

**Script reference:** `04-agents/microreact.py`, lines 262-292 (get_action_mask), lines 265-275 (two-phase masking comment), lines 488 (mask application in forward), lines 453-496 (forward with masked logits)

</details>

---

### Challenge 2: Reward Shaping and Sparse vs Dense Signal

**Setup:** `compute_reward` (lines 625-662) gives: `+0.15` for correctly looking up an operand (lines 639-648), `+0.10` for a correct compute action (line 651), `+0.60` for a correct final answer (line 656), `-0.05` penalty for wrong lookup/compute (lines 649, 652). `ENTROPY_COEFF = 0.005` (line 43) adds an entropy bonus to encourage exploration.

**Question:** If you removed all the intermediate rewards and only kept the `+0.60` final answer reward, what would the REINFORCE gradient signal look like for a correct episode? Why might training fail or be very slow with this sparse reward structure?

<details>
<summary>Reveal Answer</summary>

**Answer:** With only sparse `+0.60`, a correct episode of 4 steps (LOOKUP_A, LOOKUP_B, COMPUTE, ANSWER) would get total reward `+0.60` at the final step, while all intermediate steps get `0`. REINFORCE assigns this reward as `advantage = R - baseline` to the final step only (or distributes discounted reward backwards). Early in training when correct full episodes are rare (perhaps 5% of episodes), the gradient signal is nearly always zero — 95% of episodes contribute nothing.

**Why:** REINFORCE requires seeing positive outcomes to reinforce correct behaviors. With sparse reward, the agent must accidentally complete a correct 4-step sequence before it receives any gradient signal reinforcing those specific actions. The probability of randomly completing all 4 steps correctly (choosing the right operand lookups, correct compute, correct answer) is very low early in training. The intermediate rewards (+0.15 for correct lookup) act as "stepping stones" — even if the agent gets the final answer wrong, it still receives positive signal for individual correct sub-steps. This dense feedback means the agent can learn the lookup behavior independently of whether it also gets the compute and answer right. The entropy coefficient `ENTROPY_COEFF = 0.005` at line 43 adds `entropy * coeff` to the loss, preventing the policy from collapsing to always picking the same action before it's learned which actions are correct.

**Script reference:** `04-agents/microreact.py`, lines 625-662 (compute_reward with all reward terms), lines 639-660 (intermediate and final rewards), lines 43 (ENTROPY_COEFF), lines 453-496 (policy forward with entropy computation)

</details>

---

### Challenge 3: EMA Baseline and Advantage Variance

**Setup:** The REINFORCE baseline is an exponential moving average: `baseline = BASELINE_DECAY * baseline + (1 - BASELINE_DECAY) * episode_reward` (line 41 shows `BASELINE_DECAY = 0.95`). The advantage for each step is `advantage = episode_reward - baseline` (applied in the policy gradient update). Early in training the baseline starts at `0.0`.

**Question:** In episode 1, the agent gets total reward `+0.25`. In episode 2, the agent gets `+0.70`. What is the baseline value before episode 1, after episode 1, and before episode 2? What advantage does episode 2 receive? Why does this make training more stable than using raw rewards?

<details>
<summary>Reveal Answer</summary>

**Answer:** Before episode 1: `baseline = 0.0`. After episode 1: `baseline = 0.95 * 0.0 + 0.05 * 0.25 = 0.0125`. Before episode 2 (same as after episode 1): `baseline = 0.0125`. Episode 2 advantage: `0.70 - 0.0125 = 0.6875`. Without baseline, episode 2 advantage = `0.70` (the raw reward).

**Why:** The EMA baseline tracks the running average of recent rewards. Once training has progressed and typical rewards are around +0.40, an episode with reward +0.70 gets advantage `+0.30` (better than average — reinforce these actions), while an episode with reward +0.10 gets advantage `-0.30` (worse than average — discourage these actions). Without a baseline, all rewards are positive (given the reward structure with mostly positive terms), so all actions get positively reinforced regardless of quality — identical to the "all rewards equal +1" problem described in the PPO challenge. The EMA baseline (rather than per-batch mean) provides a smooth estimate that doesn't require a batch of episodes: it updates after each episode, enabling online learning. The `BASELINE_DECAY = 0.95` means the baseline has an effective memory of about `1/(1-0.95) = 20` recent episodes — responsive to trend changes but not noisy.

**Script reference:** `04-agents/microreact.py`, lines 41 (BASELINE_DECAY), lines 700-715 (baseline update after each episode), lines 720-735 (advantage computation and policy gradient update), lines 697-700 (baseline initialization at 0.0)

</details>

---

### Challenge 4: The Thought/Action/Observe Loop Structure

**Setup:** The agent's state encoder `encode_state` (lines 358-418) builds a fixed-size feature vector from: the question (lines 372-382), last thought (lines 384-390), last action (lines 392-398), last observation (lines 400-408), and step count (line 410). `MAX_STEPS = 3` (line 45). The comment on lines 360-368 explains the Thought/Action/Observe (TAO) loop.

**Question:** The episode terminates when the agent takes the `ANSWER` action or reaches `MAX_STEPS = 3`. If the agent takes 3 non-ANSWER actions (e.g., LOOKUP_A, LOOKUP_B, COMPUTE), it never gets to answer. What reward does it receive and why is `MAX_STEPS` set to 3 instead of, say, 10?

<details>
<summary>Reveal Answer</summary>

**Answer:** If `MAX_STEPS = 3` is reached without an ANSWER action, the episode ends with whatever intermediate rewards were earned (potentially +0.15 + 0.15 + 0.10 = +0.40 if all steps were correct) but without the +0.60 final answer bonus. The total reward is far lower than a successful episode (~1.00), providing strong signal to learn that answering is necessary within the budget.

**Why:** `MAX_STEPS = 3` is tight because the minimal correct strategy requires exactly 4 actions: LOOKUP_A, LOOKUP_B, COMPUTE, ANSWER. With `MAX_STEPS = 3`, the agent must take ANSWER as one of its first 3 steps — but it also needs the other 3 steps for retrieval and computation. This creates tension: the agent must learn to be maximally efficient. If `MAX_STEPS = 10`, the agent could meander (repeat lookups, take redundant steps) and still find time to answer, making the task easier but reducing the pressure to learn an efficient strategy. The `MAX_STEPS = 3` constraint forces the agent to learn the minimal-action strategy. The comment on line 46 notes this tight budget "forces the agent to be efficient." In production ReAct systems (as in the original ReAct paper), similar step budgets prevent agents from infinite looping on tool calls when they get confused.

**Script reference:** `04-agents/microreact.py`, lines 45 (MAX_STEPS), lines 46 (budget comment), lines 358-418 (encode_state building feature vector per step), lines 625-662 (compute_reward showing no terminal penalty, only missed +0.60), lines 665-680 (episode loop with step count check)

</details>
