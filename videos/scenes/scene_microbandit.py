"""
Scene: Multi-Armed Bandits
Script: microbandit.py
Description: Exploration vs exploitation — ε-greedy, UCB1, Thompson Sampling
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE, NM_PURPLE
from manim import *


K = 5  # number of arms shown in the visualization
ARM_TRUE_PROBS = [0.2, 0.5, 0.8, 0.35, 0.6]  # hidden reward probabilities for K=5 arms
ARM_COLORS = [NM_BLUE, NM_ORANGE, NM_GREEN, NM_PURPLE, NM_PRIMARY]


def make_arm(index: int, height: float = 2.0, width: float = 0.55) -> VGroup:
    """Create a single arm as a labeled rectangle column."""
    rect = Rectangle(
        width=width, height=height,
        color=ARM_COLORS[index], fill_opacity=0.25, stroke_width=2,
    )
    label = Text(f"A{index + 1}", font_size=16, color=NM_TEXT)
    label.next_to(rect, DOWN, buff=0.15)
    return VGroup(rect, label)


class BanditScene(NoMagicScene):
    title_text = "Multi-Armed Bandits"
    subtitle_text = "Exploration vs exploitation — ε-greedy, UCB1, Thompson Sampling"

    def animate(self):
        # === Step 1: Bandit problem setup ===
        self._show_bandit_problem()

        # === Step 2: Epsilon-Greedy ===
        self._show_epsilon_greedy()

        # === Step 3: UCB1 ===
        self._show_ucb1()

        # === Step 4: Thompson Sampling ===
        self._show_thompson_sampling()

        # === Step 5: Regret comparison ===
        self._show_regret_comparison()

        # === Step 6: Summary ===
        self._show_summary()

    # ------------------------------------------------------------------
    # Step 1: Show K arms with unknown reward probabilities
    # ------------------------------------------------------------------
    def _show_bandit_problem(self):
        # Section header
        header = Text("The Bandit Problem", font_size=30, color=NM_TEXT, weight=BOLD)
        header.to_edge(UP, buff=0.4)
        self.play(Write(header), run_time=0.7)

        # Build K arm rectangles, evenly spaced
        arms = VGroup(*[make_arm(i) for i in range(K)])
        arms.arrange(RIGHT, buff=0.45)
        arms.move_to(ORIGIN + UP * 0.3)
        self.play(
            LaggedStart(*[FadeIn(a, shift=UP * 0.3) for a in arms], lag_ratio=0.12),
            run_time=1.0,
        )

        # Show "?" labels above each arm to signal unknown probabilities
        question_marks = VGroup()
        for arm in arms:
            q = Text("?", font_size=22, color=NM_YELLOW, weight=BOLD)
            q.next_to(arm[0], UP, buff=0.2)
            question_marks.add(q)

        self.play(
            LaggedStart(*[FadeIn(q, shift=DOWN * 0.1) for q in question_marks], lag_ratio=0.1),
            run_time=0.7,
        )

        # Caption below
        caption = Text("K arms, unknown reward probabilities", font_size=18, color=NM_GRID)
        caption.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(caption), run_time=0.5)
        self.wait(0.8)

        # Reveal true probabilities (the agent cannot see these, but we show them for clarity)
        prob_labels = VGroup()
        for i, (arm, prob) in enumerate(zip(arms, ARM_TRUE_PROBS)):
            p_text = Text(f"p={prob}", font_size=14, color=ARM_COLORS[i])
            p_text.move_to(question_marks[i].get_center())
            prob_labels.add(p_text)

        self.play(
            *[Transform(question_marks[i], prob_labels[i]) for i in range(K)],
            run_time=0.8,
        )
        self.wait(0.6)

        # Clean up
        self.play(
            FadeOut(header), FadeOut(arms), FadeOut(question_marks),
            FadeOut(caption), run_time=0.7,
        )

    # ------------------------------------------------------------------
    # Step 2: Epsilon-Greedy
    # ------------------------------------------------------------------
    def _show_epsilon_greedy(self):
        # Strategy label
        strategy_label = Text("ε-Greedy", font_size=30, color=NM_BLUE, weight=BOLD)
        strategy_label.to_edge(UP, buff=0.4)
        self.play(Write(strategy_label), run_time=0.6)

        # Epsilon parameter annotation
        eps_text = MathTex(r"\varepsilon = 0.1", font_size=26, color=NM_YELLOW)
        eps_text.next_to(strategy_label, RIGHT, buff=0.6)
        self.play(FadeIn(eps_text, shift=LEFT * 0.2), run_time=0.5)

        # Show arms as estimate bars (starting flat/zero)
        arm_group = VGroup()
        estimate_rects = []
        arm_labels = []
        for i in range(K):
            # Background track
            track = Rectangle(width=0.5, height=2.5, color=NM_GRID, fill_opacity=0.15, stroke_width=1)
            # Estimate fill — zero height initially, anchored to bottom of track
            fill = Rectangle(width=0.46, height=0.02, color=ARM_COLORS[i], fill_opacity=0.7, stroke_width=0)
            fill.align_to(track, DOWN)
            label = Text(f"A{i + 1}", font_size=14, color=NM_TEXT)
            label.next_to(track, DOWN, buff=0.1)
            col = VGroup(track, fill, label)
            arm_group.add(col)
            estimate_rects.append(fill)
            arm_labels.append(label)

        arm_group.arrange(RIGHT, buff=0.5)
        arm_group.move_to(ORIGIN + DOWN * 0.2)
        self.play(FadeIn(arm_group), run_time=0.5)

        # Simulate a sequence of pulls: pull order showing exploit vs explore
        # "Best" is arm 2 (index 2, p=0.8). Show 8 pulls: mostly arm 2, one random.
        pull_sequence = [2, 2, 0, 2, 2, 4, 2, 2]  # arm indices
        pull_types = ["exploit", "exploit", "explore", "exploit", "exploit", "explore", "exploit", "exploit"]
        running_estimates = [0.0] * K
        pull_counts = [0] * K

        pull_label = Text("", font_size=16, color=NM_TEXT)
        pull_label.to_edge(DOWN, buff=1.0)
        self.add(pull_label)

        for pull_idx, (arm_i, ptype) in enumerate(zip(pull_sequence, pull_types)):
            # Indicator: which arm is being pulled
            indicator_color = NM_GREEN if ptype == "exploit" else NM_ORANGE
            indicator = Text(
                f"{'exploit' if ptype == 'exploit' else 'explore'}: A{arm_i + 1}",
                font_size=16, color=indicator_color,
            )
            indicator.to_edge(DOWN, buff=1.0)

            self.play(
                Indicate(arm_group[arm_i][0], color=indicator_color, scale_factor=1.1),
                FadeIn(indicator),
                run_time=0.35,
            )

            # Update estimate
            reward = ARM_TRUE_PROBS[arm_i]  # use expected reward for clarity
            pull_counts[arm_i] += 1
            running_estimates[arm_i] = (
                (running_estimates[arm_i] * (pull_counts[arm_i] - 1) + reward) / pull_counts[arm_i]
            )

            # Animate the estimate bar growing
            new_height = max(0.02, running_estimates[arm_i] * 2.5)
            new_fill = Rectangle(
                width=0.46, height=new_height,
                color=ARM_COLORS[arm_i], fill_opacity=0.7, stroke_width=0,
            )
            new_fill.align_to(arm_group[arm_i][0], DOWN)

            self.play(
                Transform(estimate_rects[arm_i], new_fill),
                FadeOut(indicator),
                run_time=0.3,
            )

        # Show exploit-dominant pattern caption
        exploit_note = Text(
            "10% explore (random arm)  ·  90% exploit (best estimate)",
            font_size=14, color=NM_BLUE,
        )
        exploit_note.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(exploit_note), run_time=0.5)
        self.wait(0.8)

        self.play(
            FadeOut(strategy_label), FadeOut(eps_text),
            FadeOut(arm_group), FadeOut(exploit_note), run_time=0.7,
        )

    # ------------------------------------------------------------------
    # Step 3: UCB1
    # ------------------------------------------------------------------
    def _show_ucb1(self):
        strategy_label = Text("UCB1", font_size=30, color=NM_GREEN, weight=BOLD)
        strategy_label.to_edge(UP, buff=0.4)
        self.play(Write(strategy_label), run_time=0.6)

        # UCB1 formula
        ucb_formula = MathTex(
            r"\text{UCB}_i = \bar{x}_i + c \sqrt{\frac{\ln N}{n_i}}",
            font_size=28, color=NM_YELLOW,
        )
        ucb_formula.next_to(strategy_label, DOWN, buff=0.3)
        self.play(Write(ucb_formula), run_time=1.0)
        self.wait(0.5)

        # Arm estimates + confidence bounds as error bars
        # Use representative values: A3 is best (p=0.8), A1 barely pulled
        estimates = [0.18, 0.47, 0.79, 0.33, 0.58]
        n_pulls = [3, 8, 15, 4, 10]  # unequal pulls — A1 pulled fewest
        N_total = sum(n_pulls)
        import math
        conf_bounds = [2.0 * math.sqrt(math.log(N_total) / n) for n in n_pulls]

        arm_group = VGroup()
        mean_dots = []
        error_bar_groups = []

        for i in range(K):
            # Estimate dot on a vertical axis
            x_pos = (i - 2) * 1.2  # spread across screen
            y_mean = estimates[i] * 2.5 - 1.5  # map [0,1] → [-1.5, 1.0]
            y_upper = min(y_mean + conf_bounds[i] * 1.0, 1.5)
            y_lower = max(y_mean - conf_bounds[i] * 1.0, -1.8)

            # Track line
            track = Line(
                (x_pos, -1.8, 0), (x_pos, 1.5, 0),
                color=NM_GRID, stroke_width=1,
            )

            # Mean estimate dot
            dot = Dot((x_pos, y_mean, 0), radius=0.1, color=ARM_COLORS[i])

            # Confidence interval bar
            ci_line = Line(
                (x_pos, y_lower, 0), (x_pos, y_upper, 0),
                color=ARM_COLORS[i], stroke_width=3,
            )
            cap_top = Line(
                (x_pos - 0.12, y_upper, 0), (x_pos + 0.12, y_upper, 0),
                color=ARM_COLORS[i], stroke_width=2,
            )
            cap_bot = Line(
                (x_pos - 0.12, y_lower, 0), (x_pos + 0.12, y_lower, 0),
                color=ARM_COLORS[i], stroke_width=2,
            )

            arm_label = Text(f"A{i + 1}", font_size=14, color=NM_TEXT)
            arm_label.move_to((x_pos, -2.1, 0))
            n_label = Text(f"n={n_pulls[i]}", font_size=11, color=NM_GRID)
            n_label.move_to((x_pos, -2.4, 0))

            col = VGroup(track, dot, ci_line, cap_top, cap_bot, arm_label, n_label)
            arm_group.add(col)
            mean_dots.append(dot)
            error_bar_groups.append(VGroup(ci_line, cap_top, cap_bot))

        self.play(
            LaggedStart(*[FadeIn(g) for g in arm_group], lag_ratio=0.1),
            run_time=1.0,
        )

        # Annotation: wide bounds → exploration target
        wide_note = Text("few pulls → wide bound → explore", font_size=14, color=NM_ORANGE)
        wide_note.to_edge(DOWN, buff=0.8)
        arrow_to_a1 = Arrow(
            wide_note.get_top(), arm_group[0][2].get_center() + UP * 0.3,
            buff=0.1, color=NM_ORANGE, stroke_width=2,
        )
        self.play(FadeIn(wide_note), GrowArrow(arrow_to_a1), run_time=0.7)
        self.wait(0.5)

        # Highlight A1 (fewest pulls) as selected by UCB1
        self.play(
            Indicate(arm_group[0][1], color=NM_GREEN, scale_factor=1.4),
            run_time=0.6,
        )
        selected_text = Text("UCB1 selects A1 — highest upper bound", font_size=14, color=NM_GREEN)
        selected_text.next_to(wide_note, UP, buff=0.2)
        self.play(FadeIn(selected_text), run_time=0.4)
        self.wait(0.6)

        # After more pulls, A1 bound narrows
        narrow_note = Text("more pulls → bounds narrow → converge", font_size=14, color=NM_YELLOW)
        narrow_note.to_edge(DOWN, buff=0.8)
        narrow_ci = Line(
            arm_group[0][2].get_start() + UP * 0.5,
            arm_group[0][2].get_end() + DOWN * 0.5,
            color=ARM_COLORS[0], stroke_width=3,
        )
        self.play(
            Transform(wide_note, narrow_note),
            Transform(arm_group[0][2], narrow_ci),
            FadeOut(arrow_to_a1), FadeOut(selected_text),
            run_time=0.8,
        )
        self.wait(0.6)

        self.play(
            FadeOut(strategy_label), FadeOut(ucb_formula),
            FadeOut(arm_group), FadeOut(wide_note), run_time=0.7,
        )

    # ------------------------------------------------------------------
    # Step 4: Thompson Sampling
    # ------------------------------------------------------------------
    def _show_thompson_sampling(self):
        strategy_label = Text("Thompson Sampling", font_size=30, color=NM_PURPLE, weight=BOLD)
        strategy_label.to_edge(UP, buff=0.4)
        self.play(Write(strategy_label), run_time=0.6)

        # Beta prior description
        prior_text = MathTex(
            r"\theta_k \sim \text{Beta}(\alpha_k,\, \beta_k)",
            font_size=24, color=NM_YELLOW,
        )
        prior_text.next_to(strategy_label, DOWN, buff=0.25)
        self.play(Write(prior_text), run_time=0.8)
        self.wait(0.4)

        # Draw simplified Beta distribution curves as arcs for 3 representative arms
        # We use ParametricFunction to sketch Beta-like humps
        axes_group = VGroup()
        curve_group = VGroup()
        sample_dots = VGroup()
        arm_curve_labels = VGroup()

        # Show 3 arms: A1 (few pulls, wide), A3 (many pulls, narrow/high), A5 (medium)
        arm_data = [
            # (arm_idx, alpha, beta, x_offset, color, label)
            (0, 2, 2, -3.5, ARM_COLORS[0], "A1\nα=2,β=2"),
            (2, 12, 3, 0.0, ARM_COLORS[2], "A3\nα=12,β=3"),
            (4, 6, 4, 3.5, ARM_COLORS[4], "A5\nα=6,β=4"),
        ]

        for arm_idx, alpha, beta, x_off, color, lbl in arm_data:
            # Axes
            ax = Axes(
                x_range=[0, 1, 0.5],
                y_range=[0, 4, 2],
                x_length=2.2,
                y_length=1.8,
                axis_config={"color": NM_GRID, "stroke_width": 1},
                tips=False,
            )
            ax.move_to((x_off, -0.4, 0))

            # Beta PDF curve: f(x) ∝ x^(α-1) * (1-x)^(β-1)
            # Normalise by peak for visual clarity
            def make_beta_func(a, b):
                def beta_pdf(x):
                    if x <= 0 or x >= 1:
                        return 0.0
                    val = (x ** (a - 1)) * ((1 - x) ** (b - 1))
                    return val
                return beta_pdf

            raw_fn = make_beta_func(alpha, beta)
            peak = max(raw_fn(t / 100) for t in range(1, 100))
            scale_factor = 3.5 / peak if peak > 0 else 1.0

            curve = ax.plot(
                lambda x, a=alpha, b=beta, s=scale_factor: (
                    s * ((x ** (a - 1)) * ((1 - x) ** (b - 1))) if 0 < x < 1 else 0
                ),
                x_range=[0.01, 0.99],
                color=color,
                stroke_width=2.5,
            )

            label_obj = Text(lbl, font_size=12, color=color)
            label_obj.next_to(ax, DOWN, buff=0.1)

            axes_group.add(ax)
            curve_group.add(curve)
            arm_curve_labels.add(label_obj)

        self.play(
            LaggedStart(*[Create(ax) for ax in axes_group], lag_ratio=0.15),
            run_time=0.8,
        )
        self.play(
            LaggedStart(*[Create(c) for c in curve_group], lag_ratio=0.15),
            LaggedStart(*[FadeIn(l) for l in arm_curve_labels], lag_ratio=0.15),
            run_time=1.0,
        )
        self.wait(0.4)

        # Show sampling: a vertical dash line at a sampled value for each curve
        sample_vals = [0.48, 0.82, 0.64]  # representative samples from the posteriors
        sample_lines = VGroup()
        sample_value_labels = VGroup()

        for i, ((arm_idx, alpha, beta, x_off, color, lbl), sval) in enumerate(
            zip(arm_data, sample_vals)
        ):
            ax = axes_group[i]
            x_screen = ax.c2p(sval, 0)[0]
            y_bot = ax.c2p(0, 0)[1]
            y_top = ax.c2p(0, 3.5)[1]
            dash = DashedLine(
                (x_screen, y_bot, 0), (x_screen, y_top, 0),
                color=NM_YELLOW, stroke_width=1.5, dash_length=0.07,
            )
            val_lbl = Text(f"θ={sval}", font_size=11, color=NM_YELLOW)
            val_lbl.move_to((x_screen, y_bot - 0.3, 0))
            sample_lines.add(dash)
            sample_value_labels.add(val_lbl)

        self.play(
            LaggedStart(*[Create(d) for d in sample_lines], lag_ratio=0.1),
            LaggedStart(*[FadeIn(v) for v in sample_value_labels], lag_ratio=0.1),
            run_time=0.9,
        )

        # Highlight the winner: A3 has highest sample
        winner_note = Text("Pull A3 — highest sample θ=0.82", font_size=16, color=NM_GREEN)
        winner_note.to_edge(DOWN, buff=0.6)
        self.play(
            Indicate(curve_group[1], scale_factor=1.05),
            FadeIn(winner_note),
            run_time=0.7,
        )
        self.wait(0.4)

        # Show posterior update: A3 curve gets narrower/taller after a successful pull
        narrow_curve = axes_group[1].plot(
            lambda x: 4.0 * ((x ** 13) * ((1 - x) ** 2)) / max(
                (t / 100) ** 13 * ((1 - t / 100) ** 2) for t in range(1, 100)
            ) if 0 < x < 1 else 0,
            x_range=[0.01, 0.99],
            color=ARM_COLORS[2],
            stroke_width=2.5,
        )
        update_note = Text("observe reward → posterior narrows", font_size=14, color=NM_PURPLE)
        update_note.next_to(winner_note, UP, buff=0.15)
        self.play(
            Transform(curve_group[1], narrow_curve),
            FadeIn(update_note),
            run_time=0.8,
        )
        self.wait(0.6)

        self.play(
            FadeOut(strategy_label), FadeOut(prior_text),
            FadeOut(axes_group), FadeOut(curve_group),
            FadeOut(arm_curve_labels), FadeOut(sample_lines),
            FadeOut(sample_value_labels), FadeOut(winner_note),
            FadeOut(update_note),
            run_time=0.7,
        )

    # ------------------------------------------------------------------
    # Step 5: Regret comparison
    # ------------------------------------------------------------------
    def _show_regret_comparison(self):
        header = Text("Cumulative Regret Comparison", font_size=28, color=NM_TEXT, weight=BOLD)
        header.to_edge(UP, buff=0.4)
        self.play(Write(header), run_time=0.6)

        # Draw simple axes for regret over time
        axes = Axes(
            x_range=[0, 100, 20],
            y_range=[0, 60, 15],
            x_length=7.0,
            y_length=3.5,
            axis_config={"color": NM_GRID, "stroke_width": 1.5},
            x_axis_config={"include_tip": True},
            y_axis_config={"include_tip": True},
            tips=True,
        )
        axes.move_to(DOWN * 0.3)

        x_label = Text("Rounds (T)", font_size=14, color=NM_GRID)
        x_label.next_to(axes.x_axis, DOWN, buff=0.2)
        y_label = Text("Regret", font_size=14, color=NM_GRID)
        y_label.next_to(axes.y_axis, LEFT, buff=0.15)
        y_label.rotate(PI / 2)

        self.play(Create(axes), FadeIn(x_label), FadeIn(y_label), run_time=0.7)

        import math

        # ε-Greedy: roughly linear growth (constant fraction explore forever)
        eps_curve = axes.plot(
            lambda t: 0.5 * t if t > 0 else 0,
            x_range=[1, 100],
            color=NM_BLUE,
            stroke_width=2.5,
        )

        # UCB1: O(K ln T) — logarithmic
        ucb_curve = axes.plot(
            lambda t: 10 * math.log(max(t, 1)),
            x_range=[1, 100],
            color=NM_GREEN,
            stroke_width=2.5,
        )

        # Thompson Sampling: near-optimal, sub-logarithmic empirically
        ts_curve = axes.plot(
            lambda t: 5 * math.log(max(t, 1)),
            x_range=[1, 100],
            color=NM_PURPLE,
            stroke_width=2.5,
        )

        # Animate curves one by one with labels
        eps_label = Text("ε-Greedy (linear)", font_size=14, color=NM_BLUE)
        eps_label.next_to(axes.c2p(100, 50), RIGHT, buff=-1.8)
        eps_label.shift(UP * 0.1)

        ucb_label = Text("UCB1 (log)", font_size=14, color=NM_GREEN)
        ucb_label.next_to(axes.c2p(100, 46), RIGHT, buff=-1.5)
        ucb_label.shift(DOWN * 0.2)

        ts_label = Text("Thompson (near-optimal)", font_size=14, color=NM_PURPLE)
        ts_label.next_to(axes.c2p(100, 23), RIGHT, buff=-2.8)
        ts_label.shift(DOWN * 0.2)

        self.play(Create(eps_curve), FadeIn(eps_label), run_time=0.8)
        self.play(Create(ucb_curve), FadeIn(ucb_label), run_time=0.8)
        self.play(Create(ts_curve), FadeIn(ts_label), run_time=0.8)
        self.wait(1.0)

        self.play(
            FadeOut(header), FadeOut(axes), FadeOut(x_label), FadeOut(y_label),
            FadeOut(eps_curve), FadeOut(ucb_curve), FadeOut(ts_curve),
            FadeOut(eps_label), FadeOut(ucb_label), FadeOut(ts_label),
            run_time=0.8,
        )

    # ------------------------------------------------------------------
    # Step 6: Summary
    # ------------------------------------------------------------------
    def _show_summary(self):
        insight = Text(
            "Thompson Sampling naturally balances\nexploration and exploitation",
            font_size=26, color=NM_TEXT, weight=BOLD,
            line_spacing=1.2,
        )
        insight.move_to(ORIGIN + UP * 0.6)

        sub_insights = VGroup(
            Text("ε-Greedy: simple, linear regret", font_size=17, color=NM_BLUE),
            Text("UCB1: principled, O(K ln T) regret", font_size=17, color=NM_GREEN),
            Text("Thompson Sampling: Bayesian, lowest empirical regret", font_size=17, color=NM_PURPLE),
        )
        sub_insights.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        sub_insights.next_to(insight, DOWN, buff=0.5)

        self.play(Write(insight), run_time=1.0)
        self.play(
            LaggedStart(*[FadeIn(s, shift=UP * 0.15) for s in sub_insights], lag_ratio=0.25),
            run_time=1.0,
        )
        self.wait(1.2)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.9)
