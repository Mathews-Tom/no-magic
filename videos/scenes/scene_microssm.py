"""
Scene: State Space Models (Mamba)
Script: microssm.py
Description: Linear-time sequence modeling — state transitions replace attention
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE, NM_PURPLE
from manim import *


class SSMScene(NoMagicScene):
    title_text = "State Space Models"
    subtitle_text = "Linear-time sequences — Mamba's selective state transitions"

    def animate(self):
        # === Step 1: Show the SSM recurrence ===
        formula_label = Text("SSM Recurrence", font_size=20, color=NM_PRIMARY, weight=BOLD)
        formula_label.move_to(UP * 2.8)
        self.play(Write(formula_label), run_time=0.3)

        # State equation: h_t = A * h_{t-1} + B * x_t
        # Output equation: y_t = C * h_t
        eq1 = Text("h(t) = A \u00b7 h(t-1) + B \u00b7 x(t)", font_size=18, color=NM_GREEN)
        eq2 = Text("y(t) = C \u00b7 h(t)", font_size=18, color=NM_YELLOW)
        eq_note = Text("h = hidden state, x = input, y = output", font_size=13, color=NM_GRID)
        eqs = VGroup(eq1, eq2, eq_note).arrange(DOWN, buff=0.15)
        eqs.move_to(UP * 1.8)

        self.play(
            LaggedStart(*[FadeIn(e) for e in eqs], lag_ratio=0.2),
            run_time=0.6,
        )
        self.wait(0.3)

        # === Step 2: Animate the state transition step-by-step ===
        self.play(FadeOut(eqs), FadeOut(formula_label), run_time=0.3)

        seq_label = Text("Sequential processing (5 timesteps)", font_size=16, color=NM_TEXT)
        seq_label.move_to(UP * 2.8)
        self.play(Write(seq_label), run_time=0.3)

        # Input tokens
        input_tokens = ["x\u2081", "x\u2082", "x\u2083", "x\u2084", "x\u2085"]
        input_boxes = VGroup()
        for tok in input_tokens:
            box = RoundedRectangle(
                corner_radius=0.08, width=0.7, height=0.5,
                color=NM_BLUE, fill_opacity=0.2, stroke_width=1.5,
            )
            label = Text(tok, font_size=16, color=NM_TEXT)
            label.move_to(box.get_center())
            input_boxes.add(VGroup(box, label))
        input_boxes.arrange(RIGHT, buff=0.5)
        input_boxes.move_to(UP * 1.8)

        input_label = Text("inputs", font_size=13, color=NM_BLUE)
        input_label.next_to(input_boxes, LEFT, buff=0.3)
        self.play(
            FadeIn(input_label),
            LaggedStart(*[FadeIn(b, shift=DOWN * 0.1) for b in input_boxes], lag_ratio=0.08),
            run_time=0.5,
        )

        # Hidden state boxes
        state_boxes = VGroup()
        for i in range(5):
            box = RoundedRectangle(
                corner_radius=0.1, width=0.8, height=0.6,
                color=NM_GREEN, fill_opacity=0.1, stroke_width=1.5,
            )
            label = Text(f"h\u200a{i+1}", font_size=16, color=NM_GREEN)
            label.move_to(box.get_center())
            state_boxes.add(VGroup(box, label))
        state_boxes.arrange(RIGHT, buff=0.4)
        state_boxes.move_to(DOWN * 0.0)

        state_label = Text("hidden states", font_size=13, color=NM_GREEN)
        state_label.next_to(state_boxes, LEFT, buff=0.2)

        # Output boxes
        output_boxes = VGroup()
        for i in range(5):
            box = RoundedRectangle(
                corner_radius=0.08, width=0.7, height=0.5,
                color=NM_YELLOW, fill_opacity=0.1, stroke_width=1.5,
            )
            label = Text(f"y\u200a{i+1}", font_size=16, color=NM_YELLOW)
            label.move_to(box.get_center())
            output_boxes.add(VGroup(box, label))
        output_boxes.arrange(RIGHT, buff=0.5)
        output_boxes.move_to(DOWN * 1.8)

        output_label = Text("outputs", font_size=13, color=NM_YELLOW)
        output_label.next_to(output_boxes, LEFT, buff=0.3)

        # Animate step-by-step: input -> state -> output, with state carrying forward
        for i in range(5):
            anims = []

            # Input arrow down to state
            in_arr = Arrow(
                input_boxes[i].get_bottom(), state_boxes[i].get_top(),
                color=NM_BLUE, stroke_width=1.5, buff=0.08, tip_length=0.1,
                stroke_opacity=0.6,
            )

            # State carry-forward arrow (from previous state)
            if i > 0:
                carry_arr = Arrow(
                    state_boxes[i - 1].get_right(), state_boxes[i].get_left(),
                    color=NM_GREEN, stroke_width=2, buff=0.08, tip_length=0.1,
                )
                a_label = Text("A", font_size=11, color=NM_GREEN, weight=BOLD)
                a_label.next_to(carry_arr, UP, buff=0.03)
                anims.extend([GrowArrow(carry_arr), FadeIn(a_label)])

            # Output arrow from state
            out_arr = Arrow(
                state_boxes[i].get_bottom(), output_boxes[i].get_top(),
                color=NM_YELLOW, stroke_width=1.5, buff=0.08, tip_length=0.1,
                stroke_opacity=0.6,
            )

            # Activate state
            anims.extend([
                GrowArrow(in_arr),
                FadeIn(state_boxes[i], shift=DOWN * 0.1),
                state_boxes[i][0].animate.set_fill(NM_GREEN, opacity=0.3),
            ])

            self.play(*anims, run_time=0.35)

            # Output
            self.play(
                GrowArrow(out_arr),
                FadeIn(output_boxes[i], shift=DOWN * 0.1),
                run_time=0.2,
            )

            # Dim previous state
            if i > 0:
                self.play(
                    state_boxes[i - 1][0].animate.set_fill(NM_GREEN, opacity=0.1),
                    run_time=0.1,
                )

        self.play(FadeIn(state_label), FadeIn(output_label), run_time=0.2)
        self.wait(0.3)

        # === Step 3: Selective mechanism (Mamba's key contribution) ===
        selective_label = Text(
            "Mamba: A, B, C are input-dependent (selective)",
            font_size=16, color=NM_PURPLE, weight=BOLD,
        )
        selective_label.move_to(DOWN * 2.8)

        linear_label = Text(
            "O(n) time, O(1) memory per step — no attention matrix",
            font_size=14, color=NM_YELLOW,
        )
        linear_label.move_to(DOWN * 3.3)

        self.play(Write(selective_label), run_time=0.4)
        self.play(FadeIn(linear_label, shift=UP * 0.15), run_time=0.3)
        self.wait(0.8)

        # Cleanup
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.6)
