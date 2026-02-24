"""
Scene: Batch Normalization
Script: microbatchnorm.py
Description: Normalize activations to zero mean, unit variance — stabilize training
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE, NM_PURPLE
from manim import *


class BatchNormScene(NoMagicScene):
    title_text = "Batch Normalization"
    subtitle_text = "Normalize activations — stabilize training, enable deeper networks"

    def animate(self):
        # === Step 1: Show unnormalized activations (scattered, shifted) ===
        before_label = Text("Before BatchNorm", font_size=20, color=NM_PRIMARY, weight=BOLD)
        before_label.move_to(LEFT * 3.2 + UP * 2.8)

        # Number line for activation values
        before_line = NumberLine(
            x_range=[-1, 5, 1], length=5, color=NM_GRID,
            include_numbers=True, font_size=14,
            decimal_number_config={"color": NM_TEXT, "num_decimal_places": 0},
            stroke_width=1.5,
        )
        before_line.move_to(LEFT * 3.2 + UP * 1.5)

        # Scattered dots (high mean, high variance)
        before_vals = [1.2, 2.8, 3.5, 0.7, 4.1, 2.1, 3.9, 1.5]
        before_dots = VGroup()
        for val in before_vals:
            dot = Dot(before_line.n2p(val), radius=0.07, color=NM_PRIMARY)
            dot.set_fill(opacity=0.7)
            before_dots.add(dot)

        mean_val = sum(before_vals) / len(before_vals)
        mean_line = DashedLine(
            before_line.n2p(mean_val) + UP * 0.3,
            before_line.n2p(mean_val) + DOWN * 0.3,
            color=NM_YELLOW, stroke_width=2,
        )
        mean_label = Text(f"\u03bc={mean_val:.1f}", font_size=12, color=NM_YELLOW)
        mean_label.next_to(mean_line, UP, buff=0.1)

        self.play(Write(before_label), Create(before_line), run_time=0.6)
        self.play(
            LaggedStart(*[FadeIn(d, scale=0.5) for d in before_dots], lag_ratio=0.05),
            run_time=0.6,
        )
        self.play(Create(mean_line), FadeIn(mean_label), run_time=0.4)
        self.wait(0.6)

        # === Step 2: Show the normalization formula ===
        formula = VGroup(
            Text("x\u0302 = (x - \u03bc) / \u03c3", font_size=18, color=NM_GREEN),
            Text("y = \u03b3 \u00b7 x\u0302 + \u03b2", font_size=18, color=NM_GREEN),
            Text("\u03b3, \u03b2 are learnable", font_size=13, color=NM_GRID),
        ).arrange(DOWN, buff=0.12)
        formula.move_to(UP * 0.0)

        self.play(
            LaggedStart(*[FadeIn(f) for f in formula], lag_ratio=0.2),
            run_time=0.8,
        )
        self.wait(0.6)

        # === Step 3: Show normalized activations (centered, tight) ===
        after_label = Text("After BatchNorm", font_size=20, color=NM_GREEN, weight=BOLD)
        after_label.move_to(RIGHT * 3.2 + UP * 2.8)

        after_line = NumberLine(
            x_range=[-3, 3, 1], length=5, color=NM_GRID,
            include_numbers=True, font_size=14,
            decimal_number_config={"color": NM_TEXT, "num_decimal_places": 0},
            stroke_width=1.5,
        )
        after_line.move_to(RIGHT * 3.2 + UP * 1.5)

        # Normalized values (zero mean, unit variance)
        std_val = (sum((v - mean_val) ** 2 for v in before_vals) / len(before_vals)) ** 0.5
        after_vals = [(v - mean_val) / std_val for v in before_vals]
        after_dots = VGroup()
        for val in after_vals:
            dot = Dot(after_line.n2p(val), radius=0.07, color=NM_GREEN)
            dot.set_fill(opacity=0.7)
            after_dots.add(dot)

        zero_line = DashedLine(
            after_line.n2p(0) + UP * 0.3,
            after_line.n2p(0) + DOWN * 0.3,
            color=NM_YELLOW, stroke_width=2,
        )
        zero_label = Text("\u03bc=0", font_size=12, color=NM_YELLOW)
        zero_label.next_to(zero_line, UP, buff=0.1)

        self.play(Write(after_label), Create(after_line), run_time=0.6)

        # Animate dots transforming from scattered to normalized
        transform_arrows = VGroup()
        for bd, ad in zip(before_dots, after_dots):
            arr = Arrow(
                bd.get_center(), ad.get_center(),
                color=NM_ORANGE, stroke_width=1, buff=0.1, tip_length=0.08,
                stroke_opacity=0.4,
            )
            transform_arrows.add(arr)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in transform_arrows], lag_ratio=0.03),
            run_time=0.8,
        )
        self.play(
            LaggedStart(*[FadeIn(d, scale=0.5) for d in after_dots], lag_ratio=0.05),
            FadeOut(transform_arrows),
            run_time=0.6,
        )
        self.play(Create(zero_line), FadeIn(zero_label), run_time=0.4)
        self.wait(0.6)

        # === Step 4: Training benefit ===
        benefit = VGroup(
            Text("internal covariate shift eliminated", font_size=16, color=NM_TEXT),
            Text("\u2192 higher learning rates, faster convergence", font_size=16, color=NM_YELLOW, weight=BOLD),
        ).arrange(DOWN, buff=0.12)
        benefit.move_to(DOWN * 2.5)

        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.15) for b in benefit], lag_ratio=0.2),
            run_time=0.8,
        )
        self.wait(1.6)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.9)
