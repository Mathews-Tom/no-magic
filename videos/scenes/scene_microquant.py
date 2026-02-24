"""
Scene: Quantization
Script: microquant.py
Description: Float32 → Int8 compression — mapping continuous values to discrete bins
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE
from manim import *


class QuantizationScene(NoMagicScene):
    title_text = "Quantization"
    subtitle_text = "Shrink a model 4x — Float32 to Int8"

    def animate(self):
        # === Step 1: Show float32 number line (high precision) ===
        float_label = Text("Float32 weights", font_size=22, color=NM_BLUE, weight=BOLD)
        float_label.move_to(UP * 2.8)

        # Number line
        float_line = NumberLine(
            x_range=[-2, 2, 0.5],
            length=10,
            color=NM_BLUE,
            include_numbers=True,
            font_size=16,
            decimal_number_config={"color": NM_TEXT, "num_decimal_places": 1},
            stroke_width=2,
        )
        float_line.move_to(UP * 1.8)

        # Sample weight values as dots on the line
        weight_values = [-1.73, -0.92, -0.31, 0.15, 0.67, 1.24, 1.85]
        weight_dots = VGroup()
        weight_labels = VGroup()
        for val in weight_values:
            dot = Dot(
                float_line.n2p(val), radius=0.08, color=NM_YELLOW,
            )
            dot.set_fill(opacity=0.8)
            label = Text(f"{val:.2f}", font_size=10, color=NM_TEXT)
            label.next_to(dot, UP, buff=0.12)
            weight_dots.add(dot)
            weight_labels.add(label)

        self.play(Write(float_label), Create(float_line), run_time=0.6)
        self.play(
            LaggedStart(*[FadeIn(d, scale=0.5) for d in weight_dots], lag_ratio=0.05),
            LaggedStart(*[FadeIn(l) for l in weight_labels], lag_ratio=0.05),
            run_time=0.6,
        )
        self.wait(0.4)

        # === Step 2: Show quantization parameters ===
        params = VGroup(
            Text("scale = (max - min) / 255", font_size=16, color=NM_YELLOW),
            Text("zero_point = round(-min / scale)", font_size=16, color=NM_YELLOW),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        params.move_to(RIGHT * 4.0 + UP * 0.2)
        self.play(
            LaggedStart(*[Write(p) for p in params], lag_ratio=0.2),
            run_time=0.6,
        )
        self.wait(0.3)

        # === Step 3: Show int8 number line (quantized) ===
        int_label = Text("Int8 quantized", font_size=22, color=NM_GREEN, weight=BOLD)
        int_label.move_to(DOWN * 0.5)

        int_line = NumberLine(
            x_range=[0, 255, 32],
            length=10,
            color=NM_GREEN,
            include_numbers=True,
            font_size=16,
            decimal_number_config={"color": NM_TEXT, "num_decimal_places": 0},
            stroke_width=2,
        )
        int_line.move_to(DOWN * 1.3)

        self.play(Write(int_label), Create(int_line), run_time=0.6)

        # === Step 4: Animate mapping — dots move from float line to int bins ===
        # Map float values to int8: q = round((val - min) / scale)
        fmin, fmax = -2.0, 2.0
        scale = (fmax - fmin) / 255
        quant_vals = [round((v - fmin) / scale) for v in weight_values]
        quant_vals = [max(0, min(255, q)) for q in quant_vals]

        quant_dots = VGroup()
        quant_labels = VGroup()
        for qval in quant_vals:
            dot = Dot(
                int_line.n2p(qval), radius=0.08, color=NM_GREEN,
            )
            dot.set_fill(opacity=0.8)
            label = Text(str(qval), font_size=10, color=NM_TEXT)
            label.next_to(dot, DOWN, buff=0.12)
            quant_dots.add(dot)
            quant_labels.add(label)

        # Animate arrows from float dots to int dots
        arrows = VGroup()
        for wd, qd in zip(weight_dots, quant_dots):
            arr = Arrow(
                wd.get_center(), qd.get_center(),
                color=NM_ORANGE, stroke_width=1.5, buff=0.1, tip_length=0.12,
                stroke_opacity=0.5,
            )
            arrows.add(arr)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.05),
            run_time=0.8,
        )
        self.play(
            LaggedStart(*[FadeIn(d, scale=0.5) for d in quant_dots], lag_ratio=0.05),
            LaggedStart(*[FadeIn(l) for l in quant_labels], lag_ratio=0.05),
            FadeOut(arrows),
            run_time=0.6,
        )
        self.wait(0.4)

        # === Step 5: Memory comparison ===
        comparison = VGroup(
            Text("32 bits/weight", font_size=18, color=NM_BLUE),
            Text("→", font_size=24, color=NM_TEXT),
            Text("8 bits/weight", font_size=18, color=NM_GREEN),
            Text("= 4x smaller", font_size=18, color=NM_YELLOW, weight=BOLD),
        ).arrange(RIGHT, buff=0.3)
        comparison.move_to(DOWN * 2.8)

        self.play(
            LaggedStart(*[FadeIn(c, shift=UP * 0.15) for c in comparison], lag_ratio=0.15),
            run_time=0.6,
        )
        self.wait(0.8)

        # Cleanup
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.6)
