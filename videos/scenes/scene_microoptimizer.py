"""
Scene: Optimizer Comparison
Script: microoptimizer.py
Description: SGD vs Momentum vs RMSProp vs Adam — racing down the loss landscape
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE, NM_PURPLE
from manim import *


class OptimizerScene(NoMagicScene):
    title_text = "Optimizer Comparison"
    subtitle_text = "SGD vs Momentum vs RMSProp vs Adam"

    def animate(self):
        # === Step 1: Show a stylized loss landscape ===
        landscape_label = Text("Loss Landscape", font_size=20, color=NM_TEXT, weight=BOLD)
        landscape_label.move_to(UP * 2.8)
        self.play(Write(landscape_label), run_time=0.4)

        # Draw contour-like ellipses (narrow valley)
        contours = VGroup()
        for i in range(5):
            scale = 0.4 + i * 0.35
            ellipse = Ellipse(
                width=scale * 3, height=scale * 1.0,
                color=NM_GRID, stroke_width=1, stroke_opacity=0.4 - i * 0.05,
            )
            ellipse.rotate(0.3)
            contours.add(ellipse)
        contours.move_to(DOWN * 0.3)

        # Minimum point
        minimum = Dot(ORIGIN + DOWN * 0.3, radius=0.06, color=NM_GREEN)
        min_label = Text("min", font_size=11, color=NM_GREEN)
        min_label.next_to(minimum, DOWN, buff=0.1)

        self.play(
            LaggedStart(*[Create(c) for c in contours], lag_ratio=0.08),
            FadeIn(minimum), FadeIn(min_label),
            run_time=0.9,
        )
        self.wait(0.4)

        # === Step 2: Animate 4 optimizer paths ===
        optimizers = [
            ("SGD", NM_PRIMARY, [
                (-2.5, 1.5), (-2.0, 0.8), (-1.6, 1.2), (-1.2, 0.5),
                (-0.9, 0.9), (-0.6, 0.3), (-0.4, 0.5), (-0.2, 0.1),
            ]),
            ("Momentum", NM_ORANGE, [
                (-2.5, 1.3), (-1.8, 0.3), (-1.0, 0.7), (-0.4, -0.1),
                (-0.1, 0.2), (0.0, 0.0),
            ]),
            ("RMSProp", NM_PURPLE, [
                (-2.5, 1.1), (-1.6, 0.5), (-0.9, 0.2), (-0.4, 0.0),
                (-0.1, -0.1), (0.0, 0.0),
            ]),
            ("Adam", NM_GREEN, [
                (-2.5, 0.9), (-1.5, 0.3), (-0.6, 0.0), (-0.1, -0.1),
                (0.0, 0.0),
            ]),
        ]

        legend_items = VGroup()
        paths = []

        for name, color, waypoints in optimizers:
            # Legend
            dot = Dot(radius=0.06, color=color)
            label = Text(name, font_size=13, color=color)
            label.next_to(dot, RIGHT, buff=0.1)
            legend_items.add(VGroup(dot, label))

            # Path
            points = [
                contours.get_center() + RIGHT * x + UP * y
                for x, y in waypoints
            ]
            path_dots = VGroup()
            path_lines = VGroup()
            for i, pt in enumerate(points):
                d = Dot(pt, radius=0.04, color=color)
                d.set_fill(opacity=0.8)
                path_dots.add(d)
                if i > 0:
                    line = Line(points[i - 1], pt, color=color, stroke_width=1.5, stroke_opacity=0.6)
                    path_lines.add(line)

            paths.append((path_dots, path_lines))

        legend_items.arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        legend_items.move_to(RIGHT * 5.0 + UP * 1.5)
        self.play(FadeIn(legend_items), run_time=0.4)

        # Animate paths simultaneously
        for dots, lines in paths:
            self.play(
                LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.08),
                LaggedStart(*[Create(l) for l in lines], lag_ratio=0.08),
                run_time=0.8,
            )

        self.wait(0.6)

        # === Step 3: Summary ===
        summary = VGroup(
            Text("SGD: oscillates in narrow valleys", font_size=13, color=NM_PRIMARY),
            Text("Momentum: overshoots but converges faster", font_size=13, color=NM_ORANGE),
            Text("Adam: momentum + adaptive LR \u2192 fastest convergence", font_size=13, color=NM_GREEN, weight=BOLD),
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        summary.move_to(DOWN * 2.8)

        self.play(
            LaggedStart(*[FadeIn(s, shift=UP * 0.1) for s in summary], lag_ratio=0.15),
            run_time=0.9,
        )
        self.wait(1.6)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.9)
