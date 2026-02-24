"""
Scene: Rotary Position Embedding (RoPE)
Script: microrope.py
Description: Position through rotation — how rotating Q/K vectors encodes relative distance
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE, NM_PURPLE
from manim import *
import math


class RoPEScene(NoMagicScene):
    title_text = "Rotary Position Embedding"
    subtitle_text = "Encode position by rotating Q and K vectors"

    def animate(self):
        # === Step 1: Show a 2D vector being rotated ===
        plane_label = Text("2D embedding subspace", font_size=18, color=NM_BLUE, weight=BOLD)
        plane_label.move_to(UP * 2.8)
        self.play(Write(plane_label), run_time=0.3)

        # Draw axes
        axes = Axes(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1],
            x_length=4, y_length=4,
            axis_config={"color": NM_GRID, "stroke_width": 1},
            tips=False,
        )
        axes.move_to(LEFT * 3.0 + DOWN * 0.3)

        self.play(Create(axes), run_time=0.4)

        # Show a query vector at position 0
        vec_angle_0 = 0.5  # radians
        vec_len = 1.5
        vec_end_0 = axes.c2p(vec_len * math.cos(vec_angle_0), vec_len * math.sin(vec_angle_0))

        q_arrow = Arrow(
            axes.c2p(0, 0), vec_end_0,
            color=NM_PRIMARY, stroke_width=3, buff=0, tip_length=0.15,
        )
        q_label = Text("Q (pos=0)", font_size=13, color=NM_PRIMARY)
        q_label.next_to(q_arrow.get_end(), UR, buff=0.1)

        self.play(GrowArrow(q_arrow), FadeIn(q_label), run_time=0.4)
        self.wait(0.2)

        # Rotate by theta for position 1
        theta_1 = 0.6  # rotation angle for position 1
        vec_end_1 = axes.c2p(
            vec_len * math.cos(vec_angle_0 + theta_1),
            vec_len * math.sin(vec_angle_0 + theta_1),
        )
        q_arrow_1 = Arrow(
            axes.c2p(0, 0), vec_end_1,
            color=NM_YELLOW, stroke_width=3, buff=0, tip_length=0.15,
        )
        q_label_1 = Text("Q (pos=1)", font_size=13, color=NM_YELLOW)
        q_label_1.next_to(q_arrow_1.get_end(), UR, buff=0.1)

        # Show rotation arc
        arc = Arc(
            radius=0.8, start_angle=vec_angle_0, angle=theta_1,
            arc_center=axes.c2p(0, 0), color=NM_GREEN, stroke_width=2,
        )
        theta_label = Text("\u03b8", font_size=18, color=NM_GREEN)
        theta_label.next_to(arc, RIGHT, buff=0.1)

        self.play(
            GrowArrow(q_arrow_1), FadeIn(q_label_1),
            Create(arc), FadeIn(theta_label),
            run_time=0.5,
        )

        # Position 2 — another rotation
        vec_end_2 = axes.c2p(
            vec_len * math.cos(vec_angle_0 + 2 * theta_1),
            vec_len * math.sin(vec_angle_0 + 2 * theta_1),
        )
        q_arrow_2 = Arrow(
            axes.c2p(0, 0), vec_end_2,
            color=NM_ORANGE, stroke_width=3, buff=0, tip_length=0.15,
        )
        q_label_2 = Text("Q (pos=2)", font_size=13, color=NM_ORANGE)
        q_label_2.next_to(q_arrow_2.get_end(), UP, buff=0.1)

        self.play(GrowArrow(q_arrow_2), FadeIn(q_label_2), run_time=0.4)
        self.wait(0.3)

        # === Step 2: Show the rotation matrix formula ===
        formula_group = VGroup()
        formula_title = Text("RoPE rotation (per 2D pair):", font_size=15, color=NM_TEXT)
        formula_line1 = Text("q'_2i   = q_2i \u00b7 cos(m\u03b8) - q_2i+1 \u00b7 sin(m\u03b8)", font_size=13, color=NM_GREEN)
        formula_line2 = Text("q'_2i+1 = q_2i \u00b7 sin(m\u03b8) + q_2i+1 \u00b7 cos(m\u03b8)", font_size=13, color=NM_GREEN)
        formula_note = Text("m = position index, \u03b8 = frequency per dimension", font_size=11, color=NM_GRID)

        formula_group = VGroup(formula_title, formula_line1, formula_line2, formula_note)
        formula_group.arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        formula_group.move_to(RIGHT * 3.0 + UP * 1.5)

        self.play(
            LaggedStart(*[FadeIn(f) for f in formula_group], lag_ratio=0.15),
            run_time=0.6,
        )
        self.wait(0.3)

        # === Step 3: Key insight — relative position in dot product ===
        insight_box = RoundedRectangle(
            corner_radius=0.1, width=5.5, height=1.5,
            color=NM_YELLOW, fill_opacity=0.08, stroke_width=1.5,
        )
        insight_box.move_to(RIGHT * 3.0 + DOWN * 1.2)

        insight_title = Text("Key insight", font_size=16, color=NM_YELLOW, weight=BOLD)
        insight_title.next_to(insight_box, UP, buff=0.08)

        insight_text = VGroup(
            Text("Q(pos=m)\u1d40 \u00b7 K(pos=n) depends only on (m-n)", font_size=13, color=NM_TEXT),
            Text("Rotation of Q and K by the same angle cancels", font_size=12, color=NM_GREEN),
            Text("\u2192 attention scores encode relative distance", font_size=12, color=NM_GREEN),
        ).arrange(DOWN, buff=0.08)
        insight_text.move_to(insight_box.get_center())

        self.play(FadeIn(insight_box), Write(insight_title), run_time=0.3)
        self.play(
            LaggedStart(*[FadeIn(t) for t in insight_text], lag_ratio=0.15),
            run_time=0.5,
        )
        self.wait(0.3)

        # === Step 4: Multi-frequency visualization ===
        freq_label = Text("Different dimensions rotate at different frequencies", font_size=14, color=NM_PURPLE)
        freq_label.move_to(DOWN * 3.0)

        self.play(FadeIn(freq_label, shift=UP * 0.15), run_time=0.4)
        self.wait(0.8)

        # Cleanup
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.6)
