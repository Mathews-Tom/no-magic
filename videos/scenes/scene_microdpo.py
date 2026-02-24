"""
Scene: Direct Preference Optimization
Script: microdpo.py
Description: Aligning a model with human preferences — preferred vs rejected, no reward model
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE, NM_PURPLE
from manim import *


def make_response_box(label, text_lines, color, width=4.5, height=2.5):
    """Create a labeled response box with multi-line content."""
    box = RoundedRectangle(
        corner_radius=0.15, width=width, height=height,
        color=color, fill_opacity=0.1, stroke_width=1.5,
    )
    header = Text(label, font_size=20, color=color, weight=BOLD)
    header.next_to(box, UP, buff=0.15)
    content = VGroup(*[
        Text(line, font_size=16, color=NM_TEXT) for line in text_lines
    ]).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
    content.move_to(box.get_center())
    return VGroup(header, box, content)


class DPOScene(NoMagicScene):
    title_text = "Direct Preference Optimization"
    subtitle_text = "Aligning with preferences — no reward model needed"

    def animate(self):
        # === Step 1: Show the preference pair ===
        preferred = make_response_box(
            "Preferred  (y_w)",
            ['"emma"', '"sarah"', '"alice"'],
            NM_GREEN,
        )
        rejected = make_response_box(
            "Rejected  (y_l)",
            ['"xzqp"', '"aaaa"', '"9f3k"'],
            NM_PRIMARY,
        )

        preferred.move_to(LEFT * 3 + UP * 0.5)
        rejected.move_to(RIGHT * 3 + UP * 0.5)

        vs_text = Text("vs", font_size=28, color=NM_TEXT, weight=BOLD)
        vs_text.move_to(UP * 0.5)

        self.play(
            FadeIn(preferred, shift=RIGHT * 0.3),
            FadeIn(rejected, shift=LEFT * 0.3),
            Write(vs_text),
            run_time=0.8,
        )
        self.wait(0.5)

        # === Step 2: Show log probability comparison ===
        log_label = Text("compute log-probabilities under policy and reference",
                         font_size=16, color=NM_YELLOW)
        log_label.to_edge(UP, buff=0.5)
        self.play(Write(log_label), run_time=0.5)

        # Probability bars for preferred
        pref_bar = Rectangle(width=3.0, height=0.35, color=NM_GREEN,
                             fill_opacity=0.5, stroke_width=1)
        pref_bar.move_to(LEFT * 3 + DOWN * 1.5)
        pref_bar_label = Text("log P(y_w) = -2.1", font_size=14, color=NM_GREEN)
        pref_bar_label.next_to(pref_bar, DOWN, buff=0.1)

        # Probability bars for rejected
        rej_bar = Rectangle(width=1.5, height=0.35, color=NM_PRIMARY,
                            fill_opacity=0.5, stroke_width=1)
        rej_bar.move_to(RIGHT * 3 + DOWN * 1.5)
        rej_bar_label = Text("log P(y_l) = -4.3", font_size=14, color=NM_PRIMARY)
        rej_bar_label.next_to(rej_bar, DOWN, buff=0.1)

        self.play(
            GrowFromEdge(pref_bar, LEFT), FadeIn(pref_bar_label),
            GrowFromEdge(rej_bar, LEFT), FadeIn(rej_bar_label),
            run_time=0.7,
        )
        self.wait(0.4)

        # === Step 3: Show the DPO loss formula ===
        formula = Text(
            "L = -log sigmoid( beta * (log_ratio_w - log_ratio_l) )",
            font_size=18, color=NM_YELLOW,
        )
        formula.move_to(DOWN * 2.5)
        self.play(Write(formula), run_time=0.7)
        self.wait(0.3)

        # === Step 4: Show reward margin growing ===
        margin_label = Text("reward margin", font_size=16, color=NM_TEXT)
        margin_label.move_to(DOWN * 3.2)

        # Animate: preferred bar grows, rejected shrinks
        self.play(
            pref_bar.animate.stretch_to_fit_width(3.8),
            rej_bar.animate.stretch_to_fit_width(0.8),
            FadeIn(margin_label),
            run_time=0.8,
        )

        # Arrow showing the margin between the two
        margin_arrow = DoubleArrow(
            LEFT * 3 + DOWN * 1.1, RIGHT * 3 + DOWN * 1.1,
            color=NM_YELLOW, stroke_width=1.5, buff=0,
            tip_length=0.15,
        )
        margin_val = Text("margin grows", font_size=14, color=NM_YELLOW)
        margin_val.next_to(margin_arrow, UP, buff=0.08)
        self.play(GrowFromCenter(margin_arrow), FadeIn(margin_val), run_time=0.5)
        self.wait(0.3)

        # === Step 5: Result summary ===
        result = VGroup(
            Text("Policy learns:", font_size=18, color=NM_TEXT, weight=BOLD),
            Text("increase P(preferred)", font_size=16, color=NM_GREEN),
            Text("decrease P(rejected)", font_size=16, color=NM_PRIMARY),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        result.move_to(RIGHT * 4.5 + DOWN * 2.8)

        self.play(
            LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in result], lag_ratio=0.15),
            run_time=0.6,
        )
        self.wait(0.8)

        # Cleanup
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.6)
