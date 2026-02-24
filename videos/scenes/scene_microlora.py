"""
Scene: LoRA Fine-tuning
Script: microlora.py
Description: Low-rank adaptation — freezing base weights, training tiny A·B matrices alongside
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE, NM_PURPLE
from manim import *


def make_weight_grid(rows, cols, color, fill_opacity=0.3, cell_size=0.3):
    """Build a matrix grid of colored squares."""
    cells = VGroup()
    for r in range(rows):
        for c in range(cols):
            sq = Square(
                side_length=cell_size, stroke_width=1, stroke_color=NM_GRID,
            )
            sq.set_fill(color, opacity=fill_opacity)
            sq.move_to(RIGHT * c * cell_size + DOWN * r * cell_size)
            cells.add(sq)
    cells.move_to(ORIGIN)
    return cells


class LoRAScene(NoMagicScene):
    title_text = "LoRA Fine-tuning"
    subtitle_text = "Low-rank adaptation of frozen weights"

    def animate(self):
        # === Step 1: Show the frozen base weight matrix W ===
        w_rows, w_cols = 8, 8
        w_grid = make_weight_grid(w_rows, w_cols, NM_BLUE, fill_opacity=0.35)
        w_grid.move_to(LEFT * 2.5)

        w_label = Text("W", font_size=32, color="#4a7fbf", weight=BOLD)
        w_label.next_to(w_grid, UP, buff=0.25)

        w_dims = Text(f"{w_rows}×{w_cols}", font_size=16, color=NM_TEXT)
        w_dims.next_to(w_grid, DOWN, buff=0.15)

        self.play(FadeIn(w_grid, shift=UP * 0.2), Write(w_label), FadeIn(w_dims), run_time=0.7)
        self.wait(0.3)

        # === Step 2: Freeze it — overlay a lock icon (snowflake) and dim ===
        freeze_label = Text("frozen", font_size=18, color=NM_TEXT, slant=ITALIC)
        freeze_label.next_to(w_dims, DOWN, buff=0.15)

        # Dim all cells to show "frozen"
        freeze_anims = [cell.animate.set_fill(NM_BLUE, opacity=0.15) for cell in w_grid]
        self.play(*freeze_anims, FadeIn(freeze_label), run_time=0.6)

        # Add a subtle border to emphasize frozen state
        freeze_box = SurroundingRectangle(w_grid, color=NM_BLUE, buff=0.08, stroke_width=1.5)
        freeze_box.set_stroke(opacity=0.5)
        self.play(Create(freeze_box), run_time=0.3)
        self.wait(0.3)

        # === Step 3: Introduce LoRA matrices A and B ===
        # A: (d × r) — tall and thin
        # B: (r × d) — short and wide
        rank = 2
        a_grid = make_weight_grid(w_rows, rank, NM_PRIMARY, fill_opacity=0.5, cell_size=0.3)
        b_grid = make_weight_grid(rank, w_cols, NM_GREEN, fill_opacity=0.5, cell_size=0.3)

        a_grid.move_to(RIGHT * 2.0 + UP * 0.8)
        b_grid.move_to(RIGHT * 2.0 + DOWN * 1.2)

        a_label = Text("A", font_size=28, color=NM_PRIMARY, weight=BOLD)
        a_label.next_to(a_grid, UP, buff=0.2)
        a_dims = Text(f"{w_rows}×{rank}", font_size=14, color=NM_TEXT)
        a_dims.next_to(a_grid, RIGHT, buff=0.15)

        b_label = Text("B", font_size=28, color=NM_GREEN, weight=BOLD)
        b_label.next_to(b_grid, DOWN, buff=0.2)
        b_dims = Text(f"{rank}×{w_cols}", font_size=14, color=NM_TEXT)
        b_dims.next_to(b_grid, RIGHT, buff=0.15)

        # "trainable" label
        trainable_label = Text("trainable", font_size=18, color=NM_YELLOW, weight=BOLD)
        trainable_label.move_to(RIGHT * 4.5)

        self.play(
            FadeIn(a_grid, shift=LEFT * 0.3),
            FadeIn(b_grid, shift=LEFT * 0.3),
            Write(a_label), Write(b_label),
            FadeIn(a_dims), FadeIn(b_dims),
            FadeIn(trainable_label),
            run_time=0.8,
        )
        self.wait(0.3)

        # === Step 4: Show the multiplication A · B ===
        mult_sign = Text("·", font_size=36, color=NM_TEXT)
        mult_sign.move_to(RIGHT * 2.0 + DOWN * 0.15)
        self.play(Write(mult_sign), run_time=0.3)

        # Arrow from A·B to W — the "injection"
        ab_group = VGroup(a_grid, b_grid, mult_sign)
        inject_arrow = Arrow(
            ab_group.get_left() + LEFT * 0.1,
            w_grid.get_right() + RIGHT * 0.1,
            color=NM_YELLOW, stroke_width=2.5, buff=0.15,
        )
        plus_label = Text("+", font_size=30, color=NM_YELLOW, weight=BOLD)
        plus_label.next_to(inject_arrow, UP, buff=0.1)

        self.play(GrowArrow(inject_arrow), Write(plus_label), run_time=0.5)
        self.wait(0.3)

        # === Step 5: Show the effective weight W' = W + A·B ===
        formula = Text("W' = W + A·B", font_size=24, color=NM_YELLOW, weight=BOLD)
        formula.to_edge(DOWN, buff=0.6)
        self.play(Write(formula), run_time=0.6)
        self.wait(0.3)

        # === Step 6: Animate training — A and B cells pulse ===
        # Show gradient flowing through A and B (cells brighten)
        train_label = Text("training...", font_size=18, color=NM_GREEN)
        train_label.next_to(trainable_label, DOWN, buff=0.2)
        self.play(FadeIn(train_label), run_time=0.3)

        for _ in range(2):
            a_pulse = [cell.animate.set_fill(NM_PRIMARY, opacity=0.8) for cell in a_grid]
            b_pulse = [cell.animate.set_fill(NM_GREEN, opacity=0.8) for cell in b_grid]
            self.play(*a_pulse, *b_pulse, run_time=0.3)
            a_dim = [cell.animate.set_fill(NM_PRIMARY, opacity=0.5) for cell in a_grid]
            b_dim = [cell.animate.set_fill(NM_GREEN, opacity=0.5) for cell in b_grid]
            self.play(*a_dim, *b_dim, run_time=0.3)

        self.play(FadeOut(train_label), run_time=0.2)

        # === Step 7: Parameter count comparison ===
        param_text = VGroup(
            Text("W params:  64", font_size=18, color=NM_BLUE),
            Text("A+B params: 32", font_size=18, color=NM_GREEN),
            Text("→ 50% fewer to train", font_size=18, color=NM_YELLOW, weight=BOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        param_text.to_edge(RIGHT, buff=0.5).shift(DOWN * 0.3)

        self.play(
            LaggedStart(*[FadeIn(t, shift=LEFT * 0.2) for t in param_text], lag_ratio=0.2),
            run_time=0.7,
        )
        self.wait(1.0)

        # Cleanup
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.6)
