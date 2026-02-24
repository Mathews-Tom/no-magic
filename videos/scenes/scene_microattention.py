"""
Scene: Attention Mechanism
Script: microattention.py
Description: Scaled dot-product attention — Q·K^T → scale → softmax → weighted V
"""
import sys
from pathlib import Path

# Ensure the scenes directory is on the import path for base.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW
from manim import *


# --- Helper: build a labeled matrix grid ---

def make_matrix_block(label_text, rows, cols, cell_color, label_color):
    """Build a matrix grid with a label above it.

    Returns a VGroup of (label, grid, cells) where cells is a flat list of
    Squares that can be individually colored later.
    """
    cell_size = 0.4
    cells = VGroup()
    for r in range(rows):
        for c in range(cols):
            sq = Square(side_length=cell_size, stroke_width=1, stroke_color=NM_GRID)
            sq.set_fill(cell_color, opacity=0.35)
            sq.move_to(RIGHT * c * cell_size + DOWN * r * cell_size)
            cells.add(sq)

    # Center the grid at origin
    cells.move_to(ORIGIN)

    label = Text(label_text, font_size=22, color=label_color, weight=BOLD)
    label.next_to(cells, UP, buff=0.2)

    block = VGroup(label, cells)
    return block


class AttentionScene(NoMagicScene):
    title_text = "Attention Mechanism"
    subtitle_text = "Q · K^T → softmax → weighted V"

    def animate(self):
        rows, cols = 4, 4

        # === Step 1: Show Q, K, V matrices ===
        q_block = make_matrix_block("Q", rows, cols, NM_PRIMARY, NM_PRIMARY)
        k_block = make_matrix_block("K", rows, cols, NM_BLUE, "#4a7fbf")
        v_block = make_matrix_block("V", rows, cols, NM_GREEN, NM_GREEN)

        q_block.move_to(LEFT * 4.5 + UP * 1.2)
        k_block.move_to(LEFT * 1.5 + UP * 1.2)
        v_block.move_to(RIGHT * 1.5 + UP * 1.2)

        self.play(
            FadeIn(q_block, shift=DOWN * 0.3),
            FadeIn(k_block, shift=DOWN * 0.3),
            FadeIn(v_block, shift=DOWN * 0.3),
            run_time=1.2,
        )
        self.wait(1.0)

        # === Step 2: Q · K^T — dot product ===
        step_label = Text("Q · K^T", font_size=26, color=NM_YELLOW, weight=BOLD)
        step_label.move_to(LEFT * 3.0 + DOWN * 0.6)
        self.play(Write(step_label), run_time=0.8)

        # Animate connecting lines from Q rows to K columns
        q_cells = q_block[1]  # grid cells
        k_cells = k_block[1]
        lines = VGroup()
        for r in range(rows):
            q_cell = q_cells[r * cols]  # left-most cell of row r
            k_cell = k_cells[r]  # top cell of column 0 (represents the column)
            line = Line(
                q_cell.get_right(), k_cell.get_left(),
                stroke_width=1.5, color=NM_YELLOW, stroke_opacity=0.6,
            )
            lines.add(line)

        self.play(LaggedStart(*[Create(l) for l in lines], lag_ratio=0.1), run_time=1.2)
        self.wait(0.6)
        self.play(FadeOut(lines), run_time=0.6)

        # === Step 3: Show attention score matrix ===
        scores_block = make_matrix_block("Scores", rows, rows, NM_YELLOW, NM_YELLOW)
        scores_block.move_to(RIGHT * 4.5 + UP * 1.2)
        self.play(FadeIn(scores_block, shift=LEFT * 0.3), run_time=0.9)

        # Scale label
        scale_label = Text("scale by 1/√d", font_size=20, color=NM_TEXT)
        scale_label.next_to(scores_block, DOWN, buff=0.2)
        self.play(Write(scale_label), run_time=0.8)
        self.wait(0.6)

        # === Step 4: Softmax row-by-row ===
        softmax_label = Text("softmax", font_size=26, color=NM_PRIMARY, weight=BOLD)
        softmax_label.move_to(RIGHT * 4.5 + DOWN * 0.6)

        old_step = step_label
        self.play(
            Write(softmax_label),
            FadeOut(old_step),
            run_time=0.8,
        )

        # Animate rows of the score matrix transitioning to probability colors
        score_cells = scores_block[1]
        for r in range(rows):
            row_cells = [score_cells[r * rows + c] for c in range(rows)]
            # Gradient opacity: each cell in the row gets a different intensity
            # to represent probability weights (some tokens attend more)
            opacities = [0.2 + 0.6 * (c + 1) / rows for c in range(rows)]
            anims = [
                cell.animate.set_fill(NM_PRIMARY, opacity=op)
                for cell, op in zip(row_cells, opacities)
            ]
            self.play(*anims, run_time=0.4)

        self.wait(0.6)

        # === Step 5: Attention weights × V ===
        times_v_label = Text("× V", font_size=26, color=NM_GREEN, weight=BOLD)
        times_v_label.next_to(softmax_label, DOWN, buff=0.3)
        self.play(Write(times_v_label), run_time=0.6)

        # Draw arrow from scores to V
        arrow = Arrow(
            scores_block.get_bottom() + DOWN * 0.2,
            v_block.get_right() + RIGHT * 0.3,
            color=NM_GREEN,
            stroke_width=2,
            buff=0.15,
        )
        self.play(GrowArrow(arrow), run_time=0.8)
        self.wait(0.6)

        # === Step 6: Final output ===
        # Clear middle area and show output
        output_block = make_matrix_block("Output", rows, cols, NM_GREEN, NM_GREEN)
        output_block.move_to(DOWN * 2.2)

        # Flash all output cells green
        output_cells = output_block[1]
        self.play(FadeIn(output_block, shift=UP * 0.3), run_time=0.9)

        result_box = SurroundingRectangle(output_block, color=NM_GREEN, buff=0.15, stroke_width=2)
        self.play(Create(result_box), run_time=0.8)

        result_label = Text(
            "Context-aware representations",
            font_size=20,
            color=NM_TEXT,
        )
        result_label.next_to(result_box, DOWN, buff=0.2)
        self.play(FadeIn(result_label), run_time=0.6)

        self.wait(2.0)

        # Cleanup for end card
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=0.9,
        )
