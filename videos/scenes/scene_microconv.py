"""
Scene: Convolutional Neural Network
Script: microconv.py
Description: Sliding kernel extracts spatial features — convolution + pooling
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE, NM_PURPLE
from manim import *


class ConvScene(NoMagicScene):
    title_text = "Convolutional Neural Network"
    subtitle_text = "Sliding kernels — extract spatial features from input"

    def animate(self):
        grid_size = 6
        cell_size = 0.45
        kernel_size = 3

        # === Step 1: Show input grid (image) ===
        input_label = Text("Input (6\u00d76)", font_size=18, color=NM_BLUE, weight=BOLD)
        input_label.move_to(LEFT * 3.5 + UP * 2.8)

        input_cells = VGroup()
        input_grid = []
        for r in range(grid_size):
            row = []
            for c in range(grid_size):
                sq = Square(
                    side_length=cell_size, stroke_width=0.5, stroke_color=NM_GRID,
                )
                # Create a simple pattern (brighter in center)
                dist = abs(r - 2.5) + abs(c - 2.5)
                opacity = max(0.1, 0.6 - dist * 0.1)
                sq.set_fill(NM_BLUE, opacity=opacity)
                sq.move_to(
                    LEFT * 3.5
                    + RIGHT * (c - grid_size / 2 + 0.5) * cell_size
                    + DOWN * (r - grid_size / 2 + 0.5) * cell_size
                    + UP * 0.5
                )
                input_cells.add(sq)
                row.append(sq)
            input_grid.append(row)

        self.play(Write(input_label), FadeIn(input_cells), run_time=0.5)
        self.wait(0.2)

        # === Step 2: Show the 3x3 kernel ===
        kernel_label = Text("3\u00d73 Kernel", font_size=16, color=NM_YELLOW, weight=BOLD)
        kernel_label.move_to(UP * 2.8)

        kernel_vals = [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1],
        ]
        kernel_cells = VGroup()
        for r in range(kernel_size):
            for c in range(kernel_size):
                sq = Square(
                    side_length=cell_size, stroke_width=1, stroke_color=NM_YELLOW,
                )
                val = kernel_vals[r][c]
                opacity = 0.5 if val > 0 else 0.15
                sq.set_fill(NM_YELLOW if val > 0 else NM_GRID, opacity=opacity)
                label = Text(str(val), font_size=10, color=NM_TEXT)
                label.move_to(sq.get_center())
                kernel_cells.add(VGroup(sq, label))

        kernel_cells.arrange_in_grid(rows=3, cols=3, buff=0)
        kernel_cells.move_to(UP * 1.2)

        self.play(Write(kernel_label), FadeIn(kernel_cells), run_time=0.4)
        self.wait(0.2)

        # === Step 3: Animate kernel sliding over input ===
        output_size = grid_size - kernel_size + 1  # 4x4
        output_label = Text("Feature Map (4\u00d74)", font_size=18, color=NM_GREEN, weight=BOLD)
        output_label.move_to(RIGHT * 3.5 + UP * 2.8)

        output_cells = VGroup()
        for r in range(output_size):
            for c in range(output_size):
                sq = Square(
                    side_length=cell_size, stroke_width=0.5, stroke_color=NM_GRID,
                )
                sq.set_fill(NM_GREEN, opacity=0.05)
                sq.move_to(
                    RIGHT * 3.5
                    + RIGHT * (c - output_size / 2 + 0.5) * cell_size
                    + DOWN * (r - output_size / 2 + 0.5) * cell_size
                    + UP * 0.5
                )
                output_cells.add(sq)

        self.play(Write(output_label), FadeIn(output_cells), run_time=0.3)

        # Sliding window highlight
        highlight = Rectangle(
            width=kernel_size * cell_size, height=kernel_size * cell_size,
            color=NM_PRIMARY, stroke_width=2.5, fill_opacity=0,
        )

        # Animate 4 positions (corners of the sliding path)
        positions = [(0, 0), (0, 3), (3, 0), (3, 3)]
        for idx, (r, c) in enumerate(positions):
            target = input_grid[r + 1][c + 1].get_center()
            if idx == 0:
                highlight.move_to(target)
                self.play(FadeIn(highlight), run_time=0.2)
            else:
                self.play(highlight.animate.move_to(target), run_time=0.3)

            # Light up corresponding output cell
            out_idx = r * output_size + c
            self.play(
                output_cells[out_idx].animate.set_fill(NM_GREEN, opacity=0.5),
                run_time=0.15,
            )

        # Fill remaining output cells
        remaining = [i for i in range(output_size ** 2)
                     if i not in [0, 3, 12, 15]]
        self.play(
            *[output_cells[i].animate.set_fill(NM_GREEN, opacity=0.35) for i in remaining],
            FadeOut(highlight),
            run_time=0.4,
        )
        self.wait(0.2)

        # === Step 4: Pooling ===
        pool_label = Text("Max Pooling 2\u00d72 \u2192 2\u00d72 output", font_size=16, color=NM_PURPLE)
        pool_label.move_to(DOWN * 2.2)
        self.play(Write(pool_label), run_time=0.3)

        # Show pooled result
        pooled = VGroup()
        for r in range(2):
            for c in range(2):
                sq = Square(
                    side_length=cell_size * 1.2, stroke_width=1.5, stroke_color=NM_PURPLE,
                )
                sq.set_fill(NM_PURPLE, opacity=0.3)
                pooled.add(sq)
        pooled.arrange_in_grid(rows=2, cols=2, buff=0.05)
        pooled.move_to(DOWN * 3.0)

        self.play(FadeIn(pooled, shift=UP * 0.15), run_time=0.3)

        result = Text("translation invariance + spatial hierarchy", font_size=14, color=NM_YELLOW, weight=BOLD)
        result.next_to(pooled, RIGHT, buff=0.4)
        self.play(FadeIn(result), run_time=0.3)
        self.wait(0.8)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.6)
