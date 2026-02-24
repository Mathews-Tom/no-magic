"""
Scene: Flash Attention
Script: microflash.py
Description: Tiled attention — processing Q·K^T in blocks without materializing the full NxN matrix
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE
from manim import *


class FlashAttentionScene(NoMagicScene):
    title_text = "Flash Attention"
    subtitle_text = "Tiled computation — never materialize the full N×N matrix"

    def animate(self):
        grid_size = 8
        cell_size = 0.38
        gap = 0.02

        # === Step 1: Show the full attention matrix (standard approach) ===
        std_label = Text("Standard Attention", font_size=22, color=NM_PRIMARY, weight=BOLD)
        std_label.move_to(LEFT * 3.5 + UP * 2.8)

        # Full NxN grid
        std_cells = VGroup()
        for r in range(grid_size):
            for c in range(grid_size):
                sq = Square(
                    side_length=cell_size, stroke_width=0.5, stroke_color=NM_GRID,
                )
                sq.set_fill(NM_PRIMARY, opacity=0.3)
                sq.move_to(
                    LEFT * 3.5
                    + RIGHT * (c - grid_size / 2 + 0.5) * (cell_size + gap)
                    + DOWN * (r - grid_size / 2 + 0.5) * (cell_size + gap)
                )
                std_cells.add(sq)

        mem_label = Text("O(N²) memory", font_size=16, color=NM_PRIMARY)
        mem_label.next_to(std_cells, DOWN, buff=0.25)

        self.play(Write(std_label), run_time=0.4)
        self.play(FadeIn(std_cells), run_time=0.8)
        self.play(Write(mem_label), run_time=0.4)

        # Highlight all cells simultaneously to show full materialization
        self.play(
            *[cell.animate.set_fill(NM_PRIMARY, opacity=0.6) for cell in std_cells],
            run_time=0.8,
        )
        self.play(
            *[cell.animate.set_fill(NM_PRIMARY, opacity=0.3) for cell in std_cells],
            run_time=0.4,
        )
        self.wait(0.6)

        # === Step 2: Show Flash Attention tiled grid ===
        flash_label = Text("Flash Attention", font_size=22, color=NM_GREEN, weight=BOLD)
        flash_label.move_to(RIGHT * 3.5 + UP * 2.8)

        # Same grid but divided into 4 tile blocks (2x2 tiles of 4x4 each)
        block_size = 4
        n_blocks = grid_size // block_size

        flash_cells = VGroup()
        tile_groups = {}  # (br, bc) -> list of cells
        for r in range(grid_size):
            for c in range(grid_size):
                sq = Square(
                    side_length=cell_size, stroke_width=0.5, stroke_color=NM_GRID,
                )
                sq.set_fill(NM_BLUE, opacity=0.15)
                sq.move_to(
                    RIGHT * 3.5
                    + RIGHT * (c - grid_size / 2 + 0.5) * (cell_size + gap)
                    + DOWN * (r - grid_size / 2 + 0.5) * (cell_size + gap)
                )
                flash_cells.add(sq)
                br, bc = r // block_size, c // block_size
                tile_groups.setdefault((br, bc), []).append(sq)

        # Draw tile boundaries
        tile_borders = VGroup()
        for br in range(n_blocks):
            for bc in range(n_blocks):
                tile_cells = tile_groups[(br, bc)]
                border = SurroundingRectangle(
                    VGroup(*tile_cells), color=NM_GREEN, buff=0.02, stroke_width=1.5,
                )
                border.set_stroke(opacity=0.6)
                tile_borders.add(border)

        flash_mem = Text("O(N) memory", font_size=16, color=NM_GREEN)
        flash_mem.next_to(flash_cells, DOWN, buff=0.25)

        self.play(Write(flash_label), run_time=0.4)
        self.play(FadeIn(flash_cells), FadeIn(tile_borders), run_time=0.8)
        self.play(Write(flash_mem), run_time=0.4)
        self.wait(0.6)

        # === Step 3: Animate tile-by-tile processing ===
        sram_label = Text("SRAM (fast memory)", font_size=14, color=NM_YELLOW)
        sram_label.next_to(flash_cells, RIGHT, buff=0.4).shift(UP * 0.5)

        sram_box = RoundedRectangle(
            corner_radius=0.1, width=1.6, height=1.6,
            color=NM_YELLOW, fill_opacity=0.08, stroke_width=1.5,
        )
        sram_box.next_to(sram_label, DOWN, buff=0.15)
        self.play(FadeIn(sram_label), FadeIn(sram_box), run_time=0.4)

        # Process tiles sequentially: light up one tile at a time
        tile_order = [(0, 0), (0, 1), (1, 0), (1, 1)]
        tile_colors = [NM_GREEN, NM_YELLOW, NM_ORANGE, NM_PRIMARY]

        for idx, (br, bc) in enumerate(tile_order):
            cells = tile_groups[(br, bc)]
            color = tile_colors[idx]

            # Light up tile
            self.play(
                *[cell.animate.set_fill(color, opacity=0.6) for cell in cells],
                sram_box.animate.set_fill(color, opacity=0.2),
                run_time=0.5,
            )

            # Show tile label
            tile_id = Text(f"tile ({br},{bc})", font_size=12, color=color)
            tile_id.move_to(sram_box.get_center())
            self.play(FadeIn(tile_id), run_time=0.4)
            self.wait(0.4)

            # Dim tile (processed) and clear SRAM label
            self.play(
                *[cell.animate.set_fill(NM_GREEN, opacity=0.35) for cell in cells],
                sram_box.animate.set_fill(NM_YELLOW, opacity=0.05),
                FadeOut(tile_id),
                run_time=0.4,
            )

        self.wait(0.6)

        # === Step 4: Result comparison ===
        result_text = VGroup(
            Text("Same exact output", font_size=18, color=NM_TEXT, weight=BOLD),
            Text("4x less memory", font_size=16, color=NM_GREEN),
        ).arrange(DOWN, buff=0.12)
        result_text.to_edge(DOWN, buff=0.6)

        self.play(
            LaggedStart(*[FadeIn(t, shift=UP * 0.2) for t in result_text], lag_ratio=0.2),
            run_time=0.8,
        )
        self.wait(1.6)

        # Cleanup
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.9)
