"""
Scene: Activation Checkpointing
Script: microcheckpoint.py
Description: Trade compute for memory — recompute activations during backward pass
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE, NM_PURPLE
from manim import *


class CheckpointScene(NoMagicScene):
    title_text = "Activation Checkpointing"
    subtitle_text = "O(n) memory → O(√n) by recomputing activations"

    def animate(self):
        n_layers = 8

        # === Step 1: Standard backprop — all activations stored ===
        std_label = Text("Standard Backprop", font_size=18, color=NM_PRIMARY, weight=BOLD)
        std_label.move_to(LEFT * 3.5 + UP * 2.5)
        self.play(Write(std_label), run_time=0.3)

        std_cells = VGroup()
        for i in range(n_layers):
            cell = RoundedRectangle(
                corner_radius=0.06, width=0.7, height=0.5,
                color=NM_BLUE, fill_opacity=0.3, stroke_width=1.5,
            )
            label = Text(f"a{i}", font_size=10, color=NM_TEXT)
            label.move_to(cell.get_center())
            std_cells.add(VGroup(cell, label))

        std_cells.arrange(RIGHT, buff=0.08)
        std_cells.move_to(LEFT * 3.5 + UP * 1.3)

        mem_label = Text("Memory: O(n) — store ALL activations", font_size=12, color=NM_PRIMARY)
        mem_label.next_to(std_cells, DOWN, buff=0.15)

        self.play(
            LaggedStart(*[FadeIn(c) for c in std_cells], lag_ratio=0.06),
            run_time=0.4,
        )
        self.play(FadeIn(mem_label), run_time=0.2)
        self.wait(0.2)

        # === Step 2: Checkpointed — only sqrt(n) stored ===
        ckpt_label = Text("Checkpointed", font_size=18, color=NM_GREEN, weight=BOLD)
        ckpt_label.move_to(RIGHT * 3.5 + UP * 2.5)
        self.play(Write(ckpt_label), run_time=0.3)

        # checkpoint every 2 layers (sqrt(8) ≈ 2.8)
        ckpt_cells = VGroup()
        for i in range(n_layers):
            is_checkpoint = (i % 3 == 0)
            color = NM_GREEN if is_checkpoint else NM_GRID
            opacity = 0.3 if is_checkpoint else 0.05
            cell = RoundedRectangle(
                corner_radius=0.06, width=0.7, height=0.5,
                color=color, fill_opacity=opacity, stroke_width=1.5 if is_checkpoint else 0.5,
            )
            label = Text(f"a{i}", font_size=10, color=NM_TEXT if is_checkpoint else NM_GRID)
            label.move_to(cell.get_center())
            if is_checkpoint:
                check = Text("✓", font_size=8, color=NM_GREEN)
                check.next_to(cell, UP, buff=0.03)
                ckpt_cells.add(VGroup(cell, label, check))
            else:
                ckpt_cells.add(VGroup(cell, label))

        ckpt_cells.arrange(RIGHT, buff=0.08)
        ckpt_cells.move_to(RIGHT * 3.5 + UP * 1.3)

        ckpt_mem = Text("Memory: O(√n) — store only checkpoints", font_size=12, color=NM_GREEN)
        ckpt_mem.next_to(ckpt_cells, DOWN, buff=0.15)

        self.play(
            LaggedStart(*[FadeIn(c) for c in ckpt_cells], lag_ratio=0.06),
            run_time=0.4,
        )
        self.play(FadeIn(ckpt_mem), run_time=0.2)
        self.wait(0.2)

        # === Step 3: Animate recomputation during backward ===
        recomp_label = Text("Backward Pass: Recompute Between Checkpoints", font_size=16, color=NM_YELLOW, weight=BOLD)
        recomp_label.move_to(DOWN * 0.8)
        self.play(Write(recomp_label), run_time=0.3)

        # Highlight a segment being recomputed
        highlight = SurroundingRectangle(
            VGroup(ckpt_cells[3], ckpt_cells[4], ckpt_cells[5]),
            color=NM_ORANGE, buff=0.08, stroke_width=2,
        )
        recomp_text = Text("recompute a3→a4→a5", font_size=11, color=NM_ORANGE)
        recomp_text.next_to(highlight, UP, buff=0.1)

        self.play(Create(highlight), FadeIn(recomp_text), run_time=0.3)

        # Flash the recomputed cells
        for idx in [3, 4, 5]:
            self.play(
                ckpt_cells[idx][0].animate.set_fill(NM_ORANGE, opacity=0.3),
                run_time=0.15,
            )
            self.play(
                ckpt_cells[idx][0].animate.set_fill(NM_GREEN if idx % 3 == 0 else NM_GRID, opacity=0.3 if idx % 3 == 0 else 0.05),
                run_time=0.1,
            )
        self.wait(0.2)

        # === Step 4: Trade-off summary ===
        tradeoff = VGroup(
            Text("Trade-off: 1 extra forward pass (recomputation)", font_size=14, color=NM_TEXT),
            Text("Benefit: 2-3x less memory → train deeper models", font_size=14, color=NM_GREEN, weight=BOLD),
            Text("Used by: GPT-3, LLaMA, all modern large model training", font_size=12, color=NM_GRID),
        ).arrange(DOWN, buff=0.08)
        tradeoff.move_to(DOWN * 2.5)

        self.play(
            LaggedStart(*[FadeIn(t, shift=UP * 0.1) for t in tradeoff], lag_ratio=0.15),
            run_time=0.5,
        )
        self.wait(0.8)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.6)
