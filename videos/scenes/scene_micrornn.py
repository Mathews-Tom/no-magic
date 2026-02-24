"""
Scene: RNN vs GRU
Script: micrornn.py
Description: Recurrent networks — vanishing gradients and why gating fixed them
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE, NM_PURPLE
from manim import *


class RNNScene(NoMagicScene):
    title_text = "RNN vs GRU"
    subtitle_text = "Recurrence, vanishing gradients, and the gating solution"

    def animate(self):
        # === Step 1: Show vanilla RNN unrolled ===
        rnn_label = Text("Vanilla RNN", font_size=20, color=NM_BLUE, weight=BOLD)
        rnn_label.move_to(LEFT * 3.0 + UP * 2.8)
        self.play(Write(rnn_label), run_time=0.4)

        n_steps = 5
        rnn_cells = VGroup()
        rnn_arrows = VGroup()
        for i in range(n_steps):
            cell = RoundedRectangle(
                corner_radius=0.08, width=0.7, height=0.6,
                color=NM_BLUE, fill_opacity=0.2, stroke_width=1.5,
            )
            h_label = Text(f"h{i}", font_size=12, color=NM_TEXT)
            h_label.move_to(cell.get_center())
            rnn_cells.add(VGroup(cell, h_label))

        rnn_cells.arrange(RIGHT, buff=0.3)
        rnn_cells.move_to(LEFT * 3.0 + UP * 1.5)

        # Recurrent arrows between cells
        for i in range(n_steps - 1):
            arr = Arrow(
                rnn_cells[i].get_right(), rnn_cells[i + 1].get_left(),
                color=NM_BLUE, stroke_width=1.5, buff=0.05, tip_length=0.1,
            )
            rnn_arrows.add(arr)

        self.play(
            LaggedStart(*[FadeIn(c) for c in rnn_cells], lag_ratio=0.08),
            LaggedStart(*[GrowArrow(a) for a in rnn_arrows], lag_ratio=0.08),
            run_time=0.9,
        )

        # === Step 2: Show vanishing gradient ===
        grad_label = Text("Gradient signal", font_size=14, color=NM_PRIMARY)
        grad_label.next_to(rnn_cells, DOWN, buff=0.3)
        self.play(FadeIn(grad_label), run_time=0.4)

        # Gradient bars shrinking from right to left
        grad_bars = VGroup()
        for i in range(n_steps):
            height = 0.5 * (0.6 ** (n_steps - 1 - i))
            bar = Rectangle(
                width=0.3, height=max(0.05, height),
                color=NM_PRIMARY, fill_opacity=0.6, stroke_width=0.5,
            )
            bar.next_to(rnn_cells[i], DOWN, buff=0.5)
            bar.align_to(rnn_cells[0].get_bottom() + DOWN * 0.5, DOWN)
            grad_bars.add(bar)

        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.1) for b in grad_bars], lag_ratio=0.08),
            run_time=0.8,
        )

        vanish_text = Text("gradients vanish \u2192 can't learn long dependencies", font_size=12, color=NM_PRIMARY)
        vanish_text.next_to(grad_bars, DOWN, buff=0.15)
        self.play(FadeIn(vanish_text), run_time=0.4)
        self.wait(0.6)

        # === Step 3: GRU with gates ===
        gru_label = Text("GRU (Gated Recurrent Unit)", font_size=20, color=NM_GREEN, weight=BOLD)
        gru_label.move_to(RIGHT * 3.0 + UP * 2.8)
        self.play(Write(gru_label), run_time=0.4)

        gru_cells = VGroup()
        gru_arrows = VGroup()
        for i in range(n_steps):
            cell = RoundedRectangle(
                corner_radius=0.08, width=0.9, height=0.7,
                color=NM_GREEN, fill_opacity=0.15, stroke_width=1.5,
            )
            h_label = Text(f"h{i}", font_size=11, color=NM_TEXT)
            h_label.move_to(cell.get_center() + UP * 0.12)

            # Gate indicators
            gate_r = Text("r", font_size=9, color=NM_ORANGE)
            gate_z = Text("z", font_size=9, color=NM_YELLOW)
            gates = VGroup(gate_r, gate_z).arrange(RIGHT, buff=0.1)
            gates.move_to(cell.get_center() + DOWN * 0.15)

            gru_cells.add(VGroup(cell, h_label, gates))

        gru_cells.arrange(RIGHT, buff=0.2)
        gru_cells.move_to(RIGHT * 3.0 + UP * 1.5)

        for i in range(n_steps - 1):
            arr = Arrow(
                gru_cells[i].get_right(), gru_cells[i + 1].get_left(),
                color=NM_GREEN, stroke_width=1.5, buff=0.05, tip_length=0.1,
            )
            gru_arrows.add(arr)

        self.play(
            LaggedStart(*[FadeIn(c) for c in gru_cells], lag_ratio=0.08),
            LaggedStart(*[GrowArrow(a) for a in gru_arrows], lag_ratio=0.08),
            run_time=0.9,
        )

        # GRU gradient bars — more preserved
        gru_grad_bars = VGroup()
        for i in range(n_steps):
            height = 0.5 * (0.85 ** (n_steps - 1 - i))
            bar = Rectangle(
                width=0.3, height=max(0.05, height),
                color=NM_GREEN, fill_opacity=0.6, stroke_width=0.5,
            )
            bar.next_to(gru_cells[i], DOWN, buff=0.3)
            bar.align_to(gru_cells[0].get_bottom() + DOWN * 0.3, DOWN)
            gru_grad_bars.add(bar)

        gru_note = Text("gates preserve gradient flow", font_size=12, color=NM_GREEN)
        gru_note.next_to(gru_grad_bars, DOWN, buff=0.15)

        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.1) for b in gru_grad_bars], lag_ratio=0.08),
            FadeIn(gru_note),
            run_time=0.8,
        )
        self.wait(0.6)

        # === Step 4: Gate explanation ===
        gate_explain = VGroup(
            Text("r (reset): how much past to forget", font_size=13, color=NM_ORANGE),
            Text("z (update): how much new state to mix in", font_size=13, color=NM_YELLOW),
            Text("h' = (1-z)\u00b7h + z\u00b7candidate", font_size=13, color=NM_GREEN, weight=BOLD),
        ).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        gate_explain.move_to(DOWN * 3.2)

        self.play(
            LaggedStart(*[FadeIn(g, shift=UP * 0.1) for g in gate_explain], lag_ratio=0.15),
            run_time=0.8,
        )
        self.wait(1.6)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.9)
