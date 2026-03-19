"""
Scene: Memory-Augmented Network
Script: micromemory.py
Description: Differentiable read/write heads — learning where to store and retrieve
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE, NM_PURPLE
from manim import *

MEMORY_ROWS = 6
MEMORY_COLS = 6
CELL_SIZE = 0.52
CELL_BUFF = 0.06


def make_memory_grid(rows=MEMORY_ROWS, cols=MEMORY_COLS):
    """Build a VGroup of Square cells representing the memory matrix."""
    grid = VGroup()
    for r in range(rows):
        for c in range(cols):
            cell = Square(
                side_length=CELL_SIZE,
                color=NM_GRID,
                fill_color=NM_BG,
                fill_opacity=1.0,
                stroke_width=1.5,
            )
            cell.move_to(
                RIGHT * c * (CELL_SIZE + CELL_BUFF) + DOWN * r * (CELL_SIZE + CELL_BUFF)
            )
            grid.add(cell)
    grid.center()
    return grid


def make_controller_box(label="Controller\n(MLP)", width=2.0, height=1.1):
    box = RoundedRectangle(
        corner_radius=0.15, width=width, height=height,
        color=NM_BLUE, fill_color=NM_BLUE, fill_opacity=0.25, stroke_width=2,
    )
    text = Text(label, font_size=16, color=NM_TEXT, weight=BOLD)
    return VGroup(box, text)


def make_label(text, font_size=16, color=NM_TEXT):
    return Text(text, font_size=font_size, color=color)


# Import NM_BG for fill
from base import NM_BG


class MemoryScene(NoMagicScene):
    title_text = "Memory-Augmented Network"
    subtitle_text = "Differentiable read/write heads — learning where to store and retrieve"

    def animate(self):
        # === Step 1: Display the external memory matrix ===
        grid = make_memory_grid()
        grid.move_to(RIGHT * 2.2)

        mem_label = Text("External Memory  (M rows × D dims)", font_size=18, color=NM_PURPLE, weight=BOLD)
        mem_label.next_to(grid, UP, buff=0.3)

        row_labels = VGroup()
        for r in range(MEMORY_ROWS):
            lbl = Text(f"M[{r}]", font_size=13, color=NM_GRID)
            cell_0 = grid[r * MEMORY_COLS]
            lbl.next_to(cell_0, LEFT, buff=0.18)
            row_labels.add(lbl)

        border = SurroundingRectangle(grid, color=NM_PURPLE, buff=0.12, stroke_width=2)

        self.play(
            FadeIn(grid, shift=DOWN * 0.15),
            Write(mem_label),
            run_time=0.9,
        )
        self.play(Create(border), FadeIn(row_labels), run_time=0.6)
        self.wait(0.6)

        # === Step 2: Controller architecture with read/write heads ===
        controller = make_controller_box()
        controller.move_to(LEFT * 3.5 + UP * 0.2)

        input_dot = Dot(color=NM_PRIMARY, radius=0.08)
        input_dot.move_to(LEFT * 5.5 + UP * 0.2)
        input_arrow = Arrow(
            input_dot.get_right(), controller.get_left(),
            buff=0.08, color=NM_TEXT, stroke_width=2,
        )
        input_lbl = Text("input x", font_size=13, color=NM_TEXT)
        input_lbl.next_to(input_arrow, UP, buff=0.1)

        output_dot = Dot(color=NM_GREEN, radius=0.08)
        output_dot.move_to(LEFT * 1.8 + UP * 0.2)
        output_arrow = Arrow(
            controller.get_right(), output_dot.get_left(),
            buff=0.08, color=NM_TEXT, stroke_width=2,
        )
        output_lbl = Text("output y", font_size=13, color=NM_TEXT)
        output_lbl.next_to(output_arrow, UP, buff=0.1)

        # Write head: controller → memory (top-right corner of grid)
        write_head_end = border.get_left() + UP * 0.5
        write_arrow = CurvedArrow(
            controller.get_right() + UP * 0.25,
            write_head_end,
            angle=-TAU / 8,
            color=NM_GREEN,
            stroke_width=2,
        )
        write_lbl = Text("Write Head", font_size=14, color=NM_GREEN, weight=BOLD)
        write_lbl.move_to(write_arrow.get_center() + UP * 0.3)

        # Read head: memory → controller (bottom-left of grid)
        read_head_start = border.get_left() + DOWN * 0.5
        read_arrow = CurvedArrow(
            read_head_start,
            controller.get_right() + DOWN * 0.25,
            angle=-TAU / 8,
            color=NM_ORANGE,
            stroke_width=2,
        )
        read_lbl = Text("Read Head", font_size=14, color=NM_ORANGE, weight=BOLD)
        read_lbl.move_to(read_arrow.get_center() + DOWN * 0.35)

        self.play(
            FadeIn(controller, shift=RIGHT * 0.15),
            FadeIn(input_dot), GrowArrow(input_arrow), FadeIn(input_lbl),
            run_time=0.8,
        )
        self.play(
            GrowArrow(output_arrow), FadeIn(output_dot), FadeIn(output_lbl),
            run_time=0.6,
        )
        self.play(
            Create(write_arrow), FadeIn(write_lbl),
            Create(read_arrow), FadeIn(read_lbl),
            run_time=0.9,
        )
        self.wait(0.8)

        # === Step 3: Content-based addressing — the write phase ===
        arch_group = VGroup(
            controller, input_dot, input_arrow, input_lbl,
            output_arrow, output_dot, output_lbl,
            write_arrow, write_lbl, read_arrow, read_lbl,
        )
        self.play(
            arch_group.animate.scale(0.55).to_corner(UL, buff=0.25),
            grid.animate.move_to(RIGHT * 1.8),
            mem_label.animate.next_to(grid, UP, buff=0.25).scale(0.85),
            border.animate.move_to(RIGHT * 1.8),
            row_labels.animate.shift(LEFT * 0.0),  # will reposition below
            run_time=0.8,
        )

        # Reposition row labels to match shifted grid
        row_labels_new = VGroup()
        for r in range(MEMORY_ROWS):
            lbl = Text(f"M[{r}]", font_size=13, color=NM_GRID)
            cell_0 = grid[r * MEMORY_COLS]
            lbl.next_to(cell_0, LEFT, buff=0.18)
            row_labels_new.add(lbl)
        self.remove(row_labels)
        self.add(row_labels_new)

        addressing_title = Text("Content-Based Addressing  (Write Phase)", font_size=18, color=NM_YELLOW, weight=BOLD)
        addressing_title.to_edge(DOWN, buff=2.0)
        self.play(Write(addressing_title), run_time=0.5)

        # Key vector appears
        key_box = RoundedRectangle(
            corner_radius=0.1, width=1.5, height=0.55,
            color=NM_YELLOW, fill_color=NM_YELLOW, fill_opacity=0.2, stroke_width=2,
        )
        key_text = Text("key k", font_size=15, color=NM_YELLOW, weight=BOLD)
        key_group = VGroup(key_box, key_text)
        key_group.move_to(LEFT * 3.2 + UP * 0.5)

        self.play(FadeIn(key_group, shift=RIGHT * 0.2), run_time=0.5)

        # Cosine similarity: show attention weights per row
        # Row 1 (index 1) lights up brightest
        target_row = 1
        weights = [0.05, 0.75, 0.08, 0.05, 0.04, 0.03]

        weight_bars = VGroup()
        for r in range(MEMORY_ROWS):
            w = weights[r]
            bar = Rectangle(
                width=w * 1.4, height=CELL_SIZE * 0.55,
                color=NM_YELLOW, fill_color=NM_YELLOW, fill_opacity=w,
                stroke_width=1,
            )
            cell_0 = grid[r * MEMORY_COLS]
            bar.next_to(cell_0, LEFT, buff=0.05)
            weight_bars.add(bar)

        sim_lbl = Text("cos(k, M[r])  →  attention weights", font_size=14, color=NM_YELLOW)
        sim_lbl.to_edge(DOWN, buff=1.3)

        self.play(
            *[FadeIn(wb, shift=RIGHT * 0.1) for wb in weight_bars],
            FadeIn(sim_lbl),
            run_time=0.7,
        )
        self.wait(0.4)

        # Write: addressed row changes color (filled with data)
        target_cells = [grid[target_row * MEMORY_COLS + c] for c in range(MEMORY_COLS)]
        self.play(
            *[cell.animate.set_fill(NM_GREEN, opacity=0.7) for cell in target_cells],
            run_time=0.6,
        )
        write_confirm = Text("✓ written to M[1]", font_size=14, color=NM_GREEN, weight=BOLD)
        write_confirm.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(write_confirm, shift=UP * 0.15), run_time=0.4)
        self.wait(0.5)

        # === Step 4: Sequence of 3 writes — slots light up sequentially ===
        self.play(
            FadeOut(weight_bars), FadeOut(sim_lbl), FadeOut(write_confirm),
            FadeOut(addressing_title), FadeOut(key_group),
            run_time=0.5,
        )

        seq_title = Text("Writing a Sequence  (3 inputs → 3 slots)", font_size=18, color=NM_GREEN, weight=BOLD)
        seq_title.to_edge(DOWN, buff=2.0)
        self.play(Write(seq_title), run_time=0.4)

        slot_colors = [NM_GREEN, NM_BLUE, NM_ORANGE]
        input_labels = ["x₀", "x₁", "x₂"]
        slot_indices = [0, 1, 2]

        for slot, color, inp_lbl in zip(slot_indices, slot_colors, input_labels):
            # Show input arriving
            inp_box = RoundedRectangle(
                corner_radius=0.1, width=1.1, height=0.5,
                color=color, fill_color=color, fill_opacity=0.2, stroke_width=2,
            )
            inp_txt = Text(inp_lbl, font_size=16, color=color, weight=BOLD)
            inp_group = VGroup(inp_box, inp_txt)
            inp_group.move_to(LEFT * 3.5 + UP * 0.5)

            # Show addressing weights with peak at current slot
            slot_weights = [0.04] * MEMORY_ROWS
            slot_weights[slot] = 0.80

            w_bars = VGroup()
            for r in range(MEMORY_ROWS):
                w = slot_weights[r]
                bar = Rectangle(
                    width=w * 1.2, height=CELL_SIZE * 0.55,
                    color=NM_YELLOW, fill_color=NM_YELLOW, fill_opacity=w,
                    stroke_width=1,
                )
                cell_0 = grid[r * MEMORY_COLS]
                bar.next_to(cell_0, LEFT, buff=0.05)
                w_bars.add(bar)

            self.play(FadeIn(inp_group, shift=RIGHT * 0.2), run_time=0.35)
            self.play(FadeIn(w_bars, shift=RIGHT * 0.05), run_time=0.4)

            slot_cells = [grid[slot * MEMORY_COLS + c] for c in range(MEMORY_COLS)]
            self.play(
                *[cell.animate.set_fill(color, opacity=0.7) for cell in slot_cells],
                run_time=0.45,
            )
            self.play(FadeOut(inp_group), FadeOut(w_bars), run_time=0.3)

        self.wait(0.5)

        # === Step 5: Read-back phase — the copy task ===
        self.play(FadeOut(seq_title), run_time=0.4)

        read_title = Text("Read-Back Phase  (delimiter → retrieve sequence)", font_size=18, color=NM_ORANGE, weight=BOLD)
        read_title.to_edge(DOWN, buff=2.0)
        self.play(Write(read_title), run_time=0.5)

        # Delimiter signal
        delim_box = RoundedRectangle(
            corner_radius=0.1, width=1.5, height=0.55,
            color=NM_PRIMARY, fill_color=NM_PRIMARY, fill_opacity=0.2, stroke_width=2,
        )
        delim_txt = Text("delimiter", font_size=15, color=NM_PRIMARY, weight=BOLD)
        delim_group = VGroup(delim_box, delim_txt)
        delim_group.move_to(LEFT * 3.5 + UP * 1.2)

        self.play(FadeIn(delim_group, shift=DOWN * 0.15), run_time=0.4)
        self.wait(0.3)

        # Read each slot and show output match
        read_colors = [NM_GREEN, NM_BLUE, NM_ORANGE]
        read_labels = ["x₀", "x₁", "x₂"]
        checkmarks = VGroup()

        for slot, color, lbl in zip([0, 1, 2], read_colors, read_labels):
            # Attention weights peak at this slot
            slot_weights = [0.04] * MEMORY_ROWS
            slot_weights[slot] = 0.82

            w_bars = VGroup()
            for r in range(MEMORY_ROWS):
                w = slot_weights[r]
                bar = Rectangle(
                    width=w * 1.2, height=CELL_SIZE * 0.55,
                    color=NM_ORANGE, fill_color=NM_ORANGE, fill_opacity=w,
                    stroke_width=1,
                )
                cell_0 = grid[r * MEMORY_COLS]
                bar.next_to(cell_0, LEFT, buff=0.05)
                w_bars.add(bar)

            # Output box to the right of grid
            out_box = RoundedRectangle(
                corner_radius=0.1, width=1.0, height=0.48,
                color=color, fill_color=color, fill_opacity=0.2, stroke_width=2,
            )
            out_txt = Text(lbl, font_size=15, color=color, weight=BOLD)
            out_group = VGroup(out_box, out_txt)
            right_edge = grid.get_right()
            out_group.move_to(right_edge + RIGHT * 1.0 + DOWN * (slot - 1) * 0.7)

            # Highlight read row
            slot_cells = [grid[slot * MEMORY_COLS + c] for c in range(MEMORY_COLS)]

            self.play(FadeIn(w_bars, shift=RIGHT * 0.05), run_time=0.35)
            self.play(
                *[Indicate(cell, color=NM_ORANGE, scale_factor=1.1) for cell in slot_cells],
                run_time=0.45,
            )
            self.play(FadeIn(out_group, shift=RIGHT * 0.15), FadeOut(w_bars), run_time=0.4)

            check = Text("✓", font_size=18, color=NM_GREEN, weight=BOLD)
            check.next_to(out_group, RIGHT, buff=0.15)
            checkmarks.add(check)
            self.play(FadeIn(check, shift=LEFT * 0.1), run_time=0.3)

        self.wait(0.5)

        # === Step 6: Key insight — addressing IS attention ===
        self.play(
            FadeOut(delim_group), FadeOut(read_title), FadeOut(checkmarks),
            run_time=0.5,
        )

        # Clear everything except arch_group hint
        self.play(*[FadeOut(mob) for mob in self.mobjects if mob not in [arch_group]], run_time=0.6)

        insight_title = Text(
            "Content-based addressing IS attention over memory",
            font_size=22, color=NM_YELLOW, weight=BOLD,
        )
        insight_title.move_to(UP * 2.4)

        parallel_left = Text(
            "Transformer attention:\nQ·Kᵀ / √d",
            font_size=18, color=NM_BLUE,
        )
        parallel_left.move_to(LEFT * 2.8 + UP * 0.9)

        vs_text = Text("≡", font_size=36, color=NM_TEXT)
        vs_text.move_to(UP * 0.9)

        parallel_right = Text(
            "NTM addressing:\ncos(key, M[i])",
            font_size=18, color=NM_ORANGE,
        )
        parallel_right.move_to(RIGHT * 2.8 + UP * 0.9)

        formula = MathTex(
            r"w_i = \text{softmax}\!\left(\beta \cdot \frac{k \cdot M_i}{\|k\|\,\|M_i\|}\right)",
            font_size=30, color=NM_YELLOW,
        )
        formula.move_to(DOWN * 0.4)

        self.play(Write(insight_title), run_time=0.8)
        self.play(
            FadeIn(parallel_left, shift=RIGHT * 0.2),
            FadeIn(vs_text),
            FadeIn(parallel_right, shift=LEFT * 0.2),
            run_time=0.8,
        )
        self.play(Write(formula), run_time=1.0)
        self.wait(1.0)

        # === Step 7: Summary and fade out ===
        self.play(
            FadeOut(insight_title), FadeOut(parallel_left),
            FadeOut(vs_text), FadeOut(parallel_right), FadeOut(formula),
            run_time=0.6,
        )

        summary = Text(
            "Static retrieval (RAG)  →  Dynamic memory (NTM)  →  Persistent agents (MemGPT)",
            font_size=19, color=NM_GREEN, weight=BOLD,
        )
        summary.move_to(ORIGIN)

        self.play(Write(summary), run_time=1.0)
        self.wait(1.5)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.9)
