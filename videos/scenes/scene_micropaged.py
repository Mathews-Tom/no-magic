"""
Scene: PagedAttention (vLLM)
Script: micropaged.py
Description: OS-style paged memory for KV-caches enables serving thousands of requests
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE, NM_PURPLE
from manim import *


class PagedScene(NoMagicScene):
    title_text = "PagedAttention"
    subtitle_text = "OS-style paged memory for KV-cache serving"

    def animate(self):
        # === Step 1: Show contiguous allocation problem ===
        contig_label = Text("Contiguous Allocation", font_size=18, color=NM_PRIMARY, weight=BOLD)
        contig_label.move_to(LEFT * 3.5 + UP * 2.5)
        self.play(Write(contig_label), run_time=0.4)

        # Memory bar with 3 requests — fragmented
        mem_bar = VGroup()
        colors = [NM_BLUE, NM_GREEN, NM_YELLOW]
        labels = ["Req A", "Req B", "Req C"]
        widths = [2.0, 1.2, 0.8]  # pre-allocated max lengths

        x_pos = -5.5
        for i, (w, c, lab) in enumerate(zip(widths, colors, labels)):
            block = Rectangle(width=w, height=0.5, color=c, fill_opacity=0.3, stroke_width=1.5)
            block.move_to([x_pos + w / 2, 1.3, 0])
            text = Text(lab, font_size=10, color=c)
            text.move_to(block.get_center())
            mem_bar.add(VGroup(block, text))
            x_pos += w + 0.15

        # Wasted space
        waste = Rectangle(width=1.5, height=0.5, color=NM_PRIMARY, fill_opacity=0.1, stroke_width=1, stroke_opacity=0.5)
        waste.move_to([x_pos + 0.75, 1.3, 0])
        waste_label = Text("wasted", font_size=9, color=NM_PRIMARY)
        waste_label.move_to(waste.get_center())

        waste_note = Text("pre-allocate max length → internal fragmentation", font_size=12, color=NM_PRIMARY)
        waste_note.move_to(LEFT * 3.5 + UP * 0.5)

        self.play(
            LaggedStart(*[FadeIn(b) for b in mem_bar], lag_ratio=0.1),
            FadeIn(waste), FadeIn(waste_label),
            run_time=0.6,
        )
        self.play(FadeIn(waste_note), run_time=0.4)
        self.wait(0.4)

        # === Step 2: Paged allocation ===
        paged_label = Text("PagedAttention", font_size=18, color=NM_GREEN, weight=BOLD)
        paged_label.move_to(RIGHT * 3.5 + UP * 2.5)
        self.play(Write(paged_label), run_time=0.4)

        # Physical memory blocks — 4x4 grid of pages
        page_grid = VGroup()
        page_size = 0.45
        page_assignments = [
            # (row, col, color, request_label)
            (0, 0, NM_BLUE, "A"), (0, 1, NM_BLUE, "A"), (0, 2, NM_GREEN, "B"),
            (0, 3, NM_YELLOW, "C"), (1, 0, NM_BLUE, "A"), (1, 1, NM_GREEN, "B"),
            (1, 2, None, ""), (1, 3, None, ""),
            (2, 0, None, ""), (2, 1, None, ""), (2, 2, None, ""), (2, 3, None, ""),
        ]

        for row, col, color, req in page_assignments:
            x = 2.0 + col * (page_size + 0.05)
            y = 1.3 - row * (page_size + 0.05)
            page = Rectangle(
                width=page_size, height=page_size,
                color=color if color else NM_GRID,
                fill_opacity=0.25 if color else 0.03,
                stroke_width=1.5 if color else 0.5,
            )
            page.move_to([x, y, 0])
            if req:
                lab = Text(req, font_size=8, color=color)
                lab.move_to(page.get_center())
                page_grid.add(VGroup(page, lab))
            else:
                page_grid.add(page)

        page_note = Text("non-contiguous pages → zero waste", font_size=12, color=NM_GREEN)
        page_note.move_to(RIGHT * 3.5 + DOWN * 0.5)

        block_table_label = Text("Block Table maps virtual → physical", font_size=11, color=NM_GRID)
        block_table_label.move_to(RIGHT * 3.5 + DOWN * 0.9)

        self.play(
            LaggedStart(*[FadeIn(p) for p in page_grid], lag_ratio=0.03),
            run_time=0.8,
        )
        self.play(FadeIn(page_note), FadeIn(block_table_label), run_time=0.4)
        self.wait(0.4)

        # === Step 3: Dynamic growth — new token arrives ===
        grow_label = Text("New token → allocate next free page", font_size=16, color=NM_YELLOW, weight=BOLD)
        grow_label.move_to(DOWN * 1.5)
        self.play(Write(grow_label), run_time=0.4)

        # Highlight an empty page being claimed
        target_page = page_grid[6]  # first empty page
        self.play(
            target_page.animate.set_fill(NM_BLUE, opacity=0.25).set_stroke(NM_BLUE, width=1.5),
            run_time=0.4,
        )
        self.wait(0.4)

        # === Step 4: Benefits ===
        benefits = VGroup(
            Text("near-zero memory waste (< 4% fragmentation)", font_size=14, color=NM_GREEN, weight=BOLD),
            Text("2-4x more concurrent requests vs contiguous", font_size=14, color=NM_YELLOW),
            Text("copy-on-write for beam search / parallel sampling", font_size=12, color=NM_GRID),
        ).arrange(DOWN, buff=0.08)
        benefits.move_to(DOWN * 2.8)

        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.1) for b in benefits], lag_ratio=0.15),
            run_time=0.8,
        )
        self.wait(1.6)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.9)
