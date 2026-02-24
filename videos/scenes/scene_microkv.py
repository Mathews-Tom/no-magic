"""
Scene: KV-Cache
Script: microkv.py
Description: Why autoregressive inference recomputes redundant work, and how the KV cache fixes it
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE, NM_PURPLE
from manim import *


class KVCacheScene(NoMagicScene):
    title_text = "KV-Cache"
    subtitle_text = "Memoize keys and values — stop recomputing the past"

    def animate(self):
        # === Step 1: Without cache — recompute everything each step ===
        no_cache_label = Text("Without KV-Cache", font_size=20, color=NM_PRIMARY, weight=BOLD)
        no_cache_label.move_to(LEFT * 3.5 + UP * 2.8)
        self.play(Write(no_cache_label), run_time=0.4)

        tokens_left = ["h", "e", "l", "l", "o"]
        cell_w, cell_h = 0.5, 0.4

        # Animate 4 generation steps (generating tokens 2-5)
        for step in range(1, 5):
            active = tokens_left[:step + 1]
            row = VGroup()
            for i, tok in enumerate(active):
                box = Rectangle(
                    width=cell_w, height=cell_h,
                    color=NM_PRIMARY if i < step else NM_YELLOW,
                    fill_opacity=0.25 if i < step else 0.4,
                    stroke_width=1.5,
                )
                label = Text(tok, font_size=14, color=NM_TEXT)
                label.move_to(box.get_center())
                row.add(VGroup(box, label))
            row.arrange(RIGHT, buff=0.05)
            row.move_to(LEFT * 3.5 + UP * (1.8 - step * 0.65))

            # Mark recomputed tokens
            if step > 1:
                recomp_bracket = Brace(VGroup(*row[:step]), DOWN, color=NM_PRIMARY, buff=0.05)
                recomp_text = Text("recomputed", font_size=10, color=NM_PRIMARY)
                recomp_text.next_to(recomp_bracket, DOWN, buff=0.05)
                self.play(FadeIn(row), FadeIn(recomp_bracket), FadeIn(recomp_text), run_time=0.4)
            else:
                self.play(FadeIn(row), run_time=0.4)

        self.wait(0.6)

        # === Step 2: With cache — only compute new token's K,V ===
        cache_label = Text("With KV-Cache", font_size=20, color=NM_GREEN, weight=BOLD)
        cache_label.move_to(RIGHT * 3.5 + UP * 2.8)
        self.play(Write(cache_label), run_time=0.4)

        # Show the cache growing incrementally
        cache_box = RoundedRectangle(
            corner_radius=0.1, width=3.5, height=2.5,
            color=NM_GREEN, fill_opacity=0.05, stroke_width=1.5,
        )
        cache_box.move_to(RIGHT * 3.5 + DOWN * 0.2)
        cache_title = Text("K,V Cache", font_size=14, color=NM_GREEN)
        cache_title.next_to(cache_box, UP, buff=0.1)

        self.play(FadeIn(cache_box), FadeIn(cache_title), run_time=0.4)

        cache_rows = VGroup()
        for step in range(5):
            tok = tokens_left[step]
            is_new = True  # each step adds exactly one

            kv_row = VGroup()
            # K cell
            k_box = Rectangle(
                width=0.8, height=cell_h,
                color=NM_BLUE, fill_opacity=0.3, stroke_width=1,
            )
            k_label = Text(f"K({tok})", font_size=11, color=NM_TEXT)
            k_label.move_to(k_box.get_center())
            kv_row.add(VGroup(k_box, k_label))

            # V cell
            v_box = Rectangle(
                width=0.8, height=cell_h,
                color=NM_PURPLE, fill_opacity=0.3, stroke_width=1,
            )
            v_label = Text(f"V({tok})", font_size=11, color=NM_TEXT)
            v_label.move_to(v_box.get_center())
            kv_row.add(VGroup(v_box, v_label))

            kv_row.arrange(RIGHT, buff=0.15)
            cache_rows.add(kv_row)

        cache_rows.arrange(DOWN, buff=0.08)
        cache_rows.move_to(cache_box.get_center())

        # Animate cache filling one row at a time
        for i, row in enumerate(cache_rows):
            new_label = Text("new", font_size=10, color=NM_YELLOW, weight=BOLD)
            new_label.next_to(row, RIGHT, buff=0.15)
            self.play(FadeIn(row, shift=LEFT * 0.2), FadeIn(new_label), run_time=0.4)
            if i < len(cache_rows) - 1:
                self.play(FadeOut(new_label), run_time=0.4)
            else:
                self.play(FadeOut(new_label), run_time=0.4)

        self.wait(0.6)

        # === Step 3: Comparison summary ===
        comparison = VGroup(
            Text("Without cache: O(n\u00b2) compute per token", font_size=16, color=NM_PRIMARY),
            Text("With cache:    O(n) compute per token", font_size=16, color=NM_GREEN),
            Text("Cached K,V → only compute Q for new token", font_size=16, color=NM_YELLOW, weight=BOLD),
        ).arrange(DOWN, buff=0.15)
        comparison.move_to(DOWN * 3.0)

        self.play(
            LaggedStart(*[FadeIn(c, shift=UP * 0.15) for c in comparison], lag_ratio=0.15),
            run_time=0.9,
        )
        self.wait(1.6)

        # Cleanup
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.9)
