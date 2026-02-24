"""
Scene: Byte-Pair Encoding Tokenizer
Script: microtokenizer.py
Description: Iterative pair merging — how text becomes a learned vocabulary
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE, NM_PURPLE
from manim import *


class TokenizerScene(NoMagicScene):
    title_text = "Byte-Pair Encoding"
    subtitle_text = "Iterative merging — how text becomes tokens"

    def animate(self):
        # === Step 1: Show raw text as individual characters ===
        raw_text = "l o w e r"
        chars = raw_text.split()
        char_boxes = VGroup()
        for ch in chars:
            box = RoundedRectangle(
                corner_radius=0.08, width=0.6, height=0.55,
                color=NM_BLUE, fill_opacity=0.2, stroke_width=1.5,
            )
            label = Text(ch, font_size=22, color=NM_TEXT, weight=BOLD)
            label.move_to(box.get_center())
            char_boxes.add(VGroup(box, label))

        char_boxes.arrange(RIGHT, buff=0.15)
        char_boxes.move_to(UP * 2.2)

        text_label = Text("Raw characters (bytes)", font_size=16, color=NM_BLUE)
        text_label.next_to(char_boxes, UP, buff=0.25)

        self.play(
            Write(text_label),
            LaggedStart(*[FadeIn(c, shift=DOWN * 0.15) for c in char_boxes], lag_ratio=0.08),
            run_time=0.6,
        )
        self.wait(0.3)

        # === Step 2: Show pair frequency counting ===
        freq_label = Text("Count adjacent pairs", font_size=18, color=NM_YELLOW, weight=BOLD)
        freq_label.move_to(UP * 0.8)

        pairs = [("l", "o", 1), ("o", "w", 1), ("w", "e", 1), ("e", "r", 1)]
        pair_items = VGroup()
        for a, b, count in pairs:
            pair_text = Text(f"({a},{b}): {count}", font_size=14, color=NM_TEXT)
            pair_items.add(pair_text)
        pair_items.arrange(RIGHT, buff=0.5)
        pair_items.next_to(freq_label, DOWN, buff=0.25)

        self.play(Write(freq_label), run_time=0.3)
        self.play(
            LaggedStart(*[FadeIn(p) for p in pair_items], lag_ratio=0.1),
            run_time=0.5,
        )
        self.wait(0.3)

        # === Step 3: Merge most frequent pair — animate 3 merge rounds ===
        merge_label = Text("Merge most frequent pair", font_size=18, color=NM_GREEN, weight=BOLD)
        merge_label.move_to(DOWN * 0.3)
        self.play(Write(merge_label), run_time=0.3)

        # Simulate 3 merge rounds with different words
        merge_demos = [
            {
                "before": ["l", "o", "w", "e", "r"],
                "pair": ("e", "r"),
                "after": ["l", "o", "w", "er"],
                "merge_idx": (3, 4),
                "new_token": "er",
                "vocab_id": "256",
            },
            {
                "before": ["l", "o", "w", "er"],
                "pair": ("l", "o"),
                "after": ["lo", "w", "er"],
                "merge_idx": (0, 1),
                "new_token": "lo",
                "vocab_id": "257",
            },
            {
                "before": ["lo", "w", "er"],
                "pair": ("w", "er"),
                "after": ["lo", "wer"],
                "merge_idx": (1, 2),
                "new_token": "wer",
                "vocab_id": "258",
            },
        ]

        demo_y = DOWN * 1.5

        for step, demo in enumerate(merge_demos):
            # Show before tokens
            before_boxes = VGroup()
            for tok in demo["before"]:
                box = RoundedRectangle(
                    corner_radius=0.08, width=max(0.6, len(tok) * 0.35), height=0.5,
                    color=NM_BLUE, fill_opacity=0.2, stroke_width=1.5,
                )
                label = Text(tok, font_size=18, color=NM_TEXT)
                label.move_to(box.get_center())
                before_boxes.add(VGroup(box, label))
            before_boxes.arrange(RIGHT, buff=0.12)
            before_boxes.move_to(demo_y)

            self.play(FadeIn(before_boxes), run_time=0.25)

            # Highlight the merging pair
            idx_a, idx_b = demo["merge_idx"]
            highlight = SurroundingRectangle(
                VGroup(before_boxes[idx_a], before_boxes[idx_b]),
                color=NM_PRIMARY, buff=0.06, stroke_width=2,
            )
            merge_arrow_label = Text(
                f"merge → {demo['new_token']} (id {demo['vocab_id']})",
                font_size=14, color=NM_PRIMARY,
            )
            merge_arrow_label.next_to(highlight, DOWN, buff=0.15)

            self.play(Create(highlight), FadeIn(merge_arrow_label), run_time=0.3)
            self.wait(0.2)

            # Show after tokens
            after_boxes = VGroup()
            for tok in demo["after"]:
                color = NM_GREEN if tok == demo["new_token"] else NM_BLUE
                box = RoundedRectangle(
                    corner_radius=0.08, width=max(0.6, len(tok) * 0.35), height=0.5,
                    color=color, fill_opacity=0.3 if tok == demo["new_token"] else 0.2,
                    stroke_width=1.5,
                )
                label = Text(tok, font_size=18, color=NM_TEXT)
                label.move_to(box.get_center())
                after_boxes.add(VGroup(box, label))
            after_boxes.arrange(RIGHT, buff=0.12)
            after_boxes.move_to(demo_y)

            self.play(
                FadeOut(before_boxes), FadeOut(highlight), FadeOut(merge_arrow_label),
                FadeIn(after_boxes),
                run_time=0.35,
            )
            self.wait(0.2)
            self.play(FadeOut(after_boxes), run_time=0.15)

        # === Step 4: Final result ===
        self.play(FadeOut(freq_label), FadeOut(pair_items), FadeOut(merge_label), run_time=0.3)

        result_before = Text('"lower" → 5 bytes', font_size=18, color=NM_BLUE)
        result_after = Text('"lower" → 2 tokens: [lo][wer]', font_size=18, color=NM_GREEN, weight=BOLD)
        vocab_note = Text(
            "256 byte tokens + N merges = learned vocabulary",
            font_size=16, color=NM_YELLOW,
        )
        result_group = VGroup(result_before, result_after, vocab_note).arrange(DOWN, buff=0.2)
        result_group.move_to(DOWN * 1.0)

        self.play(
            LaggedStart(*[FadeIn(r, shift=UP * 0.15) for r in result_group], lag_ratio=0.2),
            run_time=0.6,
        )
        self.wait(0.8)

        # Cleanup
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.6)
