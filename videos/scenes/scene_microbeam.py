"""
Scene: Beam Search & Decoding Strategies
Script: microbeam.py
Description: Tree-based search with beam width — exploring multiple candidate sequences
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE, NM_PURPLE
from manim import *


class BeamSearchScene(NoMagicScene):
    title_text = "Beam Search"
    subtitle_text = "Explore multiple paths — keep the top-k at each step"

    def animate(self):
        beam_width = 2

        # === Step 1: Greedy vs Beam intro ===
        greedy_label = Text("Greedy: always pick top-1", font_size=16, color=NM_ORANGE)
        beam_label = Text(f"Beam: keep top-{beam_width} at each step", font_size=16, color=NM_GREEN)
        intro = VGroup(greedy_label, beam_label).arrange(DOWN, buff=0.15)
        intro.move_to(UP * 2.5)
        self.play(
            LaggedStart(FadeIn(greedy_label), FadeIn(beam_label), lag_ratio=0.2),
            run_time=0.8,
        )
        self.wait(0.6)
        self.play(FadeOut(intro), run_time=0.4)

        # === Step 2: Build the beam search tree ===
        # Tree structure: root -> depth 1 (3 candidates) -> depth 2 (keep 2, expand) -> depth 3

        root_text = "^"  # start token
        root = RoundedRectangle(
            corner_radius=0.1, width=0.7, height=0.5,
            color=NM_YELLOW, fill_opacity=0.3, stroke_width=2,
        )
        root_label = Text(root_text, font_size=18, color=NM_TEXT, weight=BOLD)
        root_label.move_to(root.get_center())
        root_group = VGroup(root, root_label)
        root_group.move_to(UP * 2.2)

        self.play(FadeIn(root_group), run_time=0.4)

        # Depth 1: 3 candidates from root
        d1_tokens = [("J", 0.4), ("M", 0.35), ("A", 0.25)]
        d1_nodes = VGroup()
        d1_probs = VGroup()
        for tok, prob in d1_tokens:
            box = RoundedRectangle(
                corner_radius=0.08, width=0.65, height=0.45,
                color=NM_BLUE, fill_opacity=0.2, stroke_width=1.5,
            )
            label = Text(tok, font_size=16, color=NM_TEXT)
            label.move_to(box.get_center())
            prob_label = Text(f"{prob:.2f}", font_size=10, color=NM_GRID)
            node = VGroup(box, label)
            d1_nodes.add(node)
            d1_probs.add(prob_label)

        d1_nodes.arrange(RIGHT, buff=0.8)
        d1_nodes.move_to(UP * 0.8)

        # Arrows from root to depth 1
        d1_arrows = VGroup()
        for node in d1_nodes:
            arr = Arrow(
                root_group.get_bottom(), node.get_top(),
                color=NM_GRID, stroke_width=1.5, buff=0.08, tip_length=0.1,
            )
            d1_arrows.add(arr)

        for i, prob in enumerate(d1_probs):
            prob.next_to(d1_nodes[i], UP, buff=0.08)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in d1_arrows], lag_ratio=0.1),
            LaggedStart(*[FadeIn(n, shift=DOWN * 0.1) for n in d1_nodes], lag_ratio=0.1),
            LaggedStart(*[FadeIn(p) for p in d1_probs], lag_ratio=0.1),
            run_time=0.9,
        )
        self.wait(0.4)

        # === Step 3: Prune — keep top-2 (beam width) ===
        prune_label = Text(f"prune to beam width = {beam_width}", font_size=14, color=NM_PRIMARY)
        prune_label.move_to(RIGHT * 4.5 + UP * 0.8)
        self.play(Write(prune_label), run_time=0.4)

        # Fade out the pruned candidate (A, index 2)
        self.play(
            d1_nodes[2].animate.set_opacity(0.15),
            d1_arrows[2].animate.set_opacity(0.15),
            d1_probs[2].animate.set_opacity(0.15),
            run_time=0.4,
        )

        # Highlight surviving beams
        for i in range(beam_width):
            self.play(
                d1_nodes[i][0].animate.set_stroke(NM_GREEN, width=2),
                run_time=0.4,
            )

        self.wait(0.4)

        # === Step 4: Expand depth 2 from surviving beams ===
        d2_data = [
            [("Ja", 0.15), ("Jo", 0.12)],  # from J
            [("Ma", 0.14), ("Mi", 0.10)],   # from M
        ]
        d2_all_nodes = VGroup()
        d2_all_arrows = VGroup()
        d2_all_probs = VGroup()

        positions = [LEFT * 2.5, LEFT * 0.5, RIGHT * 1.5, RIGHT * 3.5]

        for beam_idx, expansions in enumerate(d2_data):
            parent = d1_nodes[beam_idx]
            for exp_idx, (tok, prob) in enumerate(expansions):
                pos_idx = beam_idx * 2 + exp_idx
                box = RoundedRectangle(
                    corner_radius=0.08, width=0.75, height=0.45,
                    color=NM_BLUE, fill_opacity=0.2, stroke_width=1.5,
                )
                label = Text(tok, font_size=14, color=NM_TEXT)
                label.move_to(box.get_center())
                node = VGroup(box, label)
                node.move_to(positions[pos_idx] + DOWN * 0.8)

                arr = Arrow(
                    parent.get_bottom(), node.get_top(),
                    color=NM_GRID, stroke_width=1.5, buff=0.08, tip_length=0.1,
                )

                prob_label = Text(f"{prob:.2f}", font_size=10, color=NM_GRID)
                prob_label.next_to(node, UP, buff=0.05)

                d2_all_nodes.add(node)
                d2_all_arrows.add(arr)
                d2_all_probs.add(prob_label)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in d2_all_arrows], lag_ratio=0.08),
            LaggedStart(*[FadeIn(n, shift=DOWN * 0.1) for n in d2_all_nodes], lag_ratio=0.08),
            LaggedStart(*[FadeIn(p) for p in d2_all_probs], lag_ratio=0.08),
            run_time=0.9,
        )

        # Prune again — keep top-2 by cumulative probability
        self.play(
            d2_all_nodes[1].animate.set_opacity(0.15),
            d2_all_arrows[1].animate.set_opacity(0.15),
            d2_all_probs[1].animate.set_opacity(0.15),
            d2_all_nodes[3].animate.set_opacity(0.15),
            d2_all_arrows[3].animate.set_opacity(0.15),
            d2_all_probs[3].animate.set_opacity(0.15),
            run_time=0.4,
        )
        # Highlight winners
        self.play(
            d2_all_nodes[0][0].animate.set_stroke(NM_GREEN, width=2),
            d2_all_nodes[2][0].animate.set_stroke(NM_GREEN, width=2),
            run_time=0.4,
        )
        self.wait(0.6)

        # === Step 5: Final result ===
        self.play(FadeOut(prune_label), run_time=0.4)
        result = VGroup(
            Text('Best beam: "Ja" (cumulative prob 0.15)', font_size=16, color=NM_GREEN, weight=BOLD),
            Text("beam search explores O(k\u00b7V) paths vs greedy's O(V)",
                 font_size=14, color=NM_YELLOW),
        ).arrange(DOWN, buff=0.12)
        result.move_to(DOWN * 2.5)

        self.play(
            LaggedStart(*[FadeIn(r, shift=UP * 0.15) for r in result], lag_ratio=0.2),
            run_time=0.8,
        )
        self.wait(1.6)

        # Cleanup
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.9)
