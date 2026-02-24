"""
Scene: Autoregressive GPT
Script: microgpt.py
Description: Token-by-token next-token prediction — the core generation loop of GPT
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE, NM_PURPLE
from manim import *


def make_token_box(text, color, fill_opacity=0.3, font_size=20, width=0.9, height=0.6):
    """Create a rounded box with a token label inside."""
    box = RoundedRectangle(
        corner_radius=0.1, width=width, height=height,
        color=color, fill_opacity=fill_opacity, stroke_width=1.5,
    )
    label = Text(text, font_size=font_size, color=NM_TEXT)
    return VGroup(box, label)


class GPTScene(NoMagicScene):
    title_text = "Autoregressive GPT"
    subtitle_text = "Next-token prediction, one step at a time"

    def animate(self):
        # === Step 1: Show the transformer block diagram ===
        block_label = Text("Transformer Block", font_size=22, color=NM_TEXT, weight=BOLD)
        block_label.to_edge(UP, buff=0.5)

        # Simplified block: Embed → Attention → MLP → Predict
        stages = ["Embed", "Attention", "MLP", "Predict"]
        stage_colors = [NM_BLUE, NM_PRIMARY, NM_PURPLE, NM_GREEN]
        stage_boxes = VGroup()

        for name, color in zip(stages, stage_colors):
            box = VGroup(
                RoundedRectangle(
                    corner_radius=0.12, width=1.6, height=0.7,
                    color=color, fill_opacity=0.2, stroke_width=1.5,
                ),
                Text(name, font_size=18, color=color),
            )
            stage_boxes.add(box)

        stage_boxes.arrange(RIGHT, buff=0.6)
        stage_boxes.next_to(block_label, DOWN, buff=0.4)

        self.play(Write(block_label), run_time=0.6)
        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in stage_boxes], lag_ratio=0.15),
            run_time=1.2,
        )

        # Arrows between stages
        arrows = VGroup()
        for i in range(len(stage_boxes) - 1):
            arrow = Arrow(
                stage_boxes[i].get_right(), stage_boxes[i + 1].get_left(),
                buff=0.08, color=NM_GRID, stroke_width=2,
            )
            arrows.add(arrow)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.1),
            run_time=0.8,
        )
        self.wait(0.6)

        # === Step 2: Show the autoregressive generation loop ===
        loop_label = Text("Autoregressive Generation", font_size=20, color=NM_YELLOW, weight=BOLD)
        loop_label.move_to(DOWN * 0.5)
        self.play(Write(loop_label), run_time=0.6)

        # Token sequence area
        tokens = ["t", "h", "o", "m", "a"]
        generated = ["s"]  # tokens to be "generated" one at a time

        token_group = VGroup()
        start_x = LEFT * 3.5

        # Show existing context tokens
        for i, tok in enumerate(tokens):
            tbox = make_token_box(tok, NM_BLUE, fill_opacity=0.4)
            tbox.move_to(start_x + RIGHT * i * 1.1 + DOWN * 1.6)
            token_group.add(tbox)

        self.play(
            LaggedStart(*[FadeIn(t, shift=UP * 0.2) for t in token_group], lag_ratio=0.1),
            run_time=0.9,
        )

        context_brace = Brace(token_group, DOWN, color=NM_TEXT)
        context_label = Text("context window", font_size=16, color=NM_TEXT)
        context_label.next_to(context_brace, DOWN, buff=0.1)
        self.play(FadeIn(context_brace), FadeIn(context_label), run_time=0.6)
        self.wait(0.6)

        # === Step 3: Animate generation — predict next token ===
        for gen_tok in generated:
            # Highlight "Predict" stage
            predict_box = stage_boxes[-1]
            self.play(Indicate(predict_box, color=NM_GREEN, scale_factor=1.1), run_time=0.6)

            # Show probability distribution as small bars
            prob_labels = ["a", "s", "e", "i", "·"]
            prob_heights = [0.3, 0.9, 0.2, 0.15, 0.1]  # "s" is highest
            prob_bars = VGroup()
            bar_group_pos = RIGHT * 3.5 + DOWN * 1.6

            for j, (pl, ph) in enumerate(zip(prob_labels, prob_heights)):
                bar = Rectangle(
                    width=0.25, height=ph,
                    color=NM_GREEN if ph == max(prob_heights) else NM_GRID,
                    fill_opacity=0.6 if ph == max(prob_heights) else 0.3,
                    stroke_width=1,
                )
                bar.move_to(bar_group_pos + RIGHT * j * 0.4 + UP * ph / 2)
                bar_label = Text(pl, font_size=12, color=NM_TEXT)
                bar_label.next_to(bar, DOWN, buff=0.05)
                prob_bars.add(VGroup(bar, bar_label))

            prob_title = Text("P(next)", font_size=14, color=NM_GREEN)
            prob_title.next_to(prob_bars, UP, buff=0.15)
            prob_bars.add(prob_title)

            self.play(FadeIn(prob_bars, shift=UP * 0.2), run_time=0.8)
            self.wait(0.8)

            # Sample the highest probability token
            new_tbox = make_token_box(gen_tok, NM_GREEN, fill_opacity=0.6)
            new_pos = token_group[-1].get_center() + RIGHT * 1.1
            new_tbox.move_to(new_pos)

            # Animate: bar highlights → new token appears
            self.play(
                FadeOut(prob_bars),
                FadeIn(new_tbox, scale=1.3),
                run_time=0.8,
            )
            token_group.add(new_tbox)

            # Update brace
            new_brace = Brace(token_group, DOWN, color=NM_TEXT)
            new_label = Text("context window", font_size=16, color=NM_TEXT)
            new_label.next_to(new_brace, DOWN, buff=0.1)
            self.play(
                ReplacementTransform(context_brace, new_brace),
                ReplacementTransform(context_label, new_label),
                run_time=0.4,
            )
            context_brace = new_brace
            context_label = new_label

        self.wait(0.6)

        # === Step 4: Show the loop arrow — "repeat" ===
        loop_arrow = CurvedArrow(
            token_group[-1].get_top() + UP * 0.1,
            stage_boxes[0].get_bottom() + DOWN * 0.1,
            color=NM_YELLOW, stroke_width=2, angle=-TAU / 4,
        )
        repeat_text = Text("repeat", font_size=16, color=NM_YELLOW)
        repeat_text.next_to(loop_arrow, RIGHT, buff=0.15)

        self.play(Create(loop_arrow), FadeIn(repeat_text), run_time=0.9)
        self.wait(1.0)

        # === Step 5: Show final generated sequence highlighted ===
        result_box = SurroundingRectangle(token_group, color=NM_GREEN, buff=0.12, stroke_width=2)
        result_label = Text("thomas", font_size=24, color=NM_GREEN, weight=BOLD)
        result_label.next_to(result_box, DOWN, buff=0.4)
        self.play(Create(result_box), Write(result_label), run_time=0.9)
        self.wait(1.6)

        # Cleanup
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.9)
