"""
Scene: BERT (Bidirectional Encoder)
Script: microbert.py
Description: Bidirectional attention — every token sees every other token to predict [MASK]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE, NM_PURPLE
from manim import *


class BERTScene(NoMagicScene):
    title_text = "BERT"
    subtitle_text = "Bidirectional attention — predict the [MASK]"

    def animate(self):
        # === Step 1: Show GPT-style causal (left-to-right) attention ===
        causal_label = Text("GPT: Causal (left-to-right)", font_size=20, color=NM_ORANGE, weight=BOLD)
        causal_label.move_to(LEFT * 3.2 + UP * 2.8)

        causal_tokens = ["T", "h", "o", "m"]
        causal_boxes = VGroup()
        for tok in causal_tokens:
            box = RoundedRectangle(
                corner_radius=0.08, width=0.6, height=0.5,
                color=NM_ORANGE, fill_opacity=0.2, stroke_width=1.5,
            )
            label = Text(tok, font_size=20, color=NM_TEXT)
            label.move_to(box.get_center())
            causal_boxes.add(VGroup(box, label))
        causal_boxes.arrange(RIGHT, buff=0.2)
        causal_boxes.move_to(LEFT * 3.2 + UP * 1.8)

        # Causal mask: lower-triangular arrows
        causal_arrows = VGroup()
        for i in range(len(causal_tokens)):
            for j in range(i + 1):  # can only attend to self and left
                if i != j:
                    arr = Arrow(
                        causal_boxes[j].get_bottom() + DOWN * 0.05,
                        causal_boxes[i].get_bottom() + DOWN * 0.3,
                        color=NM_ORANGE, stroke_width=1, buff=0.05, tip_length=0.08,
                        stroke_opacity=0.5,
                    )
                    causal_arrows.add(arr)

        self.play(Write(causal_label), run_time=0.3)
        self.play(
            LaggedStart(*[FadeIn(b, shift=DOWN * 0.1) for b in causal_boxes], lag_ratio=0.08),
            run_time=0.4,
        )
        self.play(
            LaggedStart(*[GrowArrow(a) for a in causal_arrows], lag_ratio=0.03),
            run_time=0.5,
        )

        causal_note = Text("each token sees only the left", font_size=13, color=NM_ORANGE)
        causal_note.next_to(causal_boxes, DOWN, buff=0.5)
        self.play(FadeIn(causal_note), run_time=0.2)
        self.wait(0.3)

        # === Step 2: Show BERT-style bidirectional attention ===
        bert_label = Text("BERT: Bidirectional", font_size=20, color=NM_GREEN, weight=BOLD)
        bert_label.move_to(RIGHT * 3.2 + UP * 2.8)

        bert_tokens = ["T", "[M]", "o", "m"]
        bert_colors = [NM_GREEN, NM_PRIMARY, NM_GREEN, NM_GREEN]
        bert_boxes = VGroup()
        for tok, col in zip(bert_tokens, bert_colors):
            box = RoundedRectangle(
                corner_radius=0.08, width=0.6, height=0.5,
                color=col, fill_opacity=0.3 if tok == "[M]" else 0.2,
                stroke_width=2 if tok == "[M]" else 1.5,
            )
            label = Text(tok, font_size=18, color=NM_TEXT)
            label.move_to(box.get_center())
            bert_boxes.add(VGroup(box, label))
        bert_boxes.arrange(RIGHT, buff=0.2)
        bert_boxes.move_to(RIGHT * 3.2 + UP * 1.8)

        # Bidirectional: all-to-all arrows
        bert_arrows = VGroup()
        for i in range(len(bert_tokens)):
            for j in range(len(bert_tokens)):
                if i != j:
                    arr = Arrow(
                        bert_boxes[j].get_bottom() + DOWN * 0.05,
                        bert_boxes[i].get_bottom() + DOWN * 0.3,
                        color=NM_GREEN, stroke_width=1, buff=0.05, tip_length=0.08,
                        stroke_opacity=0.4,
                    )
                    bert_arrows.add(arr)

        self.play(Write(bert_label), run_time=0.3)
        self.play(
            LaggedStart(*[FadeIn(b, shift=DOWN * 0.1) for b in bert_boxes], lag_ratio=0.08),
            run_time=0.4,
        )
        self.play(
            LaggedStart(*[GrowArrow(a) for a in bert_arrows], lag_ratio=0.02),
            run_time=0.5,
        )

        bert_note = Text("every token sees everything", font_size=13, color=NM_GREEN)
        bert_note.next_to(bert_boxes, DOWN, buff=0.5)
        self.play(FadeIn(bert_note), run_time=0.2)
        self.wait(0.4)

        # === Step 3: Show the masked prediction task ===
        # Clear side-by-side, move to center for the BERT prediction demo
        self.play(
            *[FadeOut(m) for m in [
                causal_label, causal_boxes, causal_arrows, causal_note,
                bert_label, bert_boxes, bert_arrows, bert_note,
            ]],
            run_time=0.4,
        )

        mlm_label = Text("Masked Language Modeling", font_size=22, color=NM_PRIMARY, weight=BOLD)
        mlm_label.move_to(UP * 2.5)
        self.play(Write(mlm_label), run_time=0.3)

        # Input sequence with mask
        input_tokens = ["T", "h", "[MASK]", "m", "a", "s"]
        input_boxes = VGroup()
        for tok in input_tokens:
            is_mask = tok == "[MASK]"
            box = RoundedRectangle(
                corner_radius=0.08,
                width=1.0 if is_mask else 0.6,
                height=0.55,
                color=NM_PRIMARY if is_mask else NM_BLUE,
                fill_opacity=0.35 if is_mask else 0.15,
                stroke_width=2.5 if is_mask else 1.5,
            )
            label = Text(tok, font_size=16, color=NM_PRIMARY if is_mask else NM_TEXT)
            label.move_to(box.get_center())
            input_boxes.add(VGroup(box, label))
        input_boxes.arrange(RIGHT, buff=0.15)
        input_boxes.move_to(UP * 1.2)

        self.play(
            LaggedStart(*[FadeIn(b, shift=DOWN * 0.15) for b in input_boxes], lag_ratio=0.06),
            run_time=0.5,
        )

        # Show bidirectional context arrows converging on [MASK]
        mask_idx = 2
        context_arrows = VGroup()
        for i in range(len(input_tokens)):
            if i != mask_idx:
                arr = Arrow(
                    input_boxes[i].get_bottom(),
                    input_boxes[mask_idx].get_bottom() + DOWN * 0.4,
                    color=NM_GREEN, stroke_width=1.5, buff=0.08, tip_length=0.1,
                    stroke_opacity=0.6,
                )
                context_arrows.add(arr)

        context_label = Text("all context flows to [MASK]", font_size=14, color=NM_GREEN)
        context_label.next_to(input_boxes, DOWN, buff=0.6)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in context_arrows], lag_ratio=0.06),
            FadeIn(context_label),
            run_time=0.5,
        )
        self.wait(0.3)

        # === Step 4: Prediction ===
        prediction = Text("o", font_size=28, color=NM_GREEN, weight=BOLD)
        pred_box = RoundedRectangle(
            corner_radius=0.08, width=0.6, height=0.55,
            color=NM_GREEN, fill_opacity=0.3, stroke_width=2,
        )
        prediction.move_to(input_boxes[mask_idx].get_center())
        pred_box.move_to(input_boxes[mask_idx].get_center())

        self.play(
            FadeOut(input_boxes[mask_idx]),
            FadeIn(pred_box), FadeIn(prediction),
            run_time=0.4,
        )

        result = Text(
            'T h [o] m a s  — predicted from both sides',
            font_size=16, color=NM_YELLOW, weight=BOLD,
        )
        result.move_to(DOWN * 2.0)
        self.play(FadeIn(result, shift=UP * 0.15), run_time=0.4)
        self.wait(0.8)

        # Cleanup
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.6)
