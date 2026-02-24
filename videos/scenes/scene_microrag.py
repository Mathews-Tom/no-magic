"""
Scene: Retrieval-Augmented Generation (RAG)
Script: microrag.py
Description: Query → retrieve → augment → generate — grounding generation in external knowledge
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE, NM_PURPLE
from manim import *


def make_doc_box(text, color, width=1.8, height=0.6):
    """Small document chip."""
    box = RoundedRectangle(
        corner_radius=0.08, width=width, height=height,
        color=color, fill_opacity=0.15, stroke_width=1,
    )
    label = Text(text, font_size=13, color=color)
    label.move_to(box.get_center())
    return VGroup(box, label)


class RAGScene(NoMagicScene):
    title_text = "Retrieval-Augmented Generation"
    subtitle_text = "Query → Retrieve → Augment → Generate"

    def animate(self):
        # === Step 1: Show the knowledge base (document store) ===
        kb_label = Text("Knowledge Base", font_size=20, color=NM_BLUE, weight=BOLD)
        kb_label.move_to(LEFT * 4.5 + UP * 2.5)

        docs = [
            "Paris is in France",
            "London has 8.9M people",
            "Berlin is on the Spree",
            "Tokyo is in Japan",
            "Cairo sits on the Nile",
            "Rome is in Italy",
        ]
        doc_boxes = VGroup()
        for i, doc in enumerate(docs):
            d = make_doc_box(doc, NM_BLUE)
            doc_boxes.add(d)

        doc_boxes.arrange_in_grid(rows=3, cols=2, buff=0.2)
        doc_boxes.move_to(LEFT * 4.5 + DOWN * 0.2)

        self.play(Write(kb_label), run_time=0.4)
        self.play(
            LaggedStart(*[FadeIn(d, shift=UP * 0.15) for d in doc_boxes], lag_ratio=0.08),
            run_time=1.0,
        )
        self.wait(0.6)

        # === Step 2: Show the query ===
        query_box = RoundedRectangle(
            corner_radius=0.12, width=3.5, height=0.8,
            color=NM_YELLOW, fill_opacity=0.15, stroke_width=2,
        )
        query_text = Text('"capital of France?"', font_size=18, color=NM_YELLOW)
        query_label = Text("Query", font_size=16, color=NM_YELLOW, weight=BOLD)
        query_box.move_to(RIGHT * 0.5 + UP * 2.5)
        query_text.move_to(query_box.get_center())
        query_label.next_to(query_box, UP, buff=0.1)

        self.play(FadeIn(query_box), Write(query_text), Write(query_label), run_time=0.9)
        self.wait(0.6)

        # === Step 3: BM25 retrieval — search arrow + highlight matches ===
        retrieve_label = Text("BM25 search", font_size=16, color=NM_PRIMARY, weight=BOLD)
        retrieve_label.move_to(LEFT * 1.5 + UP * 1.2)

        search_arrow = Arrow(
            query_box.get_left(), doc_boxes.get_right() + RIGHT * 0.1,
            color=NM_PRIMARY, stroke_width=2, buff=0.15,
        )
        self.play(GrowArrow(search_arrow), Write(retrieve_label), run_time=0.8)

        # Highlight top-K matches (Paris doc is the match)
        match_idx = 0  # "Paris is in France"
        match_box = doc_boxes[match_idx]
        highlight = SurroundingRectangle(match_box, color=NM_GREEN, buff=0.05, stroke_width=2)
        score_label = Text("score: 4.2", font_size=12, color=NM_GREEN)
        score_label.next_to(highlight, RIGHT, buff=0.1)

        # Also highlight a secondary match
        match2_idx = 5  # "Rome is in Italy" (partial keyword match)
        match2_box = doc_boxes[match2_idx]
        highlight2 = SurroundingRectangle(match2_box, color=NM_GREEN, buff=0.05, stroke_width=1.5)
        highlight2.set_stroke(opacity=0.5)

        self.play(Create(highlight), FadeIn(score_label), Create(highlight2), run_time=0.9)
        self.wait(0.8)

        # === Step 4: Context augmentation ===
        # Show retrieved doc flowing into the generation context
        aug_label = Text("Augment", font_size=18, color=NM_GREEN, weight=BOLD)
        aug_label.move_to(RIGHT * 1.5 + UP * 0.3)

        context_box = RoundedRectangle(
            corner_radius=0.12, width=5.0, height=1.2,
            color=NM_GREEN, fill_opacity=0.1, stroke_width=1.5,
        )
        context_box.move_to(RIGHT * 2.5 + DOWN * 0.8)
        context_header = Text("Context", font_size=14, color=NM_GREEN, weight=BOLD)
        context_header.next_to(context_box, UP, buff=0.08)
        context_content = VGroup(
            Text("Retrieved: Paris is in France", font_size=14, color=NM_BLUE),
            Text("Query: capital of France?", font_size=14, color=NM_YELLOW),
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        context_content.move_to(context_box.get_center())

        inject_arrow = Arrow(
            highlight.get_right() + RIGHT * 0.1,
            context_box.get_left(),
            color=NM_GREEN, stroke_width=2, buff=0.15,
        )

        self.play(
            Write(aug_label),
            GrowArrow(inject_arrow),
            FadeIn(context_box), Write(context_header),
            FadeIn(context_content),
            run_time=1.2,
        )
        self.wait(0.8)

        # === Step 5: Generation ===
        gen_label = Text("Generate", font_size=18, color=NM_PURPLE, weight=BOLD)
        gen_label.move_to(RIGHT * 2.5 + DOWN * 2.0)

        gen_box = RoundedRectangle(
            corner_radius=0.12, width=3.5, height=0.8,
            color=NM_PURPLE, fill_opacity=0.15, stroke_width=2,
        )
        gen_box.move_to(RIGHT * 2.5 + DOWN * 2.8)
        gen_text = Text('"Paris"', font_size=22, color=NM_PURPLE, weight=BOLD)
        gen_text.move_to(gen_box.get_center())

        gen_arrow = Arrow(
            context_box.get_bottom(),
            gen_box.get_top(),
            color=NM_PURPLE, stroke_width=2, buff=0.15,
        )

        self.play(Write(gen_label), GrowArrow(gen_arrow), run_time=0.6)
        self.play(FadeIn(gen_box), Write(gen_text), run_time=0.8)

        # Flash the answer
        self.play(Indicate(gen_box, color=NM_GREEN, scale_factor=1.05), run_time=0.6)
        self.wait(1.6)

        # Cleanup
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.9)
