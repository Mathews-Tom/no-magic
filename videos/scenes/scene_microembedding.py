"""
Scene: Word Embeddings
Script: microembedding.py
Description: Words scatter into vector space — similar names cluster via contrastive learning
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_PURPLE
from manim import *


class EmbeddingScene(NoMagicScene):
    title_text = "Word Embeddings"
    subtitle_text = "How meaning becomes geometry"

    def animate(self):
        # === Step 1: Show words as raw text ===
        words = ["anna", "anne", "ann", "bob", "rob", "robert", "luna", "lena", "leo"]
        # Groups: similar names cluster together
        groups = {
            NM_PRIMARY: ["anna", "anne", "ann"],      # ann- cluster
            NM_BLUE: ["bob", "rob", "robert"],         # -ob cluster
            NM_GREEN: ["luna", "lena", "leo"],          # l-n/l-o cluster
        }

        # Initial random positions (scattered)
        import random as rng
        rng.seed(42)
        initial_positions = {}
        for w in words:
            initial_positions[w] = [
                rng.uniform(-5.5, 5.5),
                rng.uniform(-2.5, 2.5),
                0,
            ]

        # Final clustered positions
        cluster_centers = {
            NM_PRIMARY: [-3.0, 1.0, 0],
            NM_BLUE: [3.0, 1.0, 0],
            NM_GREEN: [0.0, -1.5, 0],
        }
        final_positions = {}
        for color, group_words in groups.items():
            cx, cy, _ = cluster_centers[color]
            for i, w in enumerate(group_words):
                angle = i * 2.1 + 0.5
                final_positions[w] = [
                    cx + 0.8 * (i - 1),
                    cy + 0.4 * ((i % 2) * 2 - 1),
                    0,
                ]

        # Color lookup
        word_color = {}
        for color, group_words in groups.items():
            for w in group_words:
                word_color[w] = color

        # Create word dots with labels
        dots = {}
        labels = {}
        for w in words:
            pos = initial_positions[w]
            dot = Dot(point=pos, radius=0.12, color=word_color[w])
            dot.set_fill(opacity=0.7)
            label = Text(w, font_size=14, color=NM_TEXT)
            label.next_to(dot, DOWN, buff=0.1)
            dots[w] = dot
            labels[w] = label

        # Show sparse feature space label
        sparse_label = Text("raw text — no structure", font_size=20, color=NM_TEXT)
        sparse_label.to_edge(UP, buff=0.5)
        self.play(Write(sparse_label), run_time=0.4)

        # Scatter words onto canvas
        all_dots = [FadeIn(dots[w], scale=0.5) for w in words]
        all_labels = [FadeIn(labels[w]) for w in words]
        self.play(LaggedStart(*all_dots, lag_ratio=0.05), run_time=0.8)
        self.play(LaggedStart(*all_labels, lag_ratio=0.05), run_time=0.6)
        self.wait(0.5)

        # === Step 2: Show n-gram extraction ===
        ngram_label = Text("extract n-grams → contrastive loss", font_size=20, color=NM_YELLOW)
        ngram_label.to_edge(UP, buff=0.5)
        self.play(ReplacementTransform(sparse_label, ngram_label), run_time=0.5)
        self.wait(0.4)

        # === Step 3: Training — dots migrate to clusters ===
        training_label = Text("training: similar names attract, dissimilar repel",
                              font_size=18, color=NM_PRIMARY)
        training_label.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(training_label), run_time=0.4)

        # Animate movement to clustered positions
        move_anims = []
        label_anims = []
        for w in words:
            target = final_positions[w]
            move_anims.append(dots[w].animate.move_to(target))
            new_label_pos = [target[0], target[1] - 0.25, 0]
            label_anims.append(labels[w].animate.move_to(new_label_pos))

        self.play(*move_anims, *label_anims, run_time=2.0, rate_func=smooth)
        self.wait(0.5)

        # === Step 4: Show cluster boundaries ===
        embed_label = Text("embedding space — distance = similarity", font_size=20, color=NM_GREEN)
        embed_label.to_edge(UP, buff=0.5)
        self.play(ReplacementTransform(ngram_label, embed_label), run_time=0.5)

        # Draw dashed circles around clusters
        circles = []
        cluster_labels = [
            (NM_PRIMARY, cluster_centers[NM_PRIMARY], "ann- cluster"),
            (NM_BLUE, cluster_centers[NM_BLUE], "-ob cluster"),
            (NM_GREEN, cluster_centers[NM_GREEN], "l- cluster"),
        ]
        for color, center, name in cluster_labels:
            circle = Circle(radius=1.2, color=color, stroke_width=1.5, stroke_opacity=0.5)
            circle.set_fill(color, opacity=0.05)
            circle.move_to(center)
            cl = Text(name, font_size=14, color=color)
            cl.next_to(circle, UP, buff=0.1)
            circles.extend([circle, cl])

        self.play(
            LaggedStart(*[FadeIn(c) for c in circles], lag_ratio=0.15),
            run_time=0.8,
        )
        self.wait(0.5)

        # === Step 5: Show cosine similarity measurement ===
        # Draw a line between "anna" and "anne" with similarity score
        sim_line = DashedLine(
            dots["anna"].get_center(), dots["anne"].get_center(),
            color=NM_YELLOW, stroke_width=1.5, dash_length=0.1,
        )
        sim_score = Text("cos = 0.94", font_size=14, color=NM_YELLOW)
        sim_score.next_to(sim_line, UP, buff=0.08)

        self.play(Create(sim_line), FadeIn(sim_score), run_time=0.5)
        self.wait(0.8)

        # Cleanup
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.6)
