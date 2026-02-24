"""
Scene: Mixture of Experts
Script: micromoe.py
Description: A router dispatches tokens to specialist expert MLPs — sparse activation
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE, NM_PURPLE
from manim import *


def make_expert_box(name, color, width=1.4, height=1.8):
    """Create an expert MLP box with hidden layer visualization."""
    box = RoundedRectangle(
        corner_radius=0.1, width=width, height=height,
        color=color, fill_opacity=0.1, stroke_width=1.5,
    )
    label = Text(name, font_size=16, color=color, weight=BOLD)
    label.next_to(box, UP, buff=0.1)

    # Mini MLP layers inside the box
    layers = VGroup()
    for i in range(3):
        layer = Rectangle(
            width=width * 0.7, height=0.2,
            color=color, fill_opacity=0.25, stroke_width=0.5,
        )
        layers.add(layer)
    layers.arrange(DOWN, buff=0.15)
    layers.move_to(box.get_center())

    return VGroup(label, box, layers)


class MoEScene(NoMagicScene):
    title_text = "Mixture of Experts"
    subtitle_text = "Route each token to its specialist"

    def animate(self):
        # === Step 1: Show input tokens ===
        token_texts = ["t", "h", "o", "m"]
        token_colors = [NM_YELLOW, NM_YELLOW, NM_YELLOW, NM_YELLOW]
        tokens = VGroup()
        for i, (txt, col) in enumerate(zip(token_texts, token_colors)):
            box = RoundedRectangle(
                corner_radius=0.08, width=0.7, height=0.5,
                color=col, fill_opacity=0.3, stroke_width=1.5,
            )
            label = Text(txt, font_size=18, color=NM_TEXT)
            label.move_to(box.get_center())
            tokens.add(VGroup(box, label))

        tokens.arrange(RIGHT, buff=0.3)
        tokens.move_to(UP * 2.8)
        tokens_label = Text("Input tokens", font_size=16, color=NM_TEXT)
        tokens_label.next_to(tokens, LEFT, buff=0.3)

        self.play(
            FadeIn(tokens_label),
            LaggedStart(*[FadeIn(t, shift=DOWN * 0.2) for t in tokens], lag_ratio=0.1),
            run_time=0.9,
        )
        self.wait(0.6)

        # === Step 2: Show the router gate ===
        router = RoundedRectangle(
            corner_radius=0.15, width=5.0, height=0.8,
            color=NM_PRIMARY, fill_opacity=0.15, stroke_width=2,
        )
        router.move_to(UP * 1.3)
        router_label = Text("Router (softmax gate)", font_size=18, color=NM_PRIMARY, weight=BOLD)
        router_label.move_to(router.get_center())

        # Arrows from tokens to router
        token_arrows = VGroup()
        for t in tokens:
            arr = Arrow(
                t.get_bottom(), router.get_top(),
                color=NM_GRID, stroke_width=1.5, buff=0.1, tip_length=0.1,
            )
            token_arrows.add(arr)

        self.play(
            FadeIn(router), Write(router_label),
            LaggedStart(*[GrowArrow(a) for a in token_arrows], lag_ratio=0.05),
            run_time=0.9,
        )
        self.wait(0.6)

        # === Step 3: Show 4 expert MLPs ===
        expert_colors = [NM_GREEN, NM_BLUE, NM_ORANGE, NM_PURPLE]
        expert_names = ["Expert 1", "Expert 2", "Expert 3", "Expert 4"]
        experts = VGroup()
        for name, color in zip(expert_names, expert_colors):
            exp = make_expert_box(name, color)
            experts.add(exp)

        experts.arrange(RIGHT, buff=0.6)
        experts.move_to(DOWN * 1.0)

        self.play(
            LaggedStart(*[FadeIn(e, shift=UP * 0.2) for e in experts], lag_ratio=0.1),
            run_time=1.0,
        )
        self.wait(0.6)

        # === Step 4: Show routing scores (softmax output) ===
        scores_label = Text("top-K = 2 per token", font_size=16, color=NM_PRIMARY)
        scores_label.next_to(router, RIGHT, buff=0.3)
        self.play(Write(scores_label), run_time=0.4)

        # Route assignments: each token goes to 2 experts
        # token "t" → Expert 1, Expert 3
        # token "h" → Expert 2, Expert 4
        # token "o" → Expert 1, Expert 2
        # token "m" → Expert 3, Expert 4
        routes = [
            (0, [0, 2]),   # t → E1, E3
            (1, [1, 3]),   # h → E2, E4
            (2, [0, 1]),   # o → E1, E2
            (3, [2, 3]),   # m → E3, E4
        ]

        route_colors = [NM_YELLOW, NM_YELLOW, NM_YELLOW, NM_YELLOW]

        for tok_idx, expert_indices in routes:
            tok = tokens[tok_idx]

            # Draw routing arrows from router to selected experts
            route_arrows = VGroup()
            for ei in expert_indices:
                exp = experts[ei]
                arr = Arrow(
                    router.get_bottom(),
                    exp[1].get_top(),  # box top
                    color=expert_colors[ei],
                    stroke_width=2, buff=0.1, tip_length=0.12,
                )
                route_arrows.add(arr)

            # Highlight the token and draw its routes
            self.play(
                Indicate(tok, color=NM_YELLOW, scale_factor=1.15),
                *[GrowArrow(a) for a in route_arrows],
                run_time=0.6,
            )

            # Flash the selected experts
            flash_anims = []
            for ei in expert_indices:
                flash_anims.append(
                    experts[ei][1].animate.set_fill(expert_colors[ei], opacity=0.35)
                )
            self.play(*flash_anims, run_time=0.4)

            # Dim back
            dim_anims = [
                experts[ei][1].animate.set_fill(expert_colors[ei], opacity=0.1)
                for ei in expert_indices
            ]
            self.play(*dim_anims, FadeOut(route_arrows), run_time=0.4)

        self.wait(0.6)

        # === Step 5: Show output combination ===
        output_label = Text("weighted sum of expert outputs", font_size=16, color=NM_GREEN)
        output_label.move_to(DOWN * 2.8)

        result = Text(
            "4 experts, 2 active per token → 2x capacity, same compute",
            font_size=16, color=NM_YELLOW, weight=BOLD,
        )
        result.move_to(DOWN * 3.3)

        self.play(Write(output_label), run_time=0.6)
        self.play(FadeIn(result, shift=UP * 0.15), run_time=0.6)
        self.wait(1.6)

        # Cleanup
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.9)
