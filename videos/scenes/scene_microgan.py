"""
Scene: Generative Adversarial Network
Script: microgan.py
Description: Generator vs Discriminator — two networks competing in a minimax game
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE, NM_PURPLE
from manim import *


class GANScene(NoMagicScene):
    title_text = "Generative Adversarial Network"
    subtitle_text = "Generator vs Discriminator — learn by competing"

    def animate(self):
        # === Step 1: Show the two networks ===
        gen_box = RoundedRectangle(
            corner_radius=0.12, width=2.5, height=1.5,
            color=NM_GREEN, fill_opacity=0.12, stroke_width=2,
        )
        gen_label = Text("Generator", font_size=18, color=NM_GREEN, weight=BOLD)
        gen_label.move_to(gen_box.get_center() + UP * 0.2)
        gen_desc = Text("noise \u2192 fake data", font_size=12, color=NM_GRID)
        gen_desc.move_to(gen_box.get_center() + DOWN * 0.25)
        gen_group = VGroup(gen_box, gen_label, gen_desc)
        gen_group.move_to(LEFT * 3.5 + UP * 1.0)

        disc_box = RoundedRectangle(
            corner_radius=0.12, width=2.5, height=1.5,
            color=NM_PRIMARY, fill_opacity=0.12, stroke_width=2,
        )
        disc_label = Text("Discriminator", font_size=18, color=NM_PRIMARY, weight=BOLD)
        disc_label.move_to(disc_box.get_center() + UP * 0.2)
        disc_desc = Text("data \u2192 real/fake?", font_size=12, color=NM_GRID)
        disc_desc.move_to(disc_box.get_center() + DOWN * 0.25)
        disc_group = VGroup(disc_box, disc_label, disc_desc)
        disc_group.move_to(RIGHT * 3.5 + UP * 1.0)

        # Arrow from generator to discriminator
        gen_to_disc = Arrow(
            gen_box.get_right(), disc_box.get_left(),
            color=NM_YELLOW, stroke_width=2, buff=0.15, tip_length=0.15,
        )
        arrow_label = Text("fake samples", font_size=12, color=NM_YELLOW)
        arrow_label.next_to(gen_to_disc, UP, buff=0.08)

        self.play(
            FadeIn(gen_group, shift=RIGHT * 0.2),
            FadeIn(disc_group, shift=LEFT * 0.2),
            run_time=0.5,
        )
        self.play(GrowArrow(gen_to_disc), FadeIn(arrow_label), run_time=0.3)

        # Real data arrow into discriminator
        real_arrow = Arrow(
            disc_box.get_top() + UP * 0.8, disc_box.get_top(),
            color=NM_BLUE, stroke_width=2, buff=0.1, tip_length=0.15,
        )
        real_label = Text("real data", font_size=12, color=NM_BLUE)
        real_label.next_to(real_arrow, LEFT, buff=0.1)
        self.play(GrowArrow(real_arrow), FadeIn(real_label), run_time=0.3)
        self.wait(0.3)

        # === Step 2: Training loop — adversarial rounds ===
        round_label = Text("Adversarial training rounds", font_size=16, color=NM_TEXT)
        round_label.move_to(DOWN * 0.8)
        self.play(Write(round_label), run_time=0.2)

        for i in range(3):
            # Discriminator trains (gets better at detecting fakes)
            d_update = Text(f"Round {i+1}: D learns to detect fakes", font_size=13, color=NM_PRIMARY)
            d_update.move_to(DOWN * 1.5)
            self.play(
                FadeIn(d_update),
                disc_box.animate.set_fill(NM_PRIMARY, opacity=0.25),
                run_time=0.3,
            )
            self.play(disc_box.animate.set_fill(NM_PRIMARY, opacity=0.12), run_time=0.15)

            # Generator trains (gets better at fooling)
            g_update = Text(f"Round {i+1}: G learns to fool D", font_size=13, color=NM_GREEN)
            g_update.move_to(DOWN * 1.5)
            self.play(
                FadeOut(d_update),
                FadeIn(g_update),
                gen_box.animate.set_fill(NM_GREEN, opacity=0.25),
                run_time=0.3,
            )
            self.play(
                gen_box.animate.set_fill(NM_GREEN, opacity=0.12),
                FadeOut(g_update),
                run_time=0.15,
            )

        self.wait(0.2)

        # === Step 3: Equilibrium — D can't tell the difference ===
        self.play(FadeOut(round_label), run_time=0.15)

        equil = VGroup(
            Text("Equilibrium: D(fake) \u2248 0.5", font_size=18, color=NM_YELLOW, weight=BOLD),
            Text("discriminator can't distinguish real from generated", font_size=14, color=NM_TEXT),
            Text("min max V(D,G) = E[log D(x)] + E[log(1-D(G(z)))]", font_size=13, color=NM_GREEN),
        ).arrange(DOWN, buff=0.12)
        equil.move_to(DOWN * 2.2)

        self.play(
            LaggedStart(*[FadeIn(e, shift=UP * 0.15) for e in equil], lag_ratio=0.2),
            run_time=0.6,
        )
        self.wait(0.8)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.6)
