"""
Scene: Variational Autoencoder
Script: microvae.py
Description: Encoder → latent z → decoder, with the reparameterization trick
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE, NM_PURPLE
from manim import *


class VAEScene(NoMagicScene):
    title_text = "Variational Autoencoder"
    subtitle_text = "Encode → sample z → decode (reparameterization trick)"

    def animate(self):
        # === Step 1: Show encoder → latent → decoder pipeline ===
        encoder = RoundedRectangle(
            corner_radius=0.12, width=2.0, height=1.5,
            color=NM_BLUE, fill_opacity=0.12, stroke_width=2,
        )
        enc_label = Text("Encoder", font_size=18, color=NM_BLUE, weight=BOLD)
        enc_label.move_to(encoder.get_center() + UP * 0.15)
        enc_desc = Text("x \u2192 \u03bc, \u03c3", font_size=13, color=NM_GRID)
        enc_desc.move_to(encoder.get_center() + DOWN * 0.25)
        enc_group = VGroup(encoder, enc_label, enc_desc)
        enc_group.move_to(LEFT * 4.0 + UP * 0.5)

        # Latent space
        latent = Circle(
            radius=0.7, color=NM_YELLOW, fill_opacity=0.08, stroke_width=2,
        )
        z_label = Text("z", font_size=24, color=NM_YELLOW, weight=BOLD)
        z_label.move_to(latent.get_center())
        latent_group = VGroup(latent, z_label)
        latent_group.move_to(UP * 0.5)

        decoder = RoundedRectangle(
            corner_radius=0.12, width=2.0, height=1.5,
            color=NM_GREEN, fill_opacity=0.12, stroke_width=2,
        )
        dec_label = Text("Decoder", font_size=18, color=NM_GREEN, weight=BOLD)
        dec_label.move_to(decoder.get_center() + UP * 0.15)
        dec_desc = Text("z \u2192 x\u0302", font_size=13, color=NM_GRID)
        dec_desc.move_to(decoder.get_center() + DOWN * 0.25)
        dec_group = VGroup(decoder, dec_label, dec_desc)
        dec_group.move_to(RIGHT * 4.0 + UP * 0.5)

        # Arrows
        enc_to_lat = Arrow(
            encoder.get_right(), latent.get_left(),
            color=NM_GRID, stroke_width=2, buff=0.15, tip_length=0.12,
        )
        lat_to_dec = Arrow(
            latent.get_right(), decoder.get_left(),
            color=NM_GRID, stroke_width=2, buff=0.15, tip_length=0.12,
        )

        self.play(
            FadeIn(enc_group, shift=RIGHT * 0.2),
            FadeIn(latent_group),
            FadeIn(dec_group, shift=LEFT * 0.2),
            GrowArrow(enc_to_lat), GrowArrow(lat_to_dec),
            run_time=0.6,
        )
        self.wait(0.3)

        # === Step 2: Show mu and sigma outputs from encoder ===
        mu_box = RoundedRectangle(
            corner_radius=0.06, width=0.8, height=0.4,
            color=NM_BLUE, fill_opacity=0.2, stroke_width=1,
        )
        mu_label = Text("\u03bc", font_size=16, color=NM_BLUE)
        mu_label.move_to(mu_box.get_center())
        mu_group = VGroup(mu_box, mu_label)
        mu_group.move_to(LEFT * 1.8 + UP * 1.3)

        sigma_box = RoundedRectangle(
            corner_radius=0.06, width=0.8, height=0.4,
            color=NM_PURPLE, fill_opacity=0.2, stroke_width=1,
        )
        sigma_label = Text("\u03c3", font_size=16, color=NM_PURPLE)
        sigma_label.move_to(sigma_box.get_center())
        sigma_group = VGroup(sigma_box, sigma_label)
        sigma_group.move_to(LEFT * 1.8 + DOWN * 0.3)

        self.play(FadeIn(mu_group), FadeIn(sigma_group), run_time=0.3)

        # === Step 3: Reparameterization trick ===
        reparam_label = Text("Reparameterization Trick", font_size=18, color=NM_PRIMARY, weight=BOLD)
        reparam_label.move_to(DOWN * 1.5)
        self.play(Write(reparam_label), run_time=0.3)

        trick_formula = VGroup(
            Text("\u03b5 ~ N(0, 1)", font_size=15, color=NM_ORANGE),
            Text("z = \u03bc + \u03c3 \u00b7 \u03b5", font_size=18, color=NM_YELLOW, weight=BOLD),
            Text("gradients flow through \u03bc and \u03c3 (not through sampling)", font_size=12, color=NM_GREEN),
        ).arrange(DOWN, buff=0.12)
        trick_formula.move_to(DOWN * 2.5)

        # Epsilon arrow into latent
        eps_dot = Dot(latent.get_top() + UP * 0.5, radius=0.06, color=NM_ORANGE)
        eps_label = Text("\u03b5", font_size=14, color=NM_ORANGE)
        eps_label.next_to(eps_dot, RIGHT, buff=0.08)
        eps_arrow = Arrow(
            eps_dot.get_center(), latent.get_top(),
            color=NM_ORANGE, stroke_width=1.5, buff=0.1, tip_length=0.1,
        )

        self.play(
            FadeIn(eps_dot), FadeIn(eps_label), GrowArrow(eps_arrow),
            run_time=0.3,
        )
        self.play(
            LaggedStart(*[FadeIn(f) for f in trick_formula], lag_ratio=0.2),
            run_time=0.5,
        )

        # Pulse the latent z
        self.play(
            latent.animate.set_fill(NM_YELLOW, opacity=0.25),
            run_time=0.2,
        )
        self.play(
            latent.animate.set_fill(NM_YELLOW, opacity=0.08),
            run_time=0.2,
        )
        self.wait(0.3)

        # === Step 4: Loss components ===
        loss = VGroup(
            Text("Loss = reconstruction + KL divergence", font_size=14, color=NM_TEXT),
            Text("KL forces z \u2248 N(0,1) \u2192 smooth latent space", font_size=13, color=NM_YELLOW),
        ).arrange(DOWN, buff=0.08)
        loss.move_to(DOWN * 3.5)

        self.play(FadeIn(loss, shift=UP * 0.15), run_time=0.4)
        self.wait(0.8)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.6)
