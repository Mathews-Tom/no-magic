"""
Scene: Denoising Diffusion
Script: microdiffusion.py
Description: Pure noise → iterative denoising → clean data (the DDPM algorithm)
"""
import sys
import math
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE, NM_PURPLE
from manim import *


class DiffusionScene(NoMagicScene):
    title_text = "Denoising Diffusion"
    subtitle_text = "Noise → iterative denoising → data (DDPM)"

    def animate(self):
        random.seed(42)

        # === Step 1: Show the forward process (data → noise) ===
        forward_label = Text("Forward: add noise at each step", font_size=18, color=NM_PRIMARY, weight=BOLD)
        forward_label.move_to(UP * 2.8)
        self.play(Write(forward_label), run_time=0.4)

        n_points = 30
        n_steps = 5

        # Generate spiral data points
        def spiral_points():
            pts = []
            for i in range(n_points):
                t = i / n_points * 2 * math.pi
                r = 0.5 + t * 0.25
                x = r * math.cos(t) + random.gauss(0, 0.05)
                y = r * math.sin(t) + random.gauss(0, 0.05)
                pts.append((x, y))
            return pts

        clean_pts = spiral_points()

        # Show forward noising steps
        step_labels = ["t=0 (data)", "t=1", "t=2", "t=3", "t=T (noise)"]
        step_groups = VGroup()

        for step in range(n_steps):
            noise_scale = step / (n_steps - 1)
            dots = VGroup()
            for x, y in clean_pts:
                nx = x * (1 - noise_scale) + random.gauss(0, 1) * noise_scale
                ny = y * (1 - noise_scale) + random.gauss(0, 1) * noise_scale
                color = interpolate_color(
                    ManimColor(NM_GREEN), ManimColor(NM_PRIMARY), noise_scale
                )
                dot = Dot(
                    point=[nx * 0.8, ny * 0.8, 0],
                    radius=0.04, color=color,
                )
                dot.set_fill(opacity=0.7)
                dots.add(dot)

            label = Text(step_labels[step], font_size=11, color=NM_TEXT)

            group = VGroup(dots, label)
            dots.move_to(ORIGIN)
            label.next_to(dots, DOWN, buff=0.15)
            step_groups.add(group)

        step_groups.arrange(RIGHT, buff=0.5)
        step_groups.move_to(UP * 1.0)

        # Draw right arrows between steps
        arrows = VGroup()
        for i in range(n_steps - 1):
            arr = Arrow(
                step_groups[i].get_right(), step_groups[i + 1].get_left(),
                color=NM_GRID, stroke_width=1.5, buff=0.1, tip_length=0.1,
            )
            arrows.add(arr)

        self.play(
            LaggedStart(*[FadeIn(g) for g in step_groups], lag_ratio=0.12),
            LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.1),
            run_time=1.5,
        )
        self.wait(0.6)

        # === Step 2: Show reverse process (denoise step-by-step) ===
        self.play(
            FadeOut(forward_label), FadeOut(step_groups), FadeOut(arrows),
            run_time=0.4,
        )

        reverse_label = Text("Reverse: learned denoising (predict noise, subtract)", font_size=18, color=NM_GREEN, weight=BOLD)
        reverse_label.move_to(UP * 2.8)
        self.play(Write(reverse_label), run_time=0.4)

        # Animate a single denoising trajectory in the center
        random.seed(42)
        current_dots = VGroup()
        for _ in range(n_points):
            x = random.gauss(0, 1) * 0.8
            y = random.gauss(0, 1) * 0.8
            dot = Dot(point=[x, y, 0], radius=0.05, color=NM_PRIMARY)
            dot.set_fill(opacity=0.6)
            current_dots.add(dot)

        noise_label = Text("pure noise", font_size=14, color=NM_PRIMARY)
        noise_label.move_to(DOWN * 1.8)
        self.play(FadeIn(current_dots), FadeIn(noise_label), run_time=0.4)
        self.wait(0.4)

        denoise_steps = 4
        for step in range(denoise_steps):
            progress = (step + 1) / denoise_steps
            new_dots = VGroup()
            for i, (x, y) in enumerate(clean_pts):
                cx = current_dots[i].get_center()[0]
                cy = current_dots[i].get_center()[1]
                tx = x * 0.8
                ty = y * 0.8
                nx = cx + (tx - cx) * 0.4 + random.gauss(0, 0.05 * (1 - progress))
                ny = cy + (ty - cy) * 0.4 + random.gauss(0, 0.05 * (1 - progress))
                color = interpolate_color(
                    ManimColor(NM_PRIMARY), ManimColor(NM_GREEN), progress
                )
                dot = Dot(point=[nx, ny, 0], radius=0.05, color=color)
                dot.set_fill(opacity=0.7)
                new_dots.add(dot)

            step_text = Text(f"denoise step {step + 1}/{denoise_steps}", font_size=14, color=NM_GREEN)
            step_text.move_to(DOWN * 1.8)

            self.play(
                Transform(current_dots, new_dots),
                Transform(noise_label, step_text),
                run_time=0.6,
            )
            self.wait(0.4)

        # === Step 3: Result ===
        result = Text(
            "trained model: predict noise \u03b5, subtract it, repeat",
            font_size=16, color=NM_YELLOW, weight=BOLD,
        )
        result.move_to(DOWN * 2.5)
        self.play(FadeIn(result, shift=UP * 0.15), run_time=0.6)
        self.wait(1.6)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.9)
