"""
Scene: Dropout & Regularization
Script: microdropout.py
Description: Randomly killing neurons during training prevents overfitting
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE, NM_PURPLE
from manim import *


class DropoutScene(NoMagicScene):
    title_text = "Dropout"
    subtitle_text = "Randomly kill neurons — force distributed representations"

    def animate(self):
        # === Step 1: Show a simple neural network ===
        net_label = Text("Training (with dropout)", font_size=20, color=NM_PRIMARY, weight=BOLD)
        net_label.move_to(UP * 2.8)
        self.play(Write(net_label), run_time=0.3)

        layers = [4, 6, 6, 3]
        neuron_radius = 0.15
        layer_spacing = 2.2
        neuron_spacing = 0.55

        all_neurons = []
        all_neuron_dots = VGroup()

        for l, n_neurons in enumerate(layers):
            layer = []
            for i in range(n_neurons):
                x = (l - len(layers) / 2 + 0.5) * layer_spacing
                y = (i - n_neurons / 2 + 0.5) * neuron_spacing + 0.3
                dot = Circle(
                    radius=neuron_radius,
                    color=NM_BLUE, fill_opacity=0.3, stroke_width=1.5,
                )
                dot.move_to([x, y, 0])
                layer.append(dot)
                all_neuron_dots.add(dot)
            all_neurons.append(layer)

        # Connections
        connections = VGroup()
        for l in range(len(layers) - 1):
            for n1 in all_neurons[l]:
                for n2 in all_neurons[l + 1]:
                    line = Line(
                        n1.get_center(), n2.get_center(),
                        color=NM_GRID, stroke_width=0.5, stroke_opacity=0.3,
                    )
                    connections.add(line)

        self.play(FadeIn(connections), FadeIn(all_neuron_dots), run_time=0.5)
        self.wait(0.2)

        # === Step 2: Animate dropout — randomly deactivate neurons ===
        dropout_label = Text("p = 0.5 (drop 50%)", font_size=16, color=NM_PRIMARY)
        dropout_label.move_to(DOWN * 2.5)
        self.play(Write(dropout_label), run_time=0.2)

        import random
        random.seed(42)

        for round_num in range(3):
            # Choose neurons to drop (hidden layers only)
            dropped = []
            active = []
            for l in range(1, len(layers) - 1):  # skip input/output
                for neuron in all_neurons[l]:
                    if random.random() < 0.5:
                        dropped.append(neuron)
                    else:
                        active.append(neuron)

            # Drop animation
            drop_anims = [
                neuron.animate.set_fill(NM_PRIMARY, opacity=0.6).set_stroke(NM_PRIMARY)
                for neuron in dropped
            ]
            # X marks on dropped neurons
            x_marks = VGroup()
            for neuron in dropped:
                x1 = Line(
                    neuron.get_center() + UL * 0.1,
                    neuron.get_center() + DR * 0.1,
                    color=NM_PRIMARY, stroke_width=2,
                )
                x2 = Line(
                    neuron.get_center() + UR * 0.1,
                    neuron.get_center() + DL * 0.1,
                    color=NM_PRIMARY, stroke_width=2,
                )
                x_marks.add(x1, x2)

            self.play(*drop_anims, FadeIn(x_marks), run_time=0.3)
            self.wait(0.2)

            # Reset
            reset_anims = [
                neuron.animate.set_fill(NM_BLUE, opacity=0.3).set_stroke(NM_BLUE)
                for neuron in dropped
            ]
            self.play(*reset_anims, FadeOut(x_marks), run_time=0.2)

        self.wait(0.2)

        # === Step 3: Inference (no dropout) ===
        self.play(FadeOut(net_label), FadeOut(dropout_label), run_time=0.2)

        infer_label = Text("Inference (all neurons active, scaled)", font_size=20, color=NM_GREEN, weight=BOLD)
        infer_label.move_to(UP * 2.8)
        self.play(Write(infer_label), run_time=0.3)

        # All neurons green
        all_green = [
            neuron.animate.set_fill(NM_GREEN, opacity=0.4).set_stroke(NM_GREEN)
            for layer in all_neurons for neuron in layer
        ]
        self.play(*all_green, run_time=0.4)

        # === Step 4: Result ===
        result = VGroup(
            Text("each neuron learns independently useful features", font_size=14, color=NM_TEXT),
            Text("ensemble of 2^n subnetworks \u2192 better generalization", font_size=14, color=NM_YELLOW, weight=BOLD),
        ).arrange(DOWN, buff=0.1)
        result.move_to(DOWN * 2.8)

        self.play(
            LaggedStart(*[FadeIn(r, shift=UP * 0.15) for r in result], lag_ratio=0.2),
            run_time=0.5,
        )
        self.wait(0.8)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.6)
