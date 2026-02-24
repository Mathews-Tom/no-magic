"""
Scene: Proximal Policy Optimization (PPO)
Script: microppo.py
Description: The full RLHF loop — pretrain, reward model, clipped policy gradient
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE, NM_PURPLE
from manim import *


class PPOScene(NoMagicScene):
    title_text = "Proximal Policy Optimization"
    subtitle_text = "Clipped policy gradient — the engine behind RLHF"

    def animate(self):
        # === Step 1: Show the 3-model RLHF pipeline ===
        pipeline_label = Text("RLHF Pipeline", font_size=22, color=NM_PRIMARY, weight=BOLD)
        pipeline_label.move_to(UP * 2.8)
        self.play(Write(pipeline_label), run_time=0.4)

        # Three model boxes
        model_data = [
            ("Policy\n(LM)", NM_GREEN, "generates text"),
            ("Reward\nModel", NM_YELLOW, "scores quality"),
            ("Value\nFunction", NM_BLUE, "estimates return"),
        ]
        model_boxes = VGroup()
        model_labels = VGroup()
        for name, color, desc in model_data:
            box = RoundedRectangle(
                corner_radius=0.12, width=2.0, height=1.2,
                color=color, fill_opacity=0.15, stroke_width=2,
            )
            name_label = Text(name, font_size=15, color=color, weight=BOLD)
            name_label.move_to(box.get_center() + UP * 0.1)
            desc_label = Text(desc, font_size=11, color=NM_GRID)
            desc_label.next_to(box, DOWN, buff=0.08)
            model_boxes.add(VGroup(box, name_label))
            model_labels.add(desc_label)

        model_boxes.arrange(RIGHT, buff=0.6)
        model_boxes.move_to(UP * 1.2)

        for i, desc in enumerate(model_labels):
            desc.next_to(model_boxes[i], DOWN, buff=0.08)

        self.play(
            LaggedStart(*[FadeIn(m, shift=UP * 0.15) for m in model_boxes], lag_ratio=0.15),
            LaggedStart(*[FadeIn(d) for d in model_labels], lag_ratio=0.15),
            run_time=1.0,
        )
        self.wait(0.6)

        # === Step 2: Show the PPO loop ===
        loop_steps = [
            ("1. Generate", NM_GREEN, "policy produces completions"),
            ("2. Score", NM_YELLOW, "reward model rates each"),
            ("3. Advantage", NM_BLUE, "A = reward - value baseline"),
            ("4. Update", NM_PRIMARY, "clip ratio, update policy"),
        ]

        step_items = VGroup()
        for step_name, color, detail in loop_steps:
            step_label = Text(step_name, font_size=16, color=color, weight=BOLD)
            step_detail = Text(detail, font_size=12, color=NM_TEXT)
            step_detail.next_to(step_label, RIGHT, buff=0.2)
            step_items.add(VGroup(step_label, step_detail))

        step_items.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        step_items.move_to(LEFT * 3.0 + DOWN * 1.5)

        for item in step_items:
            self.play(FadeIn(item, shift=RIGHT * 0.2), run_time=0.4)
        self.wait(0.6)

        # === Step 3: Show the clipping mechanism ===
        clip_title = Text("PPO Clipping", font_size=18, color=NM_PRIMARY, weight=BOLD)
        clip_title.move_to(RIGHT * 3.0 + DOWN * 0.6)
        self.play(Write(clip_title), run_time=0.4)

        # Visualize the clipped ratio
        clip_box = RoundedRectangle(
            corner_radius=0.1, width=4.0, height=2.2,
            color=NM_GRID, fill_opacity=0.05, stroke_width=1,
        )
        clip_box.move_to(RIGHT * 3.0 + DOWN * 2.0)

        # Number line for ratio
        ratio_line = Line(
            clip_box.get_left() + RIGHT * 0.3,
            clip_box.get_right() + LEFT * 0.3,
            color=NM_GRID, stroke_width=1.5,
        )
        ratio_line.move_to(clip_box.get_center() + UP * 0.3)

        # Tick marks for 1-eps, 1, 1+eps
        ticks = VGroup()
        tick_labels_group = VGroup()
        eps = 0.2
        for val, label_text in [(1 - eps, "1-\u03b5"), (1.0, "1"), (1 + eps, "1+\u03b5")]:
            # Map value to position on the line
            t = (val - 0.5) / 1.0  # normalize to 0..1 range
            pos = ratio_line.point_from_proportion(t)
            tick = Line(pos + UP * 0.1, pos + DOWN * 0.1, color=NM_TEXT, stroke_width=1.5)
            label = Text(label_text, font_size=11, color=NM_TEXT)
            label.next_to(tick, DOWN, buff=0.08)
            ticks.add(tick)
            tick_labels_group.add(label)

        # Clip zone highlight
        left_pos = ratio_line.point_from_proportion((1 - eps - 0.5) / 1.0)
        right_pos = ratio_line.point_from_proportion((1 + eps - 0.5) / 1.0)
        clip_zone = Rectangle(
            width=right_pos[0] - left_pos[0],
            height=0.6,
            color=NM_GREEN, fill_opacity=0.15, stroke_width=0,
        )
        clip_zone.move_to((left_pos + right_pos) / 2 + UP * 0.3)

        clip_label = Text("allowed update range", font_size=11, color=NM_GREEN)
        clip_label.next_to(clip_zone, UP, buff=0.05)

        no_update = Text("clipped (no update)", font_size=10, color=NM_PRIMARY)
        no_update.next_to(clip_box, DOWN, buff=0.08)

        self.play(
            FadeIn(clip_box), Create(ratio_line),
            LaggedStart(*[Create(t) for t in ticks], lag_ratio=0.1),
            LaggedStart(*[FadeIn(l) for l in tick_labels_group], lag_ratio=0.1),
            FadeIn(clip_zone), FadeIn(clip_label),
            FadeIn(no_update),
            run_time=1.0,
        )

        # Animate a ratio dot moving and getting clipped
        dot = Dot(ratio_line.point_from_proportion(0.5), radius=0.08, color=NM_YELLOW)
        self.play(FadeIn(dot), run_time=0.4)
        # Move right (policy improving) — within clip range
        self.play(dot.animate.move_to(ratio_line.point_from_proportion(0.65)), run_time=0.4)
        # Move further — hits clip boundary
        self.play(dot.animate.move_to(ratio_line.point_from_proportion(0.7)), run_time=0.4)
        # Flash red at boundary
        self.play(dot.animate.set_color(NM_PRIMARY), run_time=0.4)
        self.play(dot.animate.move_to(ratio_line.point_from_proportion(0.7)), run_time=0.4)

        self.wait(0.6)

        # === Step 4: Summary ===
        summary = Text(
            "small steps only — prevents reward hacking",
            font_size=16, color=NM_YELLOW, weight=BOLD,
        )
        summary.move_to(DOWN * 3.5)
        self.play(FadeIn(summary, shift=UP * 0.15), run_time=0.6)
        self.wait(1.6)

        # Cleanup
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.9)
