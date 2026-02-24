"""
Scene: Group Relative Policy Optimization (GRPO)
Script: microgrpo.py
Description: DeepSeek's simplification — group-based reward normalization, no value function
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE, NM_PURPLE
from manim import *


class GRPOScene(NoMagicScene):
    title_text = "GRPO"
    subtitle_text = "Group-relative rewards — no critic network needed"

    def animate(self):
        # === Step 1: Show PPO's 3 models vs GRPO's simplification ===
        ppo_label = Text("PPO (3 models)", font_size=18, color=NM_ORANGE, weight=BOLD)
        ppo_label.move_to(LEFT * 3.5 + UP * 2.5)

        ppo_models = VGroup()
        for name, color in [("Policy", NM_GREEN), ("Reward", NM_YELLOW), ("Value", NM_BLUE)]:
            box = RoundedRectangle(
                corner_radius=0.08, width=1.6, height=0.6,
                color=color, fill_opacity=0.15, stroke_width=1.5,
            )
            label = Text(name, font_size=13, color=color)
            label.move_to(box.get_center())
            ppo_models.add(VGroup(box, label))
        ppo_models.arrange(DOWN, buff=0.15)
        ppo_models.move_to(LEFT * 3.5 + UP * 0.8)

        grpo_label = Text("GRPO (2 models)", font_size=18, color=NM_GREEN, weight=BOLD)
        grpo_label.move_to(RIGHT * 3.5 + UP * 2.5)

        grpo_models = VGroup()
        for name, color in [("Policy", NM_GREEN), ("Reward", NM_YELLOW)]:
            box = RoundedRectangle(
                corner_radius=0.08, width=1.6, height=0.6,
                color=color, fill_opacity=0.15, stroke_width=1.5,
            )
            label = Text(name, font_size=13, color=color)
            label.move_to(box.get_center())
            grpo_models.add(VGroup(box, label))
        grpo_models.arrange(DOWN, buff=0.15)
        grpo_models.move_to(RIGHT * 3.5 + UP * 1.1)

        # Crossed out value function
        no_value = Text("No Value Function", font_size=13, color=NM_PRIMARY)
        cross = Line(
            no_value.get_left() + LEFT * 0.1 + DOWN * 0.02,
            no_value.get_right() + RIGHT * 0.1 + DOWN * 0.02,
            color=NM_PRIMARY, stroke_width=2,
        )
        no_value_group = VGroup(no_value, cross)
        no_value_group.next_to(grpo_models, DOWN, buff=0.2)

        self.play(
            Write(ppo_label), FadeIn(ppo_models),
            Write(grpo_label), FadeIn(grpo_models), FadeIn(no_value_group),
            run_time=0.6,
        )
        self.wait(0.3)

        # === Step 2: Show group-based reward normalization ===
        self.play(
            *[FadeOut(m) for m in [ppo_label, ppo_models, grpo_label, grpo_models, no_value_group]],
            run_time=0.3,
        )

        group_label = Text("Group-Relative Advantage", font_size=20, color=NM_GREEN, weight=BOLD)
        group_label.move_to(UP * 2.5)
        self.play(Write(group_label), run_time=0.3)

        # Generate a group of completions
        completions = [
            ("output A", 0.8, NM_GREEN),
            ("output B", 0.3, NM_YELLOW),
            ("output C", 0.6, NM_BLUE),
            ("output D", 0.2, NM_ORANGE),
            ("output E", 0.9, NM_GREEN),
        ]

        comp_items = VGroup()
        bars = VGroup()
        for name, reward, color in completions:
            label = Text(name, font_size=13, color=NM_TEXT)
            bar = Rectangle(
                width=reward * 3, height=0.3,
                color=color, fill_opacity=0.4, stroke_width=1,
            )
            bar.next_to(label, RIGHT, buff=0.2)
            score = Text(f"r={reward:.1f}", font_size=11, color=NM_GRID)
            score.next_to(bar, RIGHT, buff=0.1)
            comp_items.add(VGroup(label, bar, score))
            bars.add(bar)

        comp_items.arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        comp_items.move_to(LEFT * 1.5 + UP * 0.5)

        self.play(
            LaggedStart(*[FadeIn(c, shift=RIGHT * 0.2) for c in comp_items], lag_ratio=0.08),
            run_time=0.6,
        )
        self.wait(0.2)

        # Show group mean and normalization
        mean_val = sum(r for _, r, _ in completions) / len(completions)
        mean_line = DashedLine(
            comp_items.get_left() + LEFT * 0.5 + DOWN * 0.0,
            comp_items.get_right() + RIGHT * 0.5 + DOWN * 0.0,
            color=NM_PRIMARY, stroke_width=1.5,
        )
        mean_label = Text(f"group mean = {mean_val:.2f}", font_size=13, color=NM_PRIMARY)
        mean_label.next_to(mean_line, RIGHT, buff=0.2)

        self.play(Create(mean_line), FadeIn(mean_label), run_time=0.3)

        # Show normalized advantages
        adv_label = Text("Advantage = (reward - mean) / std", font_size=15, color=NM_YELLOW, weight=BOLD)
        adv_label.move_to(DOWN * 1.8)

        insight = Text(
            "no value function — just compare within the group",
            font_size=14, color=NM_GREEN,
        )
        insight.move_to(DOWN * 2.3)

        self.play(Write(adv_label), run_time=0.3)
        self.play(FadeIn(insight, shift=UP * 0.15), run_time=0.3)
        self.wait(0.3)

        # === Step 3: Result ===
        result = Text(
            "simpler, cheaper, same alignment quality (DeepSeek-R1)",
            font_size=16, color=NM_YELLOW, weight=BOLD,
        )
        result.move_to(DOWN * 3.2)
        self.play(FadeIn(result, shift=UP * 0.15), run_time=0.4)
        self.wait(0.8)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.6)
