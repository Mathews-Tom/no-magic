"""
Scene: REINFORCE Policy Gradient
Script: microreinforce.py
Description: Log-probability weighting turns reward signals into gradient updates
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE, NM_PURPLE
from manim import *


class REINFORCEScene(NoMagicScene):
    title_text = "REINFORCE"
    subtitle_text = "Policy gradient — log P(a) * reward drives learning"

    def animate(self):
        # === Step 1: Show the policy network ===
        policy_box = RoundedRectangle(
            corner_radius=0.12, width=2.5, height=1.4,
            color=NM_GREEN, fill_opacity=0.12, stroke_width=2,
        )
        policy_label = Text("Policy π(a|s)", font_size=18, color=NM_GREEN, weight=BOLD)
        policy_label.move_to(policy_box.get_center() + UP * 0.2)
        policy_desc = Text("neural network", font_size=12, color=NM_GRID)
        policy_desc.move_to(policy_box.get_center() + DOWN * 0.2)
        policy_group = VGroup(policy_box, policy_label, policy_desc)
        policy_group.move_to(LEFT * 3.5 + UP * 1.0)

        # State input
        state_dot = Circle(radius=0.3, color=NM_BLUE, fill_opacity=0.2, stroke_width=1.5)
        state_label = Text("state s", font_size=13, color=NM_BLUE)
        state_label.move_to(state_dot.get_center())
        state_group = VGroup(state_dot, state_label)
        state_group.move_to(LEFT * 6.0 + UP * 1.0)

        # Action output
        action_dot = Circle(radius=0.3, color=NM_YELLOW, fill_opacity=0.2, stroke_width=1.5)
        action_label = Text("action a", font_size=13, color=NM_YELLOW)
        action_label.move_to(action_dot.get_center())
        action_group = VGroup(action_dot, action_label)
        action_group.move_to(LEFT * 0.5 + UP * 1.0)

        arr1 = Arrow(state_dot.get_right(), policy_box.get_left(), color=NM_GRID, stroke_width=1.5, buff=0.1, tip_length=0.1)
        arr2 = Arrow(policy_box.get_right(), action_dot.get_left(), color=NM_GRID, stroke_width=1.5, buff=0.1, tip_length=0.1)

        self.play(
            FadeIn(state_group), FadeIn(policy_group), FadeIn(action_group),
            GrowArrow(arr1), GrowArrow(arr2),
            run_time=0.5,
        )
        self.wait(0.2)

        # === Step 2: Show reward signal ===
        reward_box = RoundedRectangle(
            corner_radius=0.08, width=1.5, height=0.6,
            color=NM_ORANGE, fill_opacity=0.15, stroke_width=1.5,
        )
        reward_label = Text("Reward R", font_size=14, color=NM_ORANGE, weight=BOLD)
        reward_label.move_to(reward_box.get_center())
        reward_group = VGroup(reward_box, reward_label)
        reward_group.move_to(RIGHT * 2.5 + UP * 1.0)

        arr3 = Arrow(action_dot.get_right(), reward_box.get_left(), color=NM_GRID, stroke_width=1.5, buff=0.1, tip_length=0.1)
        self.play(GrowArrow(arr3), FadeIn(reward_group), run_time=0.3)

        # === Step 3: REINFORCE gradient formula ===
        formula_label = Text("REINFORCE Update", font_size=20, color=NM_PRIMARY, weight=BOLD)
        formula_label.move_to(DOWN * 0.5)
        self.play(Write(formula_label), run_time=0.3)

        gradient = VGroup(
            Text("∇θ J = E[ ∇θ log π(a|s) · R ]", font_size=18, color=NM_YELLOW, weight=BOLD),
            Text("log probability × reward = gradient direction", font_size=13, color=NM_GREEN),
        ).arrange(DOWN, buff=0.1)
        gradient.move_to(DOWN * 1.3)

        self.play(
            LaggedStart(*[FadeIn(g) for g in gradient], lag_ratio=0.2),
            run_time=0.5,
        )
        self.wait(0.3)

        # === Step 4: High variance problem + baseline ===
        problem = Text("Problem: high variance in R", font_size=14, color=NM_PRIMARY)
        problem.move_to(DOWN * 2.2)
        self.play(FadeIn(problem), run_time=0.2)

        solution = VGroup(
            Text("Solution: subtract baseline b", font_size=14, color=NM_GREEN, weight=BOLD),
            Text("∇θ J = E[ ∇θ log π(a|s) · (R - b) ]", font_size=16, color=NM_YELLOW, weight=BOLD),
            Text("b = running average reward → reduces variance, same expected gradient", font_size=12, color=NM_GRID),
        ).arrange(DOWN, buff=0.08)
        solution.move_to(DOWN * 3.2)

        self.play(
            LaggedStart(*[FadeIn(s, shift=UP * 0.1) for s in solution], lag_ratio=0.15),
            run_time=0.5,
        )
        self.wait(0.8)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.6)
