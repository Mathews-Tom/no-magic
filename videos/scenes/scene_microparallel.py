"""
Scene: Tensor & Pipeline Parallelism
Script: microparallel.py
Description: Distributing model layers and tensor slices across multiple devices
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE, NM_PURPLE
from manim import *


class ParallelScene(NoMagicScene):
    title_text = "Model Parallelism"
    subtitle_text = "Tensor slicing + pipeline stages across devices"

    def animate(self):
        # === Step 1: Show model too large for one GPU ===
        single_gpu = RoundedRectangle(
            corner_radius=0.1, width=2.0, height=1.2,
            color=NM_BLUE, fill_opacity=0.12, stroke_width=2,
        )
        gpu_label = Text("GPU 0", font_size=14, color=NM_BLUE, weight=BOLD)
        gpu_label.move_to(single_gpu.get_center() + UP * 0.15)
        gpu_mem = Text("24 GB", font_size=11, color=NM_GRID)
        gpu_mem.move_to(single_gpu.get_center() + DOWN * 0.15)
        gpu_group = VGroup(single_gpu, gpu_label, gpu_mem)
        gpu_group.move_to(UP * 2.0)

        model_label = Text("70B Model = 140 GB", font_size=16, color=NM_PRIMARY, weight=BOLD)
        model_label.next_to(gpu_group, RIGHT, buff=0.5)

        no_fit = Text("does not fit", font_size=13, color=NM_PRIMARY)
        no_fit.next_to(model_label, DOWN, buff=0.1)

        self.play(FadeIn(gpu_group), FadeIn(model_label), FadeIn(no_fit), run_time=0.4)
        self.wait(0.2)

        self.play(
            FadeOut(gpu_group), FadeOut(model_label), FadeOut(no_fit),
            run_time=0.2,
        )

        # === Step 2: Tensor Parallelism ===
        tp_label = Text("Tensor Parallelism", font_size=20, color=NM_GREEN, weight=BOLD)
        tp_label.move_to(LEFT * 3.5 + UP * 2.5)
        self.play(Write(tp_label), run_time=0.3)

        tp_desc = Text("split weight matrices across GPUs", font_size=12, color=NM_GRID)
        tp_desc.next_to(tp_label, DOWN, buff=0.1)
        self.play(FadeIn(tp_desc), run_time=0.2)

        # Weight matrix split into 4 column slices
        full_matrix = VGroup()
        colors = [NM_BLUE, NM_GREEN, NM_YELLOW, NM_ORANGE]
        for i in range(4):
            col = Rectangle(
                width=0.6, height=1.6,
                color=colors[i], fill_opacity=0.2, stroke_width=1.5,
            )
            lab = Text(f"GPU {i}", font_size=9, color=colors[i])
            lab.move_to(col.get_center())
            full_matrix.add(VGroup(col, lab))

        full_matrix.arrange(RIGHT, buff=0.04)
        full_matrix.move_to(LEFT * 3.5 + UP * 0.5)

        w_label = Text("W (weight matrix)", font_size=11, color=NM_TEXT)
        w_label.next_to(full_matrix, DOWN, buff=0.1)

        allreduce = Text("AllReduce to combine outputs", font_size=11, color=NM_PRIMARY)
        allreduce.next_to(w_label, DOWN, buff=0.08)

        self.play(
            LaggedStart(*[FadeIn(c, shift=DOWN * 0.1) for c in full_matrix], lag_ratio=0.1),
            run_time=0.4,
        )
        self.play(FadeIn(w_label), FadeIn(allreduce), run_time=0.2)
        self.wait(0.2)

        # === Step 3: Pipeline Parallelism ===
        pp_label = Text("Pipeline Parallelism", font_size=20, color=NM_YELLOW, weight=BOLD)
        pp_label.move_to(RIGHT * 3.5 + UP * 2.5)
        self.play(Write(pp_label), run_time=0.3)

        pp_desc = Text("split layers across GPUs", font_size=12, color=NM_GRID)
        pp_desc.next_to(pp_label, DOWN, buff=0.1)
        self.play(FadeIn(pp_desc), run_time=0.2)

        # Pipeline stages
        stages = VGroup()
        stage_arrows = VGroup()
        for i in range(4):
            stage = RoundedRectangle(
                corner_radius=0.08, width=1.2, height=0.7,
                color=colors[i], fill_opacity=0.15, stroke_width=1.5,
            )
            lab = VGroup(
                Text(f"GPU {i}", font_size=10, color=colors[i], weight=BOLD),
                Text(f"Layers {i*4}-{i*4+3}", font_size=9, color=NM_GRID),
            ).arrange(DOWN, buff=0.03)
            lab.move_to(stage.get_center())
            stages.add(VGroup(stage, lab))

        stages.arrange(DOWN, buff=0.15)
        stages.move_to(RIGHT * 3.5 + UP * 0.3)

        for i in range(3):
            arr = Arrow(
                stages[i].get_bottom(), stages[i + 1].get_top(),
                color=NM_GRID, stroke_width=1.5, buff=0.05, tip_length=0.1,
            )
            stage_arrows.add(arr)

        self.play(
            LaggedStart(*[FadeIn(s) for s in stages], lag_ratio=0.1),
            LaggedStart(*[GrowArrow(a) for a in stage_arrows], lag_ratio=0.1),
            run_time=0.5,
        )

        bubble_note = Text("micro-batches fill the pipeline bubble", font_size=11, color=NM_YELLOW)
        bubble_note.next_to(stages, RIGHT, buff=0.2)
        self.play(FadeIn(bubble_note), run_time=0.2)
        self.wait(0.2)

        # === Step 4: Combined result ===
        result = VGroup(
            Text("Tensor ∥ = split matrices within a layer", font_size=14, color=NM_GREEN),
            Text("Pipeline ∥ = split layers across stages", font_size=14, color=NM_YELLOW),
            Text("combined: train 70B+ parameter models", font_size=14, color=NM_PRIMARY, weight=BOLD),
        ).arrange(DOWN, buff=0.08)
        result.move_to(DOWN * 2.8)

        self.play(
            LaggedStart(*[FadeIn(r, shift=UP * 0.1) for r in result], lag_ratio=0.15),
            run_time=0.5,
        )
        self.wait(0.8)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.6)
