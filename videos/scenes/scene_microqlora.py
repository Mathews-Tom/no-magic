"""
Scene: QLoRA
Script: microqlora.py
Description: 4-bit quantized base model + full-precision LoRA adapters
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE, NM_PURPLE
from manim import *


class QLoRAScene(NoMagicScene):
    title_text = "QLoRA"
    subtitle_text = "4-bit quantized weights + full-precision LoRA adapters"

    def animate(self):
        # === Step 1: Show standard fine-tuning memory cost ===
        standard_label = Text("Standard Fine-tuning", font_size=18, color=NM_PRIMARY, weight=BOLD)
        standard_label.move_to(LEFT * 3.5 + UP * 2.5)

        std_bar = Rectangle(
            width=2.5, height=0.6,
            color=NM_PRIMARY, fill_opacity=0.3, stroke_width=1.5,
        )
        std_bar.move_to(LEFT * 3.5 + UP * 1.5)
        std_text = Text("16-bit weights\n+ gradients\n+ optimizer states", font_size=11, color=NM_TEXT)
        std_text.move_to(std_bar.get_center())
        std_mem = Text("~48 GB for 7B model", font_size=13, color=NM_PRIMARY)
        std_mem.next_to(std_bar, DOWN, buff=0.15)

        self.play(
            Write(standard_label),
            FadeIn(std_bar), FadeIn(std_text), FadeIn(std_mem),
            run_time=0.8,
        )
        self.wait(0.4)

        # === Step 2: Show QLoRA architecture ===
        qlora_label = Text("QLoRA", font_size=18, color=NM_GREEN, weight=BOLD)
        qlora_label.move_to(RIGHT * 3.0 + UP * 2.5)
        self.play(Write(qlora_label), run_time=0.4)

        # Quantized base model (large, blue)
        base_box = RoundedRectangle(
            corner_radius=0.1, width=3.5, height=1.8,
            color=NM_BLUE, fill_opacity=0.1, stroke_width=2,
        )
        base_box.move_to(RIGHT * 3.0 + UP * 0.5)
        base_label = Text("Base Model (4-bit quantized)", font_size=14, color=NM_BLUE, weight=BOLD)
        base_label.move_to(base_box.get_center() + UP * 0.4)

        # Frozen indicator
        frozen = Text("FROZEN", font_size=11, color=NM_GRID, weight=BOLD)
        frozen.move_to(base_box.get_center())

        # 4-bit cells inside base model
        quant_cells = VGroup()
        for i in range(8):
            cell = Rectangle(
                width=0.35, height=0.3,
                color=NM_BLUE, fill_opacity=0.2, stroke_width=0.5,
            )
            bit_label = Text("4b", font_size=8, color=NM_GRID)
            bit_label.move_to(cell.get_center())
            quant_cells.add(VGroup(cell, bit_label))
        quant_cells.arrange(RIGHT, buff=0.05)
        quant_cells.move_to(base_box.get_center() + DOWN * 0.45)

        self.play(
            FadeIn(base_box), Write(base_label), FadeIn(frozen), FadeIn(quant_cells),
            run_time=0.8,
        )

        # LoRA adapters (small, green)
        lora_a = RoundedRectangle(
            corner_radius=0.06, width=0.6, height=1.2,
            color=NM_GREEN, fill_opacity=0.25, stroke_width=1.5,
        )
        a_label = Text("A", font_size=16, color=NM_GREEN, weight=BOLD)
        a_label.move_to(lora_a.get_center())
        a_dim = Text("d\u00d7r", font_size=10, color=NM_GRID)
        a_dim.next_to(lora_a, DOWN, buff=0.05)
        lora_a_group = VGroup(lora_a, a_label, a_dim)
        lora_a_group.next_to(base_box, RIGHT, buff=0.3).shift(UP * 0.2)

        lora_b = RoundedRectangle(
            corner_radius=0.06, width=1.2, height=0.6,
            color=NM_GREEN, fill_opacity=0.25, stroke_width=1.5,
        )
        b_label = Text("B", font_size=16, color=NM_GREEN, weight=BOLD)
        b_label.move_to(lora_b.get_center())
        b_dim = Text("r\u00d7d", font_size=10, color=NM_GRID)
        b_dim.next_to(lora_b, DOWN, buff=0.05)
        lora_b_group = VGroup(lora_b, b_label, b_dim)
        lora_b_group.next_to(lora_a_group, DOWN, buff=0.3)

        fp16_label = Text("FP16", font_size=10, color=NM_GREEN, weight=BOLD)
        fp16_label.next_to(lora_a, LEFT, buff=0.08)

        self.play(
            FadeIn(lora_a_group, shift=LEFT * 0.2),
            FadeIn(lora_b_group, shift=LEFT * 0.2),
            FadeIn(fp16_label),
            run_time=0.6,
        )
        self.wait(0.4)

        # === Step 3: Memory comparison ===
        qlora_mem = Text("~6 GB for 7B model", font_size=13, color=NM_GREEN, weight=BOLD)
        qlora_mem.move_to(RIGHT * 3.0 + DOWN * 1.5)

        savings = Text("8x memory reduction \u2192 fine-tune on a single GPU", font_size=16, color=NM_YELLOW, weight=BOLD)
        savings.move_to(DOWN * 2.5)

        technique = Text(
            "4-bit NormalFloat (NF4) + double quantization + paged optimizers",
            font_size=12, color=NM_GRID,
        )
        technique.move_to(DOWN * 3.0)

        self.play(FadeIn(qlora_mem), run_time=0.4)
        self.play(FadeIn(savings, shift=UP * 0.15), FadeIn(technique), run_time=0.6)
        self.wait(1.6)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.9)
