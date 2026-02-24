"""
no-magic repository overview — 60-second animated montage for LinkedIn.
Silent autoplay optimized: text overlays + animations, no voiceover.
"""

from manim import *
import numpy as np


# === COLOR PALETTE ===
BG_COLOR = "#0d1117"       # GitHub dark
ACCENT_BLUE = "#58a6ff"
ACCENT_GREEN = "#3fb950"
ACCENT_ORANGE = "#d29922"
ACCENT_PURPLE = "#bc8cff"
ACCENT_RED = "#f85149"
ACCENT_TEAL = "#39d353"
TEXT_DIM = "#8b949e"
TEXT_BRIGHT = "#e6edf3"


class NoMagicOverview(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        self.act1_title()
        self.act2_montage()
        self.act3_structure()
        self.act4_cta()

    # =================================================================
    # ACT 1: Title + Tagline (0–7s)
    # =================================================================
    def act1_title(self):
        # Repo name — large, bold
        title = Text("no-magic", font_size=72, weight=BOLD, color=TEXT_BRIGHT)
        # Python icon substitute — a simple ">" prompt glyph
        prompt = Text(">>>", font_size=48, color=ACCENT_GREEN)
        prompt.next_to(title, LEFT, buff=0.4)
        title_group = VGroup(prompt, title).move_to(UP * 0.5)

        tagline = Text(
            'model.fit() isn\'t an explanation',
            font_size=36, color=ACCENT_ORANGE, slant=ITALIC
        )
        tagline.next_to(title_group, DOWN, buff=0.5)

        subtitle = Text(
            "AI/ML from scratch  ·  pure Python  ·  zero dependencies",
            font_size=22, color=TEXT_DIM
        )
        subtitle.next_to(tagline, DOWN, buff=0.4)

        self.play(FadeIn(title, shift=DOWN * 0.3), FadeIn(prompt, shift=RIGHT * 0.3), run_time=1.0)
        self.play(Write(tagline), run_time=1.2)
        self.play(FadeIn(subtitle, shift=UP * 0.2), run_time=0.8)
        self.wait(2.0)

        self.play(
            *[FadeOut(mob, shift=UP * 0.5) for mob in [title_group, tagline, subtitle]],
            run_time=0.6
        )
        self.wait(0.3)

    # =================================================================
    # ACT 2: Algorithm Montage (7–38s)
    # ~5 seconds per algorithm, 6 algorithms
    # =================================================================
    def act2_montage(self):
        self.montage_tokenizer()
        self.montage_attention()
        self.montage_moe()
        self.montage_flash()
        self.montage_diffusion()
        self.montage_gpt()

    # --- microtokenizer: text → tokens ---
    def montage_tokenizer(self):
        label = self._montage_label("microtokenizer.py", "01-foundations")

        raw = Text("understanding", font_size=44, color=TEXT_BRIGHT)
        raw.move_to(UP * 0.5)
        self.play(FadeIn(label), Write(raw), run_time=0.8)
        self.wait(0.3)

        # Split into BPE-style subwords
        pieces = ["under", "##stand", "##ing"]
        colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE]
        tokens = VGroup()
        for piece, color in zip(pieces, colors):
            tok = VGroup(
                RoundedRectangle(
                    width=len(piece) * 0.35 + 0.6, height=0.7,
                    corner_radius=0.15, color=color, fill_opacity=0.25,
                    stroke_width=2
                ),
                Text(piece, font_size=28, color=color)
            )
            tokens.add(tok)

        tokens.arrange(RIGHT, buff=0.25)
        tokens.move_to(DOWN * 0.8)

        # Arrow from raw to tokens
        arrow = Arrow(raw.get_bottom(), tokens.get_top(), buff=0.2, color=TEXT_DIM, stroke_width=2)

        self.play(GrowArrow(arrow), run_time=0.5)
        self.play(
            LaggedStart(*[FadeIn(t, scale=0.8) for t in tokens], lag_ratio=0.2),
            run_time=1.0
        )

        # Show token IDs
        ids = ["[42]", "[187]", "[93]"]
        id_texts = VGroup()
        for i, (tid, tok) in enumerate(zip(ids, tokens)):
            id_text = Text(tid, font_size=20, color=colors[i])
            id_text.next_to(tok, DOWN, buff=0.2)
            id_texts.add(id_text)

        self.play(FadeIn(id_texts, shift=UP * 0.1), run_time=0.6)
        self.wait(1.0)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.4)
        self.wait(0.15)

    # --- microattention: attention matrix ---
    def montage_attention(self):
        label = self._montage_label("microattention.py", "03-systems")

        # Attention matrix as colored grid
        n = 6
        grid = VGroup()
        np.random.seed(42)
        weights = np.random.dirichlet(np.ones(n), size=n)

        for i in range(n):
            for j in range(n):
                w = weights[i][j]
                cell = Square(
                    side_length=0.55,
                    fill_opacity=w * 1.5,
                    fill_color=ACCENT_BLUE,
                    stroke_width=0.5,
                    stroke_color=GREY_D
                )
                cell.move_to(RIGHT * j * 0.58 + DOWN * i * 0.58)
                grid.add(cell)

        grid.move_to(LEFT * 1.5 + DOWN * 0.3)

        # Axis labels
        q_label = Text("Queries", font_size=22, color=ACCENT_BLUE)
        q_label.next_to(grid, LEFT, buff=0.3)
        q_label.rotate(PI / 2)
        k_label = Text("Keys", font_size=22, color=ACCENT_GREEN)
        k_label.next_to(grid, UP, buff=0.3)

        # Formula
        formula = Text(
            "softmax(QKᵀ / √d)",
            font_size=28, color=TEXT_BRIGHT
        )
        formula.move_to(RIGHT * 3 + UP * 0.5)

        # Variants list
        variants = VGroup(
            Text("• Scaled dot-product", font_size=20, color=ACCENT_BLUE),
            Text("• Multi-head", font_size=20, color=ACCENT_GREEN),
            Text("• Grouped-query", font_size=20, color=ACCENT_ORANGE),
            Text("• Multi-query", font_size=20, color=ACCENT_PURPLE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        variants.move_to(RIGHT * 3.2 + DOWN * 1)

        self.play(FadeIn(label), run_time=0.4)
        self.play(
            LaggedStart(*[FadeIn(cell, scale=0.5) for cell in grid], lag_ratio=0.01),
            run_time=1.2
        )
        self.play(FadeIn(q_label), FadeIn(k_label), run_time=0.5)
        self.play(Write(formula), run_time=0.8)
        self.play(
            LaggedStart(*[FadeIn(v, shift=RIGHT * 0.2) for v in variants], lag_ratio=0.2),
            run_time=1.0
        )
        self.wait(1.0)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.4)
        self.wait(0.15)

    # --- micromoe: expert routing ---
    def montage_moe(self):
        label = self._montage_label("micromoe.py", "02-alignment")

        # Input token
        input_box = VGroup(
            RoundedRectangle(width=1.8, height=0.7, corner_radius=0.1,
                             color=TEXT_BRIGHT, fill_opacity=0.1, stroke_width=2),
            Text("input", font_size=22, color=TEXT_BRIGHT)
        ).move_to(LEFT * 4.5)

        # Router
        router = VGroup(
            RoundedRectangle(width=1.8, height=0.9, corner_radius=0.1,
                             color=ACCENT_ORANGE, fill_opacity=0.2, stroke_width=2),
            Text("Router", font_size=22, color=ACCENT_ORANGE)
        ).move_to(LEFT * 1.5)

        # Experts
        expert_colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_PURPLE, ACCENT_RED]
        expert_labels = ["Expert 1", "Expert 2", "Expert 3", "Expert 4"]
        experts = VGroup()
        for i, (name, color) in enumerate(zip(expert_labels, expert_colors)):
            exp = VGroup(
                RoundedRectangle(width=1.6, height=0.65, corner_radius=0.1,
                                 color=color, fill_opacity=0.15, stroke_width=2),
                Text(name, font_size=18, color=color)
            )
            experts.add(exp)

        experts.arrange(DOWN, buff=0.2)
        experts.move_to(RIGHT * 2)

        # Output
        output_box = VGroup(
            RoundedRectangle(width=1.8, height=0.7, corner_radius=0.1,
                             color=ACCENT_TEAL, fill_opacity=0.1, stroke_width=2),
            Text("output", font_size=22, color=ACCENT_TEAL)
        ).move_to(RIGHT * 5)

        self.play(FadeIn(label), FadeIn(input_box, shift=RIGHT * 0.3), run_time=0.5)

        # Input → Router
        a1 = Arrow(input_box.get_right(), router.get_left(), buff=0.15,
                    color=TEXT_DIM, stroke_width=2)
        self.play(FadeIn(router), GrowArrow(a1), run_time=0.5)

        # Router → Experts (top-k=2 routing: highlight 2)
        self.play(
            LaggedStart(*[FadeIn(e, shift=RIGHT * 0.2) for e in experts], lag_ratio=0.12),
            run_time=0.8
        )

        # Routing arrows — highlight top-2
        arrows_to_exp = VGroup()
        for i, exp in enumerate(experts):
            color = expert_colors[i] if i in [0, 2] else GREY_D
            width = 2.5 if i in [0, 2] else 1
            a = Arrow(router.get_right(), exp.get_left(), buff=0.15,
                      color=color, stroke_width=width)
            arrows_to_exp.add(a)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrows_to_exp], lag_ratio=0.08),
            run_time=0.6
        )

        # Highlight selected experts
        self.play(
            experts[0][0].animate.set_fill(opacity=0.4),
            experts[2][0].animate.set_fill(opacity=0.4),
            run_time=0.4
        )

        # Top-k label
        topk = Text("top-k = 2", font_size=20, color=ACCENT_ORANGE)
        topk.next_to(router, DOWN, buff=0.3)
        self.play(FadeIn(topk), run_time=0.4)

        # Selected → Output
        a_out1 = Arrow(experts[0].get_right(), output_box.get_left() + UP * 0.15,
                       buff=0.15, color=expert_colors[0], stroke_width=2)
        a_out2 = Arrow(experts[2].get_right(), output_box.get_left() + DOWN * 0.15,
                       buff=0.15, color=expert_colors[2], stroke_width=2)

        self.play(FadeIn(output_box), GrowArrow(a_out1), GrowArrow(a_out2), run_time=0.5)
        self.wait(1.0)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.4)
        self.wait(0.15)

    # --- microflash: tiled computation ---
    def montage_flash(self):
        label = self._montage_label("microflash.py", "03-systems")

        # Standard attention: full NxN matrix
        title_std = Text("Standard Attention", font_size=24, color=ACCENT_RED)
        title_std.move_to(LEFT * 3.5 + UP * 2.5)

        full_grid = VGroup()
        for i in range(8):
            for j in range(8):
                cell = Square(
                    side_length=0.35, fill_opacity=0.3,
                    fill_color=ACCENT_RED, stroke_width=0.5, stroke_color=GREY_D
                )
                cell.move_to(LEFT * 3.5 + RIGHT * j * 0.38 + DOWN * i * 0.38 + UP * 0.8)
                full_grid.add(cell)

        mem_label = Text("O(N²) memory", font_size=18, color=ACCENT_RED)
        mem_label.next_to(full_grid, DOWN, buff=0.3)

        # Flash attention: tiled blocks
        title_flash = Text("Flash Attention", font_size=24, color=ACCENT_GREEN)
        title_flash.move_to(RIGHT * 3 + UP * 2.5)

        tile_grid = VGroup()
        tile_colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE, ACCENT_PURPLE]
        for bi in range(4):
            for bj in range(4):
                tile = Square(
                    side_length=0.7, fill_opacity=0.2,
                    fill_color=tile_colors[(bi + bj) % 4],
                    stroke_width=1.5,
                    stroke_color=tile_colors[(bi + bj) % 4]
                )
                tile.move_to(RIGHT * 3 + RIGHT * bj * 0.78 + DOWN * bi * 0.78 + UP * 0.8)
                tile_grid.add(tile)

        mem_label2 = Text("O(N) memory — tiled", font_size=18, color=ACCENT_GREEN)
        mem_label2.next_to(tile_grid, DOWN, buff=0.3)

        # Divider
        divider = Line(UP * 2.8, DOWN * 2.2, color=GREY_D, stroke_width=1)

        self.play(FadeIn(label), run_time=0.4)
        self.play(Write(title_std), Write(title_flash), Create(divider), run_time=0.6)

        # Full matrix appears all at once
        self.play(
            LaggedStart(*[FadeIn(c, scale=0.5) for c in full_grid], lag_ratio=0.005),
            run_time=1.0
        )
        self.play(FadeIn(mem_label), run_time=0.4)
        self.wait(0.3)

        # Tiled blocks appear one by one with highlight sweep
        for i, tile in enumerate(tile_grid):
            self.play(FadeIn(tile, scale=0.8), run_time=0.1)

        # Highlight active tile sweep
        highlight = SurroundingRectangle(tile_grid[0], color=YELLOW, stroke_width=3, buff=0.05)
        self.play(Create(highlight), run_time=0.25)
        for i in [1, 5, 10, 15]:
            if i < len(tile_grid):
                self.play(highlight.animate.move_to(tile_grid[i]), run_time=0.2)

        self.play(FadeOut(highlight), FadeIn(mem_label2), run_time=0.4)
        self.wait(0.8)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.4)
        self.wait(0.15)

    # --- microdiffusion: noise → signal ---
    def montage_diffusion(self):
        label = self._montage_label("microdiffusion.py", "01-foundations")

        steps = 6
        np.random.seed(7)
        cell_size = 0.3
        grid_n = 8

        all_grids = []
        for step in range(steps):
            noise_level = 1.0 - step / (steps - 1)
            grid = VGroup()
            for i in range(grid_n):
                for j in range(grid_n):
                    # Blend from noise (random grey) to pattern (checkerboard-like)
                    pattern_val = ((i + j) % 2) * 0.8 + 0.1
                    noise_val = np.random.uniform(0, 1)
                    val = noise_level * noise_val + (1 - noise_level) * pattern_val

                    cell = Square(
                        side_length=cell_size,
                        fill_opacity=val,
                        fill_color=ACCENT_PURPLE,
                        stroke_width=0,
                    )
                    cell.move_to(RIGHT * j * (cell_size + 0.02) + DOWN * i * (cell_size + 0.02))
                    grid.add(cell)

            grid.move_to(ORIGIN)
            all_grids.append(grid)

        # Step labels
        step_labels = [f"t={steps - 1 - i}" for i in range(steps)]

        self.play(FadeIn(label), run_time=0.4)

        # Title
        title = Text("Denoising Process", font_size=28, color=ACCENT_PURPLE)
        title.to_edge(UP, buff=0.6)
        self.play(Write(title), run_time=0.5)

        # Show first (noisiest) grid
        current_grid = all_grids[0]
        step_text = Text(step_labels[0], font_size=22, color=TEXT_DIM)
        step_text.to_edge(DOWN, buff=0.8)

        self.play(FadeIn(current_grid), FadeIn(step_text), run_time=0.6)
        self.wait(0.4)

        # Animate denoising steps
        for i in range(1, steps):
            new_step_text = Text(step_labels[i], font_size=22, color=TEXT_DIM)
            new_step_text.to_edge(DOWN, buff=0.8)

            self.play(
                ReplacementTransform(current_grid, all_grids[i]),
                ReplacementTransform(step_text, new_step_text),
                run_time=0.6
            )
            current_grid = all_grids[i]
            step_text = new_step_text
            self.wait(0.15)

        # Final "clean" flash
        self.play(Indicate(current_grid, color=WHITE, scale_factor=1.05), run_time=0.5)
        self.wait(0.6)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.4)
        self.wait(0.15)

    # --- microgpt: training loop + loss ---
    def montage_gpt(self):
        label = self._montage_label("microgpt.py", "01-foundations")

        # Architecture diagram (simplified)
        layers = [
            ("Embedding", ACCENT_BLUE),
            ("Self-Attention", ACCENT_ORANGE),
            ("Feed-Forward", ACCENT_GREEN),
            ("Softmax", ACCENT_PURPLE),
        ]

        arch = VGroup()
        for name, color in layers:
            block = VGroup(
                RoundedRectangle(width=3, height=0.6, corner_radius=0.1,
                                 color=color, fill_opacity=0.2, stroke_width=2),
                Text(name, font_size=20, color=color)
            )
            arch.add(block)

        arch.arrange(DOWN, buff=0.15)
        arch.move_to(LEFT * 3.5 + DOWN * 0.2)

        arch_title = Text("Transformer", font_size=24, color=TEXT_BRIGHT)
        arch_title.next_to(arch, UP, buff=0.3)

        # Loss curve on right
        axes = Axes(
            x_range=[0, 50, 10],
            y_range=[0, 4, 1],
            x_length=4.5,
            y_length=2.8,
            axis_config={"color": GREY_D, "stroke_width": 1, "include_ticks": False},
        ).move_to(RIGHT * 3 + DOWN * 0.2)

        x_label = Text("epoch", font_size=16, color=TEXT_DIM)
        x_label.next_to(axes, DOWN, buff=0.15)
        y_label = Text("loss", font_size=16, color=TEXT_DIM)
        y_label.next_to(axes, LEFT, buff=0.15)

        # Exponential decay loss curve
        loss_curve = axes.plot(
            lambda x: 3.5 * np.exp(-0.08 * x) + 0.3,
            x_range=[0, 50],
            color=ACCENT_RED,
            stroke_width=2.5
        )

        loss_title = Text("Training Loss", font_size=22, color=ACCENT_RED)
        loss_title.next_to(axes, UP, buff=0.3)

        self.play(FadeIn(label), run_time=0.4)

        # Build architecture
        self.play(Write(arch_title), run_time=0.4)
        arrows_arch = VGroup()
        for i, block in enumerate(arch):
            self.play(FadeIn(block, shift=DOWN * 0.2), run_time=0.3)
            if i < len(arch) - 1:
                a = Arrow(
                    arch[i].get_bottom(), arch[i + 1].get_top(),
                    buff=0.05, color=GREY_D, stroke_width=1.5
                )
                arrows_arch.add(a)
                self.play(GrowArrow(a), run_time=0.15)

        # Draw loss curve
        self.play(Write(loss_title), run_time=0.4)
        self.play(Create(axes), FadeIn(x_label), FadeIn(y_label), run_time=0.5)
        self.play(Create(loss_curve), run_time=1.8)

        # Generated text sample
        gen_text = Text('"Aelira\nKarthen\nZylox"', font_size=20, color=ACCENT_TEAL)
        gen_text.next_to(axes, DOWN, buff=0.6)
        gen_label = Text("Generated names ↑", font_size=16, color=TEXT_DIM)
        gen_label.next_to(gen_text, DOWN, buff=0.15)

        self.play(FadeIn(gen_text, shift=UP * 0.2), FadeIn(gen_label), run_time=0.6)
        self.wait(0.8)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.4)
        self.wait(0.15)

    # =================================================================
    # ACT 3: Repository Structure + Stats (38–52s)
    # =================================================================
    def act3_structure(self):
        # Section title
        section = Text("30 scripts  ·  3 tiers  ·  zero dependencies", font_size=30, color=TEXT_BRIGHT)
        section.to_edge(UP, buff=0.5)
        self.play(Write(section), run_time=1.0)

        # Three tier columns
        tiers = [
            ("01-foundations", "11 scripts", ACCENT_BLUE, [
                "microgpt", "micrornn", "microtokenizer",
                "microembedding", "microrag", "microdiffusion",
                "microvae", "microbert", "microconv",
                "microgan", "microoptimizer"
            ]),
            ("02-alignment", "9 scripts", ACCENT_GREEN, [
                "microlora", "microdpo", "microppo",
                "micromoe", "microgrpo", "microreinforce",
                "microqlora", "microbatchnorm", "microdropout"
            ]),
            ("03-systems", "10 scripts", ACCENT_ORANGE, [
                "microattention", "microkv", "microquant",
                "microflash", "microbeam", "microrope",
                "microssm", "micropaged", "microparallel",
                "microcheckpoint"
            ]),
        ]

        columns = VGroup()
        for tier_name, count, color, scripts in tiers:
            # Header
            header = VGroup(
                Text(tier_name, font_size=24, weight=BOLD, color=color),
                Text(count, font_size=18, color=TEXT_DIM),
            ).arrange(DOWN, buff=0.15)

            # Script list
            script_list = VGroup()
            for s in scripts:
                t = Text(s + ".py", font_size=13, color=color)
                t.set_opacity(0.75)
                script_list.add(t)
            script_list.arrange(DOWN, buff=0.08, aligned_edge=LEFT)

            col = VGroup(header, script_list).arrange(DOWN, buff=0.3)
            columns.add(col)

        columns.arrange(RIGHT, buff=1.0, aligned_edge=UP)
        columns.move_to(DOWN * 0.3)

        # Animate columns appearing
        for col in columns:
            header, scripts = col[0], col[1]
            self.play(FadeIn(header, shift=UP * 0.2), run_time=0.5)
            self.play(
                LaggedStart(*[FadeIn(s, shift=RIGHT * 0.1) for s in scripts], lag_ratio=0.04),
                run_time=0.7
            )

        self.wait(0.6)

        # Key stats bar at bottom — includes repo metrics
        stats = VGroup(
            Text("★ 500+", font_size=24, weight=BOLD, color=ACCENT_ORANGE),
            Text("·", font_size=22, color=GREY_D),
            Text("55 forks", font_size=22, color=TEXT_BRIGHT),
            Text("·", font_size=22, color=GREY_D),
            Text("pure Python", font_size=22, color=ACCENT_TEAL),
            Text("·", font_size=22, color=GREY_D),
            Text("zero pip install", font_size=22, color=ACCENT_TEAL),
        ).arrange(RIGHT, buff=0.3)
        stats.to_edge(DOWN, buff=0.5)

        self.play(FadeIn(stats, shift=UP * 0.2), run_time=0.6)
        self.wait(2.0)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.5)
        self.wait(0.3)

    # =================================================================
    # ACT 4: CTA + GitHub URL (52–62s)
    # =================================================================
    def act4_cta(self):
        cta = Text("Clone and run in 30 seconds", font_size=36, color=TEXT_BRIGHT, weight=BOLD)
        cta.move_to(UP * 1.5)

        # Terminal-style command
        terminal_bg = RoundedRectangle(
            width=10, height=1.8, corner_radius=0.2,
            color="#161b22", fill_opacity=0.9, stroke_width=1, stroke_color=GREY_D
        )
        terminal_bg.move_to(DOWN * 0.2)

        cmd1 = Text("$ git clone github.com/Mathews-Tom/no-magic", font_size=20, color=ACCENT_GREEN)
        cmd2 = Text("$ python 01-foundations/microgpt.py", font_size=20, color=ACCENT_GREEN)
        cmds = VGroup(cmd1, cmd2).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        cmds.move_to(terminal_bg)

        url = Text(
            "github.com/Mathews-Tom/no-magic",
            font_size=32, weight=BOLD, color=ACCENT_BLUE
        )
        url.move_to(DOWN * 2)

        # Star icon substitute
        star = Text("★", font_size=28, color=ACCENT_ORANGE)
        star.next_to(url, RIGHT, buff=0.4)

        self.play(Write(cta), run_time=1.0)
        self.play(FadeIn(terminal_bg), run_time=0.4)
        self.play(Write(cmd1), run_time=1.0)
        self.play(Write(cmd2), run_time=0.8)
        self.wait(0.6)

        self.play(FadeIn(url, shift=UP * 0.2), FadeIn(star, scale=1.5), run_time=0.7)

        # Hold for screenshot
        self.wait(4.5)

    # =================================================================
    # Helpers
    # =================================================================
    def _montage_label(self, filename: str, tier: str) -> VGroup:
        """Top-right label showing current script name and tier."""
        name = Text(filename, font_size=20, weight=BOLD, color=TEXT_BRIGHT)
        tier_text = Text(tier, font_size=16, color=TEXT_DIM)
        group = VGroup(name, tier_text).arrange(DOWN, buff=0.1, aligned_edge=RIGHT)
        group.to_corner(UR, buff=0.4)
        return group
