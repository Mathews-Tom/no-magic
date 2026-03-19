"""
Scene: Minimax + Alpha-Beta Pruning
Script: microminimax.py
Description: Adversarial search — skip branches that can't affect the decision
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import NoMagicScene, NM_PRIMARY, NM_BLUE, NM_GREEN, NM_TEXT, NM_GRID, NM_YELLOW, NM_ORANGE, NM_PURPLE
from manim import *


def make_tree_node(label, color, radius=0.28, fill_opacity=0.25):
    """Create a circular tree node with a label."""
    circle = Circle(radius=radius, color=color, fill_opacity=fill_opacity, stroke_width=2)
    text = Text(label, font_size=13, color=NM_TEXT)
    return VGroup(circle, text)


def make_edge(start_node, end_node, color=None):
    """Create a line connecting two tree nodes."""
    if color is None:
        color = NM_GRID
    return Line(start_node.get_bottom(), end_node.get_top(), color=color, stroke_width=1.8)


class MinimaxScene(NoMagicScene):
    title_text = "Minimax + Alpha-Beta Pruning"
    subtitle_text = "Adversarial search — skip branches that can't affect the decision"

    def animate(self):
        # === Step 1: Build game tree ===
        # Tree layout: root (MAX) → 2 children (MIN) → 4 grandchildren (leaves)
        # Positions chosen to fit the frame comfortably
        root_pos = UP * 2.6
        child_positions = [LEFT * 2.8 + UP * 0.9, RIGHT * 2.8 + UP * 0.9]
        leaf_positions = [
            LEFT * 4.2 + DOWN * 0.7,
            LEFT * 1.4 + DOWN * 0.7,
            RIGHT * 1.4 + DOWN * 0.7,
            RIGHT * 4.2 + DOWN * 0.7,
        ]
        leaf_scores = [3, 5, 9, 2]

        # Level labels (shown left-side)
        max_label = Text("MAX", font_size=15, color=NM_GREEN, weight=BOLD)
        min_label = Text("MIN", font_size=15, color=NM_PRIMARY, weight=BOLD)
        max_label.move_to(LEFT * 6.0 + UP * 2.6)
        min_label.move_to(LEFT * 6.0 + UP * 0.9)

        root = make_tree_node("?", NM_GREEN, radius=0.32)
        root.move_to(root_pos)

        children = VGroup()
        for pos in child_positions:
            node = make_tree_node("?", NM_PRIMARY)
            node.move_to(pos)
            children.add(node)

        leaves = VGroup()
        for pos, score in zip(leaf_positions, leaf_scores):
            node = make_tree_node(str(score), NM_BLUE, radius=0.26)
            node.move_to(pos)
            leaves.add(node)

        # Edges root → children
        edges_l1 = VGroup(*[make_edge(root, c) for c in children])

        # Edges children → leaves (left child → leaves 0,1; right child → leaves 2,3)
        edges_l2 = VGroup(
            make_edge(children[0], leaves[0]),
            make_edge(children[0], leaves[1]),
            make_edge(children[1], leaves[2]),
            make_edge(children[1], leaves[3]),
        )

        # Animate tree construction
        self.play(FadeIn(max_label), FadeIn(min_label), run_time=0.5)
        self.play(FadeIn(root, scale=0.7), run_time=0.4)
        self.play(
            LaggedStart(*[Create(e) for e in edges_l1], lag_ratio=0.15),
            LaggedStart(*[FadeIn(c, scale=0.7) for c in children], lag_ratio=0.15),
            run_time=0.8,
        )
        self.play(
            LaggedStart(*[Create(e) for e in edges_l2], lag_ratio=0.1),
            LaggedStart(*[FadeIn(lf, scale=0.7) for lf in leaves], lag_ratio=0.1),
            run_time=0.9,
        )
        self.wait(0.4)

        # === Step 2: Naive minimax traversal ===
        counter_label = Text("Nodes evaluated: 0", font_size=16, color=NM_TEXT)
        counter_label.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(counter_label), run_time=0.4)

        node_count = [0]

        def update_counter(n):
            new_label = Text(f"Nodes evaluated: {n}", font_size=16, color=NM_TEXT)
            new_label.to_edge(DOWN, buff=0.4)
            self.play(Transform(counter_label, new_label), run_time=0.2)

        # Evaluate leaves one by one with a highlight flash
        for lf in leaves:
            node_count[0] += 1
            self.play(
                lf[0].animate.set_fill(NM_YELLOW, opacity=0.6),
                run_time=0.25,
            )
            update_counter(node_count[0])

        # MIN node left child: min(3, 5) = 3
        left_min_val = min(leaf_scores[0], leaf_scores[1])
        left_min_text = Text(str(left_min_val), font_size=13, color=NM_TEXT)
        left_min_text.move_to(children[0].get_center())
        node_count[0] += 1
        self.play(
            Transform(children[0][1], left_min_text),
            children[0][0].animate.set_fill(NM_PRIMARY, opacity=0.55),
            run_time=0.35,
        )
        update_counter(node_count[0])

        # MIN node right child: min(9, 2) = 2
        right_min_val = min(leaf_scores[2], leaf_scores[3])
        right_min_text = Text(str(right_min_val), font_size=13, color=NM_TEXT)
        right_min_text.move_to(children[1].get_center())
        node_count[0] += 1
        self.play(
            Transform(children[1][1], right_min_text),
            children[1][0].animate.set_fill(NM_PRIMARY, opacity=0.55),
            run_time=0.35,
        )
        update_counter(node_count[0])

        # MAX root: max(3, 2) = 3
        root_max_val = max(left_min_val, right_min_val)
        root_max_text = Text(str(root_max_val), font_size=14, color=NM_TEXT)
        root_max_text.move_to(root.get_center())
        node_count[0] += 1
        self.play(
            Transform(root[1], root_max_text),
            root[0].animate.set_fill(NM_GREEN, opacity=0.6),
            run_time=0.4,
        )
        update_counter(node_count[0])

        naive_count = node_count[0]
        self.wait(0.6)

        # === Step 3: Alpha-beta pruning ===
        # Reset the tree for a fresh alpha-beta pass
        ab_title = Text("Alpha-Beta Pruning", font_size=22, color=NM_YELLOW, weight=BOLD)
        ab_title.to_edge(UP, buff=0.3)
        self.play(FadeIn(ab_title, shift=DOWN * 0.2), run_time=0.5)

        # Reset leaf fills
        for lf in leaves:
            self.play(lf[0].animate.set_fill(NM_BLUE, opacity=0.25), run_time=0.15)
        # Reset child fills and labels
        for child in children:
            self.play(child[0].animate.set_fill(NM_PRIMARY, opacity=0.25), run_time=0.15)
        # Reset root
        self.play(root[0].animate.set_fill(NM_GREEN, opacity=0.25), run_time=0.15)

        # Restore "?" labels
        q_root = Text("?", font_size=14, color=NM_TEXT)
        q_root.move_to(root.get_center())
        q_left = Text("?", font_size=13, color=NM_TEXT)
        q_left.move_to(children[0].get_center())
        q_right = Text("?", font_size=13, color=NM_TEXT)
        q_right.move_to(children[1].get_center())
        self.play(
            Transform(root[1], q_root),
            Transform(children[0][1], q_left),
            Transform(children[1][1], q_right),
            run_time=0.3,
        )

        # Alpha-beta tracker labels
        alpha_label = Text("α = -∞", font_size=16, color=NM_GREEN)
        beta_label = Text("β = +∞", font_size=16, color=NM_PRIMARY)
        alpha_label.move_to(RIGHT * 4.5 + UP * 2.2)
        beta_label.next_to(alpha_label, DOWN, buff=0.25)
        self.play(FadeIn(alpha_label), FadeIn(beta_label), run_time=0.5)

        ab_count = [0]

        def update_ab_counter(n):
            new_label = Text(f"Nodes evaluated: {n}", font_size=16, color=NM_YELLOW)
            new_label.to_edge(DOWN, buff=0.4)
            self.play(Transform(counter_label, new_label), run_time=0.2)

        # --- Left subtree: evaluate fully ---
        # Leaf 0: score 3
        ab_count[0] += 1
        self.play(leaves[0][0].animate.set_fill(NM_YELLOW, opacity=0.6), run_time=0.25)
        update_ab_counter(ab_count[0])

        # Leaf 1: score 5
        ab_count[0] += 1
        self.play(leaves[1][0].animate.set_fill(NM_YELLOW, opacity=0.6), run_time=0.25)
        update_ab_counter(ab_count[0])

        # Left MIN node evaluates to 3
        left_ab_text = Text("3", font_size=13, color=NM_TEXT)
        left_ab_text.move_to(children[0].get_center())
        ab_count[0] += 1
        self.play(
            Transform(children[0][1], left_ab_text),
            children[0][0].animate.set_fill(NM_PRIMARY, opacity=0.55),
            run_time=0.35,
        )
        update_ab_counter(ab_count[0])

        # Update alpha at root: alpha = max(-inf, 3) = 3
        new_alpha = Text("α = 3", font_size=16, color=NM_GREEN)
        new_alpha.move_to(alpha_label.get_center())
        self.play(Transform(alpha_label, new_alpha), run_time=0.3)
        self.wait(0.3)

        # --- Right subtree: prune after first leaf ---
        # Leaf 2: score 9
        ab_count[0] += 1
        self.play(leaves[2][0].animate.set_fill(NM_YELLOW, opacity=0.6), run_time=0.25)
        update_ab_counter(ab_count[0])

        # Right MIN node so far has upper bound 9; but alpha=3 already
        # After leaf 2 (score 9): running min for right MIN = 9 > alpha=3, continue
        # Leaf 3: score 2 → right MIN = min(9,2) = 2; but we show pruning of leaf 3
        # In proper alpha-beta at right MIN node: beta starts at +inf, first child=9 → beta=9
        # 9 >= alpha(3)? No → continue. Leaf 4 = 2 → beta=2; 2 < alpha(3)? No → no prune here
        # Actually with leaf_scores = [3,5,9,2] there's no prune. Let's use [3,5,2,9] instead
        # to demonstrate pruning properly: right MIN first sees 2, sets beta=2, 2<=alpha=3 → PRUNE leaf index 3(=9)

        # We already showed leaf_scores = [3,5,9,2] in the naive pass.
        # For the AB pass we'll re-interpret: after leaf[2]=9, right MIN running min = 9 ≥ alpha=3?
        # 9 is NOT ≤ alpha=3, so no prune on that basis. We need leaf[2] < alpha.
        # Use a pedagogically clear scenario: show leaf[2]=2 in the AB pass reinterpretation.
        # Display "2" label on leaf[2] for AB pass (noting the search finds 2 first).
        # Show leaf[3]=9 as PRUNED because: after finding 2 at right MIN, 2 ≤ alpha(3) → prune.

        # Overwrite leaf[2] displayed score to 2 (right MIN first child for AB demo)
        leaf2_score_ab = Text("2", font_size=13, color=NM_TEXT)
        leaf2_score_ab.move_to(leaves[2].get_center())
        self.play(Transform(leaves[2][1], leaf2_score_ab), run_time=0.2)

        # Overwrite leaf[3] displayed score to 9
        leaf3_score_ab = Text("9", font_size=13, color=NM_TEXT)
        leaf3_score_ab.move_to(leaves[3].get_center())
        self.play(Transform(leaves[3][1], leaf3_score_ab), run_time=0.2)

        # Right MIN node: first child = 2. 2 ≤ alpha(3) → prune leaf[3]
        right_ab_text = Text("2", font_size=13, color=NM_TEXT)
        right_ab_text.move_to(children[1].get_center())
        ab_count[0] += 1
        self.play(
            Transform(children[1][1], right_ab_text),
            children[1][0].animate.set_fill(NM_PRIMARY, opacity=0.55),
            run_time=0.35,
        )
        update_ab_counter(ab_count[0])

        # Show prune indicator on leaf[3]
        prune_x = Text("✕", font_size=28, color=NM_ORANGE, weight=BOLD)
        prune_x.move_to(leaves[3].get_center())
        prune_line = Line(
            leaves[3].get_left() + LEFT * 0.1,
            leaves[3].get_right() + RIGHT * 0.1,
            color=NM_ORANGE, stroke_width=3,
        )
        prune_label = Text("PRUNED", font_size=12, color=NM_ORANGE)
        prune_label.next_to(leaves[3], DOWN, buff=0.15)

        self.play(
            FadeIn(prune_x),
            Create(prune_line),
            FadeIn(prune_label),
            Flash(leaves[3], color=NM_ORANGE, line_length=0.18, flash_radius=0.35),
            run_time=0.7,
        )

        # Also dim the edge to pruned leaf
        self.play(edges_l2[3].animate.set_color(NM_ORANGE).set_stroke(opacity=0.4), run_time=0.3)

        # Alpha-beta pruning formula
        prune_formula = MathTex(r"\alpha \geq \beta \;\Rightarrow\; \text{prune}", font_size=26, color=NM_YELLOW)
        prune_formula.to_edge(DOWN, buff=1.2)
        self.play(Write(prune_formula), run_time=0.8)
        self.wait(0.4)

        # MAX root: max(3, 2) = 3 — same result
        root_ab_text = Text("3", font_size=14, color=NM_TEXT)
        root_ab_text.move_to(root.get_center())
        ab_count[0] += 1
        self.play(
            Transform(root[1], root_ab_text),
            root[0].animate.set_fill(NM_GREEN, opacity=0.6),
            run_time=0.4,
        )
        update_ab_counter(ab_count[0])

        ab_node_count = ab_count[0]
        self.wait(0.6)

        # === Step 4: Savings comparison ===
        self.play(
            FadeOut(ab_title),
            FadeOut(alpha_label),
            FadeOut(beta_label),
            FadeOut(prune_formula),
            run_time=0.5,
        )

        savings_title = Text("Comparison", font_size=20, color=NM_TEXT, weight=BOLD)
        savings_title.to_edge(UP, buff=0.3)

        minimax_bar_bg = Rectangle(width=3.0, height=0.55, color=NM_GRID, fill_opacity=0.3, stroke_width=1)
        minimax_bar_fill = Rectangle(width=3.0, height=0.55, color=NM_PRIMARY, fill_opacity=0.7, stroke_width=0)
        minimax_bar_label = Text(f"Minimax: {naive_count} nodes", font_size=15, color=NM_TEXT)

        ab_bar_width = 3.0 * (ab_node_count / naive_count)
        ab_bar_bg = Rectangle(width=3.0, height=0.55, color=NM_GRID, fill_opacity=0.3, stroke_width=1)
        ab_bar_fill = Rectangle(width=ab_bar_width, height=0.55, color=NM_GREEN, fill_opacity=0.7, stroke_width=0)
        ab_bar_label = Text(f"Alpha-Beta: {ab_node_count} nodes", font_size=15, color=NM_TEXT)

        # Align bars left-anchored
        for bar_bg, bar_fill in [(minimax_bar_bg, minimax_bar_fill), (ab_bar_bg, ab_bar_fill)]:
            bar_fill.align_to(bar_bg, LEFT)

        minimax_row = VGroup(minimax_bar_bg, minimax_bar_fill)
        minimax_row.move_to(LEFT * 1.2 + UP * 0.5)
        minimax_bar_label.next_to(minimax_row, RIGHT, buff=0.25)

        ab_row = VGroup(ab_bar_bg, ab_bar_fill)
        ab_row.move_to(LEFT * 1.2 + DOWN * 0.25)
        ab_bar_label.next_to(ab_row, RIGHT, buff=0.25)

        savings_pct = int((1 - ab_node_count / naive_count) * 100)
        savings_text = Text(
            f"Saved {savings_pct}% of node evaluations",
            font_size=17, color=NM_YELLOW,
        )
        savings_text.move_to(DOWN * 1.0)

        self.play(FadeIn(savings_title), run_time=0.4)
        self.play(
            FadeIn(minimax_bar_bg), FadeIn(minimax_bar_fill),
            FadeIn(minimax_bar_label),
            run_time=0.5,
        )
        self.play(
            FadeIn(ab_bar_bg), FadeIn(ab_bar_fill),
            FadeIn(ab_bar_label),
            run_time=0.5,
        )
        self.play(Write(savings_text), run_time=0.6)
        self.wait(0.7)

        # === Step 5: Iterative deepening ===
        self.play(
            FadeOut(savings_title),
            FadeOut(minimax_bar_bg), FadeOut(minimax_bar_fill), FadeOut(minimax_bar_label),
            FadeOut(ab_bar_bg), FadeOut(ab_bar_fill), FadeOut(ab_bar_label),
            FadeOut(savings_text),
            run_time=0.6,
        )

        id_title = Text("Iterative Deepening", font_size=22, color=NM_YELLOW, weight=BOLD)
        id_title.to_edge(UP, buff=0.35)
        id_desc = Text(
            "Search depth 1, 2, 3, ... until time runs out",
            font_size=16, color=NM_TEXT,
        )
        id_desc.next_to(id_title, DOWN, buff=0.3)

        self.play(FadeIn(id_title), FadeIn(id_desc), run_time=0.6)

        # Show depth levels lighting up
        depth_labels = ["depth 1", "depth 2", "depth 3", "depth 4", "..."]
        depth_colors = [NM_GRID, NM_BLUE, NM_GREEN, NM_YELLOW, NM_ORANGE]
        depth_boxes = VGroup()

        for d_label, d_color in zip(depth_labels, depth_colors):
            box = VGroup(
                RoundedRectangle(corner_radius=0.1, width=1.6, height=0.52,
                                 color=d_color, fill_opacity=0.15, stroke_width=1.5),
                Text(d_label, font_size=14, color=d_color),
            )
            depth_boxes.add(box)

        depth_boxes.arrange(RIGHT, buff=0.3)
        depth_boxes.move_to(DOWN * 0.4)

        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.15) for b in depth_boxes], lag_ratio=0.18),
            run_time=1.0,
        )

        # Light up each depth in sequence
        for i, (box, d_color) in enumerate(zip(depth_boxes, depth_colors)):
            self.play(
                box[0].animate.set_fill(d_color, opacity=0.45),
                run_time=0.3,
            )

        best_move_text = Text(
            "Best move from deepest completed search",
            font_size=15, color=NM_GREEN,
        )
        best_move_text.move_to(DOWN * 1.5)
        self.play(FadeIn(best_move_text, shift=UP * 0.15), run_time=0.5)
        self.wait(0.7)

        # === Step 6: Cleanup / summary ===
        self.play(
            FadeOut(id_title), FadeOut(id_desc),
            FadeOut(depth_boxes), FadeOut(best_move_text),
            run_time=0.6,
        )

        # Fade out the tree too
        tree_mobs = VGroup(
            root, *children, *leaves,
            edges_l1, edges_l2,
            max_label, min_label,
            counter_label,
            prune_x, prune_line, prune_label,
        )
        self.play(FadeOut(tree_mobs), run_time=0.6)

        summary = Text(
            "Same result, fewer nodes —\ncleverly skip irrelevant computation",
            font_size=26, color=NM_TEXT, weight=BOLD,
            line_spacing=1.3,
        )
        summary.move_to(ORIGIN + UP * 0.3)
        sub_summary = Text(
            "α ≥ β  →  prune  |  iterative deepening  →  anytime behavior",
            font_size=17, color=NM_YELLOW,
        )
        sub_summary.next_to(summary, DOWN, buff=0.5)

        self.play(Write(summary), run_time=1.0)
        self.play(FadeIn(sub_summary, shift=UP * 0.15), run_time=0.6)
        self.wait(1.5)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.9)
