"""
Shared base class and color palette for all no-magic algorithm visualization scenes.

Every scene inherits NoMagicScene and overrides animate() with algorithm-specific
animation logic. Branded start/end frames and algorithm title card are handled
by the base class.
"""
from pathlib import Path

from manim import *


# no-magic color palette — matches GitHub dark mode aesthetics
NM_BG = "#1a1a2e"
NM_PRIMARY = "#e94560"
NM_BLUE = "#0f3460"
NM_GREEN = "#16c79a"
NM_TEXT = "#eaeaea"
NM_GRID = "#2a2a4a"
NM_YELLOW = "#f5c542"
NM_ORANGE = "#e97d32"
NM_PURPLE = "#9b59b6"

# Resolve asset paths relative to the repo root (two levels up from scenes/)
_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
START_FRAME = str(_ASSETS_DIR / "video_start_frame.png")
END_FRAME = str(_ASSETS_DIR / "video_end_frame.png")


class NoMagicScene(Scene):
    """Base scene with branded start/end frames and algorithm title card.

    Subclasses set title_text and subtitle_text as class attributes,
    then implement animate() for the algorithm-specific visualization.

    Frame sequence: branded start frame → algorithm title → animation → branded end frame
    """

    title_text: str = "Algorithm"
    subtitle_text: str = "one-liner description"

    def construct(self):
        self.camera.background_color = NM_BG
        self.show_start_frame()
        self.show_title()
        self.animate()
        self.show_end_frame()

    def show_start_frame(self):
        """Display the branded start frame PNG for 2.5 seconds, then fade out."""
        frame = ImageMobject(START_FRAME)
        frame.height = config.frame_height
        self.play(FadeIn(frame), run_time=0.8)
        self.wait(2.0)
        self.play(FadeOut(frame), run_time=0.6)

    def show_title(self):
        """Show the algorithm-specific title card over the dark background."""
        title = Text(self.title_text, font_size=48, color=NM_TEXT, weight=BOLD)
        subtitle = Text(self.subtitle_text, font_size=24, color=NM_PRIMARY)
        subtitle.next_to(title, DOWN, buff=0.3)
        self.play(Write(title), FadeIn(subtitle, shift=UP * 0.2), run_time=1.2)
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(subtitle), run_time=0.6)

    def show_end_frame(self):
        """Display the branded end frame PNG for 3 seconds."""
        frame = ImageMobject(END_FRAME)
        frame.height = config.frame_height
        self.play(FadeIn(frame), run_time=0.8)
        self.wait(2.5)

    def animate(self):
        raise NotImplementedError("Subclasses must implement animate()")
