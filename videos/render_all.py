"""
Render all no-magic algorithm visualization scenes.

Python alternative to render.sh with the same feature set.

Usage:
    python videos/render_all.py                           # all scenes, MP4 + GIF
    python videos/render_all.py --full-only               # MP4s only (1080p60)
    python videos/render_all.py --preview-only            # GIF previews only (480p15)
    python videos/render_all.py --quality medium           # custom quality preset
    python videos/render_all.py microattention            # single scene
    python videos/render_all.py microattention microgpt   # multiple scenes
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SCENES_DIR = Path(__file__).resolve().parent / "scenes"
RENDERS_DIR = Path(__file__).resolve().parent / "renders"
PREVIEWS_DIR = Path(__file__).resolve().parent / "previews"

SCENE_MAP = {
    "scene_microattention.py": "AttentionScene",
    "scene_microbatchnorm.py": "BatchNormScene",
    "scene_microbeam.py": "BeamSearchScene",
    "scene_microbert.py": "BERTScene",
    "scene_microcheckpoint.py": "CheckpointScene",
    "scene_microconv.py": "ConvScene",
    "scene_microdiffusion.py": "DiffusionScene",
    "scene_microdpo.py": "DPOScene",
    "scene_microdropout.py": "DropoutScene",
    "scene_microembedding.py": "EmbeddingScene",
    "scene_microflash.py": "FlashAttentionScene",
    "scene_microgan.py": "GANScene",
    "scene_microgpt.py": "GPTScene",
    "scene_microgrpo.py": "GRPOScene",
    "scene_microkv.py": "KVCacheScene",
    "scene_microlora.py": "LoRAScene",
    "scene_micromoe.py": "MoEScene",
    "scene_microoptimizer.py": "OptimizerScene",
    "scene_micropaged.py": "PagedScene",
    "scene_microparallel.py": "ParallelScene",
    "scene_microppo.py": "PPOScene",
    "scene_microqlora.py": "QLoRAScene",
    "scene_microquant.py": "QuantizationScene",
    "scene_microrag.py": "RAGScene",
    "scene_microreinforce.py": "ReinforceScene",
    "scene_micrornn.py": "RNNScene",
    "scene_microrope.py": "RoPEScene",
    "scene_microssm.py": "SSMScene",
    "scene_microtokenizer.py": "TokenizerScene",
    "scene_microvae.py": "VAEScene",
}

QUALITY_MAP = {
    "low": ("-ql", "480p15"),
    "medium": ("-qm", "720p30"),
    "high": ("-qh", "1080p60"),
    "4k": ("-qk", "2160p60"),
}


def get_short_name(scene_file: str) -> str:
    """scene_microXXX.py → microXXX"""
    return scene_file.replace("scene_", "").replace(".py", "")


def get_duration(path: Path) -> float:
    """Get video/gif duration via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True
    )
    return float(result.stdout.strip()) if result.returncode == 0 else 0


def render_scene(
    scene_file: str,
    scene_class: str,
    quality_flag: str,
    quality_dir: str,
    fmt: str,
    output_dir: Path,
) -> tuple[str, str]:
    """Render a single scene. Returns (output_name, status)."""
    scene_path = SCENES_DIR / scene_file
    short_name = get_short_name(scene_file)
    ext = fmt
    output_name = f"{short_name}.{ext}"
    output_path = output_dir / output_name

    media_dir = SCENES_DIR / "media"

    cmd = [
        sys.executable, "-m", "manim", "render",
        quality_flag,
        f"--format={fmt}",
        f"--media_dir={media_dir}",
        str(scene_path),
        scene_class,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        stderr_tail = result.stderr[-300:] if result.stderr else "no stderr"
        return output_name, f"FAILED: {stderr_tail}"

    # Find rendered file in Manim's nested output
    rendered = None
    for match in media_dir.rglob(f"{scene_class}.{ext}"):
        if quality_dir in str(match):
            rendered = match
            break
    if not rendered:
        for match in media_dir.rglob(f"{scene_class}.{ext}"):
            rendered = match
            break

    if not rendered:
        return output_name, "NOT FOUND"

    shutil.copy2(rendered, output_path)
    duration = get_duration(output_path)
    size_kb = output_path.stat().st_size / 1024

    return output_name, f"{duration:.1f}s, {size_kb:.0f} KB"


def optimize_gif(gif_path: Path) -> None:
    """Optimize a GIF preview: extract 8s of core animation, reduce palette."""
    if not gif_path.exists():
        return

    has_ffmpeg = shutil.which("ffmpeg") is not None
    has_gifsicle = shutil.which("gifsicle") is not None

    if not has_ffmpeg or not has_gifsicle:
        return

    tmp_gif = Path(f"/tmp/{gif_path.stem}_opt.gif")

    # Extract 8s of core animation (skip 5s branded frames), 10fps, 400px wide
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(gif_path), "-ss", "5", "-t", "8",
         "-vf", "fps=10,scale=400:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=sierra2_4a",
         str(tmp_gif)],
        capture_output=True
    )

    if tmp_gif.exists():
        subprocess.run(
            ["gifsicle", "--optimize=3", "--lossy=80", str(tmp_gif), "-o", str(gif_path)],
            capture_output=True
        )
        tmp_gif.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Render no-magic algorithm visualization scenes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python videos/render_all.py                         Render all (MP4 + GIF)
  python videos/render_all.py --full-only             MP4s only (1080p60)
  python videos/render_all.py --preview-only          GIF previews only (480p15)
  python videos/render_all.py --quality medium         720p30 renders
  python videos/render_all.py microattention microgpt  Specific scenes only
  python videos/render_all.py --skip-optimize          Skip GIF optimization
        """,
    )
    parser.add_argument("scenes", nargs="*",
                        help="Specific scene names to render (e.g., microattention microgpt)")
    parser.add_argument("--quality", choices=QUALITY_MAP.keys(), default="high",
                        help="Render quality (default: high)")
    parser.add_argument("--full-only", action="store_true",
                        help="Render full MP4s only, skip GIF previews")
    parser.add_argument("--preview-only", action="store_true",
                        help="Render GIF previews only, skip full MP4s")
    parser.add_argument("--skip-optimize", action="store_true",
                        help="Skip GIF optimization step (ffmpeg + gifsicle)")

    args = parser.parse_args()

    render_full = not args.preview_only
    render_preview = not args.full_only

    quality_flag, quality_dir = QUALITY_MAP[args.quality]

    # Filter scenes if specific names given
    if args.scenes:
        filtered = {}
        for scene_file, scene_class in SCENE_MAP.items():
            short_name = get_short_name(scene_file)
            if short_name in args.scenes:
                filtered[scene_file] = scene_class
        if not filtered:
            print(f"No matching scenes found for: {args.scenes}")
            print(f"Available: {', '.join(get_short_name(f) for f in sorted(SCENE_MAP))}")
            sys.exit(1)
        scenes = filtered
    else:
        scenes = SCENE_MAP

    RENDERS_DIR.mkdir(exist_ok=True)
    PREVIEWS_DIR.mkdir(exist_ok=True)

    total = len(scenes)
    full_results = []
    preview_results = []

    for i, (scene_file, scene_class) in enumerate(sorted(scenes.items()), 1):
        short_name = get_short_name(scene_file)
        print(f"\n[{i}/{total}] {short_name}")

        if render_full:
            print(f"  MP4 ({quality_dir})...", end=" ", flush=True)
            name, status = render_scene(
                scene_file, scene_class, quality_flag, quality_dir, "mp4", RENDERS_DIR
            )
            print(status)
            full_results.append((name, status))

        if render_preview:
            print(f"  GIF (480p15)...", end=" ", flush=True)
            name, status = render_scene(
                scene_file, scene_class, "-ql", "480p15", "gif", PREVIEWS_DIR
            )
            print(status)
            preview_results.append((name, status))

            if not args.skip_optimize:
                gif_path = PREVIEWS_DIR / f"{short_name}.gif"
                optimize_gif(gif_path)

    # Clean up Manim's nested media directory
    media_dir = SCENES_DIR / "media"
    if media_dir.exists():
        shutil.rmtree(media_dir, ignore_errors=True)

    # Summary
    print("\n" + "=" * 50)
    print("RENDER SUMMARY")
    print("=" * 50)

    if full_results:
        print(f"\nFull renders ({quality_dir}) → {RENDERS_DIR}/")
        for name, status in full_results:
            print(f"  {name:<30} {status}")

    if preview_results:
        print(f"\nGIF previews → {PREVIEWS_DIR}/")
        for name, status in preview_results:
            print(f"  {name:<30} {status}")


if __name__ == "__main__":
    main()
