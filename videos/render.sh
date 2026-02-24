#!/bin/bash
# Batch render all no-magic algorithm visualization scenes.
# Usage: bash videos/render.sh [--preview-only] [--full-only] [--skip-optimize] [scene_name]
#
# Examples:
#   bash videos/render.sh                    # render all (MP4 + GIF)
#   bash videos/render.sh microattention     # render single scene
#   bash videos/render.sh --preview-only     # GIF previews only
#   bash videos/render.sh --full-only        # MP4s only
#   bash videos/render.sh --skip-optimize    # skip GIF optimization step
set -e

SCENES_DIR="videos/scenes"
PREVIEW_DIR="videos/previews"
RELEASE_DIR="videos/renders"

RENDER_PREVIEW=true
RENDER_FULL=true
OPTIMIZE_GIFS=true
FILTER=""

for arg in "$@"; do
    case "$arg" in
        --preview-only) RENDER_FULL=false ;;
        --full-only) RENDER_PREVIEW=false ;;
        --skip-optimize) OPTIMIZE_GIFS=false ;;
        *) FILTER="$arg" ;;
    esac
done

mkdir -p "$PREVIEW_DIR" "$RELEASE_DIR"

rendered_scenes=()

for scene_file in "$SCENES_DIR"/scene_*.py; do
    name=$(basename "$scene_file" .py | sed 's/scene_//')

    # Apply filter if specified
    if [ -n "$FILTER" ] && [ "$name" != "$FILTER" ]; then
        continue
    fi

    # Extract scene class name (first class ending in "Scene")
    class_name=$(grep -m1 -oE 'class \w+Scene' "$scene_file" | sed 's/class //')

    if [ -z "$class_name" ]; then
        echo "SKIP: $scene_file (no *Scene class found)"
        continue
    fi

    echo "=== Rendering: $name ($class_name) ==="

    if [ "$RENDER_FULL" = true ]; then
        echo "  MP4 (1080p60)..."
        manim -qh "$scene_file" "$class_name" \
            --media_dir "$RELEASE_DIR" 2>&1 | tail -1
    fi

    if [ "$RENDER_PREVIEW" = true ]; then
        echo "  GIF (480p15)..."
        manim -ql --format gif "$scene_file" "$class_name" \
            --media_dir "$PREVIEW_DIR" 2>&1 | tail -1
    fi

    rendered_scenes+=("$name")
    echo "  Done: $name"
done

# === Flatten Manim's nested output to clean filenames ===
echo ""
echo "Flattening output files..."

if [ "$RENDER_FULL" = true ]; then
    for mp4 in "$RELEASE_DIR"/videos/scene_micro*/1080p60/*.mp4; do
        [ -f "$mp4" ] || continue
        scene_dir=$(basename "$(dirname "$(dirname "$mp4")")")
        flat_name="${scene_dir#scene_}.mp4"
        cp "$mp4" "$RELEASE_DIR/$flat_name"
        echo "  MP4: $flat_name"
    done
    rm -rf "$RELEASE_DIR/videos" "$RELEASE_DIR/images" "$RELEASE_DIR/Tex" "$RELEASE_DIR/texts" 2>/dev/null
fi

if [ "$RENDER_PREVIEW" = true ]; then
    for gif in "$PREVIEW_DIR"/videos/scene_micro*/480p15/*.gif; do
        [ -f "$gif" ] || continue
        scene_dir=$(basename "$(dirname "$(dirname "$gif")")")
        flat_name="${scene_dir#scene_}.gif"
        cp "$gif" "$PREVIEW_DIR/$flat_name"
        echo "  GIF: $flat_name (raw)"
    done
    rm -rf "$PREVIEW_DIR/videos" "$PREVIEW_DIR/images" "$PREVIEW_DIR/Tex" "$PREVIEW_DIR/texts" 2>/dev/null
fi

# === Optimize GIFs (ffmpeg palette + gifsicle) ===
if [ "$RENDER_PREVIEW" = true ] && [ "$OPTIMIZE_GIFS" = true ]; then
    echo ""
    echo "Optimizing GIF previews..."

    # Check for required tools
    if ! command -v ffmpeg &>/dev/null; then
        echo "  WARNING: ffmpeg not found — skipping GIF optimization"
    elif ! command -v gifsicle &>/dev/null; then
        echo "  WARNING: gifsicle not found — skipping GIF optimization"
    else
        for name in "${rendered_scenes[@]}"; do
            raw_gif="$PREVIEW_DIR/${name}.gif"
            [ -f "$raw_gif" ] || continue
            tmp_gif="/tmp/${name}_opt.gif"
            # Extract 8s of core animation (skip 5s branded frames), 10fps, 400px wide, 128-color palette
            ffmpeg -y -i "$raw_gif" -ss 5 -t 8 \
                -vf "fps=10,scale=400:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=sierra2_4a" \
                "$tmp_gif" 2>/dev/null
            gifsicle --optimize=3 --lossy=80 "$tmp_gif" -o "$raw_gif"
            rm -f "$tmp_gif"
            size=$(ls -lh "$raw_gif" | awk '{print $5}')
            echo "  Optimized: ${name}.gif ($size)"
        done
    fi
fi

echo ""
echo "Rendering complete."
[ "$RENDER_FULL" = true ] && echo "  Full renders:  $RELEASE_DIR/"
[ "$RENDER_PREVIEW" = true ] && echo "  GIF previews:  $PREVIEW_DIR/"
