"""
Sprite-sheet grid -> animated GIF builder for symm.city bot.

Generates a single sequential animation grid image (2x2, 3x3, or 4x4) with Nano Banana 2.1
and processes it in pure Python / Pillow into a stabilized, gutter-trimmed, shared-palette
looping GIF suitable for Telegram's `send_animation`.
"""

import io
import math
from typing import Any

import numpy as np
from PIL import Image

SUPPORTED_GRID_SIZES: dict[str, tuple[int, int]] = {
    "2x2": (2, 2),
    "3x3": (3, 3),
    "4x4": (4, 4),
}

GRID_SAFE_ASPECT_RATIOS: dict[str, float] = {
    "1:1": 1.0,
    "4:3": 4.0 / 3.0,
    "3:4": 3.0 / 4.0,
    "3:2": 3.0 / 2.0,
    "2:3": 2.0 / 3.0,
    "16:9": 16.0 / 9.0,
    "9:16": 9.0 / 16.0,
}


def infer_aspect_ratio_from_image_bytes(raw_bytes: bytes) -> str:
    """Snap a reference image's dimensions to the closest Nano Banana grid-safe aspect ratio."""
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            w, h = img.size
        if w <= 0 or h <= 0:
            return "1:1"
        ratio = w / h
        best_key = min(
            GRID_SAFE_ASPECT_RATIOS.keys(),
            key=lambda k: abs(math.log(ratio / GRID_SAFE_ASPECT_RATIOS[k])),
        )
        return best_key
    except Exception:
        return "1:1"


def build_sprite_sheet_prompt(
    scene_prompt: str,
    *,
    grid_size: str = "3x3",
    loop_mode: str = "loop",
    has_reference_images: bool = False,
) -> str:
    """
    Wrap a creative animation brief in strict geometric & temporal sprite-sheet instructions
    for `gemini-nano-banana-2.1`.
    """
    cols, rows = SUPPORTED_GRID_SIZES.get(grid_size, (3, 3))
    total = cols * rows

    row_guide_parts = []
    for r in range(rows):
        start_f = r * cols + 1
        end_f = (r + 1) * cols
        row_guide_parts.append(f"Row {r + 1} left-to-right = Frames {start_f}..{end_f}")
    reading_order_str = "; ".join(row_guide_parts)

    if loop_mode == "pingpong":
        continuity_rule = (
            f"- SMOOTH INCREMENTAL MOTION: Advance the action evenly in {total} small consecutive steps "
            f"from initial pose in Frame 1 to peak pose in Frame {total} (playback will bounce smoothly "
            f"forward and backward as a seamless loop)."
        )
    else:
        continuity_rule = (
            f"- SEAMLESS CYCLICAL LOOP: Choreograph the {total} consecutive steps as a complete continuous cycle "
            f"so Frame {total} transitions smoothly right back into Frame 1 with no jump cut."
        )

    ref_rule = ""
    if has_reference_images:
        ref_rule = (
            "- REFERENCE IMAGE CONSISTENCY: Strictly preserve the exact subject appearance, character/object design, "
            "colors, lighting, art style, and background setting from the provided reference image(s), bringing the scene "
            f"to life across all {total} frames.\n"
        )

    return (
        f"Create a single sequential animation sprite sheet arranged in an exact {rows}x{cols} grid "
        f"({rows} rows by {cols} columns = {total} equal-sized rectangular panels in strict reading order: "
        f"{reading_order_str}).\n\n"
        f"MANDATORY SPRITE-SHEET LAYOUT & CAMERA RULES:\n"
        f"- FULL-BLEED EDGE-TO-EDGE GRID: Every panel must touch adjacent panels and the image edges directly with "
        f"ZERO outer margins, ZERO border frames, ZERO white or black divider lines/gutters, ZERO frame numbers, "
        f"and ZERO text labels or UI overlays.\n"
        f"- LOCKED CAMERA & EXACT REGISTRATION: Keep the camera position, focal length, horizon, background elements, "
        f"lighting, and subject scale identically anchored at the exact same relative coordinates inside every single cell. "
        f"Only the animated motion should change smoothly from frame to frame.\n"
        f"{continuity_rule}\n"
        f"{ref_rule}\n"
        f"ANIMATION SCENE & MOTION DESCRIPTION:\n"
        f"{scene_prompt.strip()}"
    )


def _trim_symmetric_outer_margin(arr: np.ndarray) -> np.ndarray:
    """
    Remove an accidental solid uniform frame/letterbox around the entire grid sheet,
    trimming ONLY symmetrically on opposite edges so row/column grid divisions never skew.
    """
    h, w, _ = arr.shape
    max_y = min(int(h * 0.04), 48)
    max_x = min(int(w * 0.04), 48)

    pad_y = 0
    if max_y >= 2 and h > 128:
        top_ref = arr[0, :, :].astype(np.float32).mean(axis=0)
        bot_ref = arr[h - 1, :, :].astype(np.float32).mean(axis=0)
        if np.max(np.abs(top_ref - bot_ref)) < 12.0:
            for k in range(max_y):
                t_row = arr[k, :, :].astype(np.float32)
                b_row = arr[h - 1 - k, :, :].astype(np.float32)
                if (
                    t_row.std(axis=0).max() < 5.5
                    and b_row.std(axis=0).max() < 5.5
                    and np.max(np.abs(t_row.mean(axis=0) - top_ref)) < 10.0
                    and np.max(np.abs(b_row.mean(axis=0) - bot_ref)) < 10.0
                ):
                    pad_y = k + 1
                else:
                    break

    pad_x = 0
    if max_x >= 2 and w > 128:
        left_ref = arr[:, 0, :].astype(np.float32).mean(axis=0)
        right_ref = arr[:, w - 1, :].astype(np.float32).mean(axis=0)
        if np.max(np.abs(left_ref - right_ref)) < 12.0:
            for k in range(max_x):
                l_col = arr[:, k, :].astype(np.float32)
                r_col = arr[:, w - 1 - k, :].astype(np.float32)
                if (
                    l_col.std(axis=0).max() < 5.5
                    and r_col.std(axis=0).max() < 5.5
                    and np.max(np.abs(l_col.mean(axis=0) - left_ref)) < 10.0
                    and np.max(np.abs(r_col.mean(axis=0) - right_ref)) < 10.0
                ):
                    pad_x = k + 1
                else:
                    break

    if pad_y > 0 or pad_x > 0:
        arr = arr[pad_y : h - pad_y, pad_x : w - pad_x, :]
    return arr


def _detect_seam_gutter_inset(
    cells: list[np.ndarray], rows: int, cols: int
) -> tuple[int, int]:
    """
    Detect if the image generator drew thin solid divider lines along interior cell seams,
    returning `(inset_y, inset_x)` to crop uniformly from all 4 sides of every cell.
    """
    cell_h, cell_w, _ = cells[0].shape
    min_inset_y = max(2, cell_h // 160)
    min_inset_x = max(2, cell_w // 160)
    max_g_y = min(10, max(min_inset_y, int(cell_h * 0.035)))
    max_g_x = min(10, max(min_inset_x, int(cell_w * 0.035)))

    detected_y = 0
    if rows > 1 and cell_h > 64:
        ref_depth = min(cell_h - 1, max_g_y + 3)
        for k in range(max_g_y):
            stds = []
            jumps = []
            for r in range(rows):
                for c in range(cols):
                    cell = cells[r * cols + c].astype(np.float32)
                    if r > 0:
                        line = cell[k, :, :]
                        inner = cell[ref_depth, :, :]
                        stds.append(float(line.std(axis=0).mean()))
                        jumps.append(float(np.abs(line.mean(axis=0) - inner.mean(axis=0)).max()))
                    if r < rows - 1:
                        line = cell[cell_h - 1 - k, :, :]
                        inner = cell[cell_h - 1 - ref_depth, :, :]
                        stds.append(float(line.std(axis=0).mean()))
                        jumps.append(float(np.abs(line.mean(axis=0) - inner.mean(axis=0)).max()))
            if stds and float(np.median(stds)) < 8.5 and float(np.median(jumps)) > 16.0:
                detected_y = k + 1
            else:
                break

    detected_x = 0
    if cols > 1 and cell_w > 64:
        ref_depth = min(cell_w - 1, max_g_x + 3)
        for k in range(max_g_x):
            stds = []
            jumps = []
            for r in range(rows):
                for c in range(cols):
                    cell = cells[r * cols + c].astype(np.float32)
                    if c > 0:
                        line = cell[:, k, :]
                        inner = cell[:, ref_depth, :]
                        stds.append(float(line.std(axis=0).mean()))
                        jumps.append(float(np.abs(line.mean(axis=0) - inner.mean(axis=0)).max()))
                    if c < cols - 1:
                        line = cell[:, cell_w - 1 - k, :]
                        inner = cell[:, cell_w - 1 - ref_depth, :]
                        stds.append(float(line.std(axis=0).mean()))
                        jumps.append(float(np.abs(line.mean(axis=0) - inner.mean(axis=0)).max()))
            if stds and float(np.median(stds)) < 8.5 and float(np.median(jumps)) > 16.0:
                detected_x = k + 1
            else:
                break

    return max(min_inset_y, detected_y), max(min_inset_x, detected_x)


def _stabilize_frame_jitter(frames: list[np.ndarray]) -> list[np.ndarray]:
    """
    Correct small accidental inter-cell translational jitter (±4px max) relative to Frame 0's
    outer background border ring, cropping symmetrically so zero border fill is introduced.
    """
    if len(frames) < 2:
        return frames

    h, w, _ = frames[0].shape
    max_shift = min(4, h // 64, w // 64)
    if max_shift < 1 or h < 96 or w < 96:
        return frames

    # Luminance arrays for fast L1 registration on the outer 15% border ring
    lums = [
        (0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]).astype(np.float32)
        for f in frames
    ]

    ih0, ih1 = max_shift, h - max_shift
    iw0, iw1 = max_shift, w - max_shift
    inner_h = ih1 - ih0
    inner_w = iw1 - iw0
    ring_y = max(4, int(inner_h * 0.15))
    ring_x = max(4, int(inner_w * 0.15))

    ring_mask = np.zeros((inner_h, inner_w), dtype=bool)
    ring_mask[:ring_y, :] = True
    ring_mask[-ring_y:, :] = True
    ring_mask[:, :ring_x] = True
    ring_mask[:, -ring_x:] = True

    ref_crop = lums[0][ih0:ih1, iw0:iw1]
    ref_ring = ref_crop[ring_mask]

    shifts: list[tuple[int, int]] = [(0, 0)]
    for idx in range(1, len(frames)):
        cur = lums[idx]
        zero_err = float(np.mean(np.abs(cur[ih0:ih1, iw0:iw1][ring_mask] - ref_ring)))
        if zero_err < 1.5:
            shifts.append((0, 0))
            continue

        best_err = zero_err
        best_dy, best_dx = 0, 0
        for dy in range(-max_shift, max_shift + 1):
            for dx in range(-max_shift, max_shift + 1):
                if dy == 0 and dx == 0:
                    continue
                # Shifting frame by (dy, dx) samples `cur[ih0 - dy : ih1 - dy, iw0 - dx : iw1 - dx]`
                cand = cur[ih0 - dy : ih1 - dy, iw0 - dx : iw1 - dx][ring_mask]
                err = float(np.mean(np.abs(cand - ref_ring)))
                if err < best_err:
                    best_err = err
                    best_dy, best_dx = dy, dx

        # Only apply shift if it meaningfully improves background border alignment (>= 12% drop)
        if best_err <= zero_err * 0.88:
            shifts.append((best_dy, best_dx))
        else:
            shifts.append((0, 0))

    crop_y = max(abs(dy) for dy, _ in shifts)
    crop_x = max(abs(dx) for _, dx in shifts)
    if crop_y == 0 and crop_x == 0:
        return frames

    aligned: list[np.ndarray] = []
    out_h = h - 2 * crop_y
    out_w = w - 2 * crop_x
    for f, (dy, dx) in zip(frames, shifts):
        y0 = crop_y - dy
        x0 = crop_x - dx
        aligned.append(f[y0 : y0 + out_h, x0 : x0 + out_w, :])
    return aligned


def slice_grid_to_gif_sync(
    sheet_bytes: bytes,
    *,
    grid_size: str = "3x3",
    fps: int = 8,
    loop_mode: str = "loop",
    stabilize: bool = True,
    target_long_side: int = 448,
) -> dict[str, Any]:
    """
    Slice a single sprite-sheet grid image into clean aligned frames and encode
    a zero-flicker 256-color looping animated GIF using Pillow.
    """
    cols, rows = SUPPORTED_GRID_SIZES.get(grid_size, (3, 3))
    clamped_fps = max(3, min(16, int(fps or 8)))
    # Round frame delay to exact 10ms (1 centisecond) increments >= 60ms for Telegram/ffmpeg compatibility
    frame_ms = max(60, min(350, int(round(1000.0 / clamped_fps / 10.0)) * 10))

    with Image.open(io.BytesIO(sheet_bytes)) as raw_img:
        sheet_rgb = raw_img.convert("RGB")

    sheet_arr = _trim_symmetric_outer_margin(np.asarray(sheet_rgb))
    sh_h, sh_w, _ = sheet_arr.shape

    cell_w = sh_w // cols
    cell_h = sh_h // rows
    if cell_w < 16 or cell_h < 16:
        raise ValueError(f"Generated sprite sheet is too small ({sh_w}x{sh_h}) for {grid_size} grid.")

    raw_cells: list[np.ndarray] = []
    for r in range(rows):
        y0 = int(round(r * (sh_h - cell_h) / max(1, rows - 1))) if rows > 1 else 0
        for c in range(cols):
            x0 = int(round(c * (sh_w - cell_w) / max(1, cols - 1))) if cols > 1 else 0
            raw_cells.append(sheet_arr[y0 : y0 + cell_h, x0 : x0 + cell_w, :])

    inset_y, inset_x = _detect_seam_gutter_inset(raw_cells, rows, cols)
    if cell_h - 2 * inset_y >= 32 and cell_w - 2 * inset_x >= 32:
        cropped_cells = [
            c[inset_y : cell_h - inset_y, inset_x : cell_w - inset_x, :]
            for c in raw_cells
        ]
    else:
        cropped_cells = raw_cells

    if stabilize:
        clean_cells = _stabilize_frame_jitter(cropped_cells)
    else:
        clean_cells = cropped_cells

    cur_h, cur_w, _ = clean_cells[0].shape
    scale = target_long_side / float(max(cur_w, cur_h))
    # Ensure even width & height so Telegram's yuv420p MP4 animation transcoder never pads/crops 1px
    out_w = max(64, int(round(cur_w * scale / 2.0)) * 2)
    out_h = max(64, int(round(cur_h * scale / 2.0)) * 2)

    pil_rgb_frames: list[Image.Image] = []
    for arr in clean_cells:
        im = Image.fromarray(arr, mode="RGB")
        if im.size != (out_w, out_h):
            im = im.resize((out_w, out_h), resample=Image.Resampling.LANCZOS)
        pil_rgb_frames.append(im)

    # Build a single global 256-color master palette across all unique frames so static
    # backgrounds never shimmer or shift palette indices between frames.
    thumb_w = min(160, out_w)
    thumb_h = max(16, int(round(out_h * (thumb_w / float(out_w)))))
    montage = Image.new("RGB", (thumb_w * len(pil_rgb_frames), thumb_h))
    for idx, im in enumerate(pil_rgb_frames):
        montage.paste(
            im.resize((thumb_w, thumb_h), resample=Image.Resampling.BILINEAR),
            (idx * thumb_w, 0),
        )
    master_palette = montage.quantize(colors=256, method=Image.Quantize.MAXCOVERAGE)

    quantized_unique: list[Image.Image] = [
        im.quantize(palette=master_palette, dither=Image.Dither.NONE)
        for im in pil_rgb_frames
    ]

    if loop_mode == "pingpong" and len(quantized_unique) >= 3:
        cycle_frames = quantized_unique + quantized_unique[-2:0:-1]
    else:
        cycle_frames = list(quantized_unique)

    # Repeat short cycles to >= ~1.8s (up to 36 total frames) so Telegram's inline loop plays seamlessly
    cycle_ms = len(cycle_frames) * frame_ms
    repeats = 1
    if 0 < cycle_ms < 1800:
        max_reps = max(1, 36 // len(cycle_frames))
        repeats = min(max_reps, max(1, int(math.ceil(1800.0 / cycle_ms))))

    playback_frames = cycle_frames * repeats

    gif_buf = io.BytesIO()
    playback_frames[0].save(
        gif_buf,
        format="GIF",
        save_all=True,
        append_images=playback_frames[1:],
        duration=frame_ms,
        loop=0,
        optimize=True,
    )
    gif_bytes = gif_buf.getvalue()

    return {
        "gif_bytes": gif_bytes,
        "width": out_w,
        "height": out_h,
        "unique_frames": len(quantized_unique),
        "total_frames": len(playback_frames),
        "frame_duration_ms": frame_ms,
        "total_duration_ms": len(playback_frames) * frame_ms,
    }
