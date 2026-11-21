"""
FFmpeg & Pillow video/GIF processing utilities for symm.city bot.

Uses system `ffmpeg` if available, falling back automatically to the self-contained
`imageio-ffmpeg` static binary so everything works identically on Windows and
inside Railway's Linux `railpack-runtime` container.
"""

import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)


def get_ffmpeg_exe() -> str:
    """Resolve `ffmpeg` executable path from PATH or `imageio-ffmpeg` wheel."""
    sys_bin = shutil.which("ffmpeg")
    if sys_bin:
        return sys_bin
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def infer_omni_aspect_ratio(raw_image_bytes: bytes) -> str:
    """
    Map an input image's dimensions to Gemini Omni 1.2 Flash's supported aspect ratios
    (`"16:9"` landscape or `"9:16"` portrait).
    """
    try:
        with Image.open(io.BytesIO(raw_image_bytes)) as img:
            w, h = img.size
        if h > w * 1.05:
            return "9:16"
        return "16:9"
    except Exception:
        return "16:9"


def _parse_ffmpeg_duration_seconds(stderr_text: str) -> int | None:
    """Extract rounded duration in seconds from `ffmpeg -i` stderr output."""
    if not stderr_text:
        return None
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr_text)
    if not m:
        return None
    hours = int(m.group(1))
    mins = int(m.group(2))
    secs = float(m.group(3))
    total = hours * 3600.0 + mins * 60.0 + secs
    return max(1, int(round(total)))


def _parse_ffmpeg_dimensions(stderr_text: str) -> tuple[int | None, int | None]:
    """Extract `(width, height)` from `ffmpeg -i` Video stream line."""
    if not stderr_text:
        return None, None
    for line in stderr_text.splitlines():
        if "Video:" in line:
            m = re.search(r"\b(\d{2,5})x(\d{2,5})\b", line)
            if m:
                return int(m.group(1)), int(m.group(2))
    return None, None


def mp4_to_gif_sync(
    mp4_bytes: bytes,
    *,
    fps: int = 16,
    max_long_side: int = 512,
    max_telegram_inline_bytes: int = 9_500_000,
) -> dict[str, Any]:
    """
    Convert an MP4 video buffer into a high-fluidity, studio-grade looping `.gif`
    using FFmpeg's two-stream Lanczos + 256-color `palettegen` / `paletteuse` pipeline.
    Automatically steps down slightly if needed to stay under Telegram's ~10MB inline
    autoplay threshold.
    """
    ffmpeg = get_ffmpeg_exe()
    clamped_fps = max(10, min(24, int(fps or 16)))

    with tempfile.TemporaryDirectory(prefix="symm_gif_") as tmpdir:
        in_path = os.path.join(tmpdir, "source.mp4")
        out_path = os.path.join(tmpdir, "output.gif")
        with open(in_path, "wb") as f:
            f.write(mp4_bytes)

        attempts = [
            (clamped_fps, max_long_side),
            (13, 416),
            (10, 352),
        ]

        gif_bytes = b""
        for idx, (cur_fps, cur_long) in enumerate(attempts):
            # Scale so the longer dimension is `cur_long` (preserving aspect ratio, divisible by 2)
            scale_expr = (
                f"scale='if(gte(iw,ih),min({cur_long},iw),-2)':'if(gte(iw,ih),-2,min({cur_long},ih))':flags=lanczos"
            )
            vf = (
                f"fps={cur_fps},{scale_expr},"
                f"split[s0][s1];"
                f"[s0]palettegen=max_colors=256:stats_mode=full[p];"
                f"[s1][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle"
            )
            cmd = [
                ffmpeg,
                "-y",
                "-i",
                in_path,
                "-an",
                "-vf",
                vf,
                "-loop",
                "0",
                out_path,
            ]
            proc = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=90
            )
            if proc.returncode != 0 or not os.path.exists(out_path):
                err_tail = (proc.stderr or "")[-400:].strip()
                raise RuntimeError(f"FFmpeg MP4->GIF conversion failed: {err_tail}")

            with open(out_path, "rb") as gf:
                gif_bytes = gf.read()

            if len(gif_bytes) <= max_telegram_inline_bytes or idx == len(attempts) - 1:
                break

        with Image.open(io.BytesIO(gif_bytes)) as gif_img:
            width, height = gif_img.size
            n_frames = int(getattr(gif_img, "n_frames", 1))

        return {
            "gif_bytes": gif_bytes,
            "width": width,
            "height": height,
            "frame_count": n_frames,
        }


def finalize_telegram_mp4_sync(
    mp4_bytes: bytes,
    *,
    strip_audio: bool = False,
) -> dict[str, Any]:
    """
    Prepare a generated MP4 video for instant inline streaming playback in Telegram:
      - Moves MOOV atom to front (`-movflags +faststart`)
      - Strips audio stream (`-an`) only if `strip_audio=True`
      - Extracts exact `(width, height, duration)` + a <=320px JPEG thumbnail for `send_video`
    """
    ffmpeg = get_ffmpeg_exe()

    with tempfile.TemporaryDirectory(prefix="symm_mp4_") as tmpdir:
        in_path = os.path.join(tmpdir, "raw.mp4")
        out_path = os.path.join(tmpdir, "faststart.mp4")
        thumb_path = os.path.join(tmpdir, "thumb.jpg")

        with open(in_path, "wb") as f:
            f.write(mp4_bytes)

        cmd = [ffmpeg, "-y", "-i", in_path, "-c:v", "copy"]
        if strip_audio:
            cmd.append("-an")
        else:
            cmd.extend(["-c:a", "aac", "-b:a", "160k"])
        cmd.extend(["-movflags", "+faststart", out_path])

        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60
        )
        if proc.returncode != 0 or not os.path.exists(out_path):
            # Fallback re-encode if stream copy encountered unusual container layout
            fb_cmd = [
                ffmpeg,
                "-y",
                "-i",
                in_path,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "veryfast",
            ]
            if strip_audio:
                fb_cmd.append("-an")
            else:
                fb_cmd.extend(["-c:a", "aac", "-b:a", "128k"])
            fb_cmd.extend(["-movflags", "+faststart", out_path])
            fb_proc = subprocess.run(
                fb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120
            )
            if fb_proc.returncode != 0 or not os.path.exists(out_path):
                err_tail = (fb_proc.stderr or proc.stderr or "")[-400:].strip()
                raise RuntimeError(f"FFmpeg MP4 faststart failed: {err_tail}")
            probe_stderr = fb_proc.stderr
        else:
            probe_stderr = proc.stderr

        duration_sec = _parse_ffmpeg_duration_seconds(probe_stderr)
        width, height = _parse_ffmpeg_dimensions(probe_stderr)

        # Extract a clean frame thumbnail (Telegram requires <=320px width & height, <200KB JPEG)
        thumb_bytes: bytes | None = None
        thumb_cmd = [
            ffmpeg,
            "-y",
            "-ss",
            "0.1",
            "-i",
            out_path,
            "-frames:v",
            "1",
            thumb_path,
        ]
        t_proc = subprocess.run(
            thumb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30
        )
        if t_proc.returncode == 0 and os.path.exists(thumb_path):
            try:
                with Image.open(thumb_path) as tim:
                    if not width or not height:
                        width, height = tim.size
                    tim_rgb = tim.convert("RGB")
                    tim_rgb.thumbnail((320, 320), Image.Resampling.LANCZOS)
                    tbuf = io.BytesIO()
                    tim_rgb.save(tbuf, format="JPEG", quality=87, optimize=True)
                    thumb_bytes = tbuf.getvalue()
            except Exception as te:
                logger.debug("Thumbnail extraction skipped: %s", te)

        with open(out_path, "rb") as vf:
            final_mp4 = vf.read()

        return {
            "video_bytes": final_mp4,
            "width": width,
            "height": height,
            "duration": duration_sec,
            "thumbnail_bytes": thumb_bytes,
        }


def prep_input_video_for_omni_sync(
    video_bytes: bytes,
    *,
    max_duration_sec: int = 15,
    strip_audio: bool = False,
) -> bytes:
    """
    Normalize an input video for Gemini Omni 1.2 Flash editing or extension:
      - Trims to `<= max_duration_sec` (Omni 1.2 Flash accepts up to 15s input videos)
      - Scales down proportionally ONLY if exceeding 720p (1280x720 landscape / 720x1280 portrait)
      - Strips audio (`-an`) when `strip_audio=True` so Omni Flash generates brand-new audio from scratch
    """
    ffmpeg = get_ffmpeg_exe()
    cap_dur = max(1, min(15, int(max_duration_sec or 15)))

    with tempfile.TemporaryDirectory(prefix="symm_prep_") as tmpdir:
        in_path = os.path.join(tmpdir, "input.mp4")
        out_path = os.path.join(tmpdir, "prepped.mp4")
        with open(in_path, "wb") as f:
            f.write(video_bytes)

        scale_vf = (
            "scale='if(gte(iw,ih),min(1280,iw),min(720,iw))':-2:flags=lanczos"
        )
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            in_path,
            "-t",
            str(cap_dur),
            "-vf",
            scale_vf,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
        ]
        if strip_audio:
            cmd.append("-an")
        else:
            cmd.extend(["-c:a", "aac", "-b:a", "128k", "-ac", "2"])
        cmd.append(out_path)

        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=90
        )
        if proc.returncode != 0 or not os.path.exists(out_path):
            if not strip_audio:
                # Source video may have had no audio track; retry with `-an`
                cmd_no_audio = [
                    ffmpeg,
                    "-y",
                    "-i",
                    in_path,
                    "-t",
                    str(cap_dur),
                    "-vf",
                    scale_vf,
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-preset",
                    "veryfast",
                    "-an",
                    out_path,
                ]
                proc2 = subprocess.run(
                    cmd_no_audio,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=90,
                )
                if proc2.returncode != 0 or not os.path.exists(out_path):
                    err_tail = (proc2.stderr or "")[-400:].strip()
                    raise RuntimeError(f"FFmpeg video prep failed: {err_tail}")
            else:
                err_tail = (proc.stderr or "")[-400:].strip()
                raise RuntimeError(f"FFmpeg video prep failed: {err_tail}")

        with open(out_path, "rb") as f:
            return f.read()


def extract_first_frame_png_sync(video_bytes: bytes) -> tuple[bytes, str] | None:
    """Extract the first frame of a video buffer as `(png_bytes, 'image/png')`."""
    ffmpeg = get_ffmpeg_exe()
    with tempfile.TemporaryDirectory(prefix="symm_frame_") as tmpdir:
        in_path = os.path.join(tmpdir, "in.mp4")
        out_path = os.path.join(tmpdir, "frame0.png")
        with open(in_path, "wb") as f:
            f.write(video_bytes)
        proc = subprocess.run(
            [ffmpeg, "-y", "-i", in_path, "-frames:v", "1", out_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        if proc.returncode == 0 and os.path.exists(out_path):
            with open(out_path, "rb") as pf:
                raw = pf.read()
            if raw:
                return raw, "image/png"
    return None
