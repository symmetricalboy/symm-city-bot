"""
Top 3 Leaderboard Podium Image Renderer for symm.city bot.

Renders a high-resolution dark-mode Top 3 Podium card (matching `/history`'s `#0b0f19` / `#111827`
visual aesthetic) using 2x supersampled Pillow drawing + bundled DejaVu Sans TrueType fonts,
complete with circular Telegram profile avatars (with monogram fallbacks), vector crown & rank
medallions, 3D beveled Gold/Silver/Bronze pedestals, CXP totals, and Level title pills.
"""

import asyncio
from collections import OrderedDict
import io
import logging
import time
import unicodedata
from datetime import datetime, timezone

from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageDraw, ImageFilter, ImageFont

logger = logging.getLogger(__name__)

# 15-minute in-memory LRU cache for circular avatar thumbnails: user_id -> (timestamp, bytes | None)
_AVATAR_CACHE: OrderedDict[int, tuple[float, bytes | None]] = OrderedDict()
_AVATAR_TTL_SECONDS = 900.0
_AVATAR_CACHE_MAX = 128

_FONT_PATH_REGULAR: str | None = None
_FONT_PATH_BOLD: str | None = None


def _get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    global _FONT_PATH_REGULAR, _FONT_PATH_BOLD
    if _FONT_PATH_REGULAR is None:
        _FONT_PATH_REGULAR = font_manager.findfont(
            FontProperties(family="DejaVu Sans", weight="normal")
        )
    if _FONT_PATH_BOLD is None:
        _FONT_PATH_BOLD = font_manager.findfont(
            FontProperties(family="DejaVu Sans", weight="bold")
        )
    path = _FONT_PATH_BOLD if bold else _FONT_PATH_REGULAR
    return ImageFont.truetype(path, size=size)


def normalize_podium_text(text: str, fallback: str = "Member") -> str:
    """
    Normalize stylized Unicode (e.g. Mathematical Double-Struck `𝔸𝕞𝕚𝕣` -> `Amir`)
    and strip supplemental color emojis, rare math operators, and pictographs not in DejaVu Sans.
    """
    if not text:
        return fallback
    norm = unicodedata.normalize("NFKC", text)
    cleaned: list[str] = []
    for ch in norm:
        cp = ord(ch)
        if cp >= 0x1F000 or cp in (0xFE0F, 0xFE0E, 0x200D):
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("C"):
            continue
        # Allow standard Latin, IPA/phonetic small-caps, Greek, Cyrillic, general punctuation,
        # currency, arrows, and geometric shapes supported by DejaVu Sans; skip obscure blocks.
        allowed = (
            (0x0020 <= cp <= 0x052F)  # Basic + Extended Latin, IPA, Greek, Cyrillic
            or (0x1D00 <= cp <= 0x1EFF)  # Phonetic small-caps & Latin Extended Additional
            or (0x2000 <= cp <= 0x206F)  # General Punctuation (•, —, …)
            or (0x20A0 <= cp <= 0x20CF)  # Currency
            or cp in (0x25C6, 0x25C7, 0x25C8, 0x2605, 0x2606)  # ◆ ◇ ◈ ★ ☆
        )
        if not allowed:
            continue
        cleaned.append(ch)
    collapsed = " ".join("".join(cleaned).split()).strip()
    return collapsed if collapsed else fallback


def extract_initials(name: str) -> str:
    """Extract 1–2 clean uppercase initials for monogram fallback avatars."""
    clean = normalize_podium_text(name, fallback="").lstrip("@")
    alnum_words = [
        "".join(ch for ch in part if ch.isalnum())
        for part in clean.split()
    ]
    alnum_words = [w for w in alnum_words if w]
    if len(alnum_words) >= 2:
        return (alnum_words[0][0] + alnum_words[1][0]).upper()
    if len(alnum_words) == 1:
        w = alnum_words[0]
        return w[:2].upper() if len(w) >= 2 else w[0].upper()
    return "?"


async def fetch_user_avatar_bytes(bot, user_id: int) -> bytes | None:
    """Fetch a user's or channel's Telegram profile photo bytes with a 15-minute in-memory cache."""
    if not bot or not user_id:
        return None

    now = time.monotonic()
    cached = _AVATAR_CACHE.get(user_id)
    if cached is not None:
        ts, data = cached
        if now - ts < _AVATAR_TTL_SECONDS:
            _AVATAR_CACHE.move_to_end(user_id)
            return data

    photo_bytes: bytes | None = None
    try:
        if user_id > 0:
            photos = await asyncio.wait_for(
                bot.get_user_profile_photos(user_id=user_id, limit=1),
                timeout=2.5,
            )
            if photos and photos.photos and photos.photos[0]:
                sizes = photos.photos[0]
                chosen = sizes[min(1, len(sizes) - 1)]
                tg_file = await asyncio.wait_for(chosen.get_file(), timeout=2.5)
                raw = await asyncio.wait_for(tg_file.download_as_bytearray(), timeout=3.0)
                photo_bytes = bytes(raw)

        if photo_bytes is None:
            chat = await asyncio.wait_for(bot.get_chat(user_id), timeout=2.5)
            if getattr(chat, "photo", None) and getattr(chat.photo, "small_file_id", None):
                tg_file = await asyncio.wait_for(
                    bot.get_file(chat.photo.small_file_id), timeout=2.5
                )
                raw = await asyncio.wait_for(tg_file.download_as_bytearray(), timeout=3.0)
                photo_bytes = bytes(raw)
    except Exception as e:
        logger.debug("Avatar fetch skipped for %s: %s", user_id, e)

    _AVATAR_CACHE[user_id] = (now, photo_bytes)
    _AVATAR_CACHE.move_to_end(user_id)
    while len(_AVATAR_CACHE) > _AVATAR_CACHE_MAX:
        _AVATAR_CACHE.popitem(last=False)

    return photo_bytes


def _hex_rgb(hex_str: str) -> tuple[int, int, int]:
    h = hex_str.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hex_rgba(hex_str: str, alpha: int = 255) -> tuple[int, int, int, int]:
    r, g, b = _hex_rgb(hex_str)
    return r, g, b, alpha


def _lerp_color(
    c1: tuple[int, int, int], c2: tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return (
        int(round(c1[0] + (c2[0] - c1[0]) * t)),
        int(round(c1[1] + (c2[1] - c1[1]) * t)),
        int(round(c1[2] + (c2[2] - c1[2]) * t)),
    )


def _make_vertical_gradient(
    width: int,
    height: int,
    top_hex: str,
    bottom_hex: str,
    alpha_top: int = 255,
    alpha_bottom: int = 255,
) -> Image.Image:
    """Create a smooth vertical RGBA gradient image."""
    c_top = _hex_rgb(top_hex)
    c_bot = _hex_rgb(bottom_hex)
    strip = Image.new("RGBA", (1, max(1, height)))
    px = strip.load()
    denom = max(1, height - 1)
    for y in range(height):
        t = y / denom
        r, g, b = _lerp_color(c_top, c_bot, t)
        a = int(round(alpha_top + (alpha_bottom - alpha_top) * t))
        px[0, y] = (r, g, b, a)
    return strip.resize((max(1, width), max(1, height)), resample=Image.Resampling.BILINEAR)


def _add_radial_glow(
    canvas: Image.Image,
    cx: int,
    cy: int,
    radius: int,
    color_hex: str,
    max_alpha: int = 55,
):
    """Composite a soft radial spotlight onto `canvas`."""
    r, g, b = _hex_rgb(color_hex)
    size = radius * 2
    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(glow)
    steps = 36
    for i in range(steps, 0, -1):
        frac = i / steps
        rad = int(radius * frac)
        alpha = int(max_alpha * ((1.0 - frac) ** 1.85))
        if alpha <= 0 or rad <= 0:
            continue
        draw.ellipse(
            (radius - rad, radius - rad, radius + rad, radius + rad),
            fill=(r, g, b, alpha),
        )
    canvas.alpha_composite(glow, dest=(cx - radius, cy - radius))


def _fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    start_size: int,
    min_size: int,
    bold: bool = True,
) -> tuple[str, ImageFont.FreeTypeFont]:
    """Choose font size (and truncate with ellipsis if needed) so `text` fits within `max_width`."""
    for sz in range(start_size, min_size - 1, -2):
        fnt = _get_font(sz, bold=bold)
        bbox = draw.textbbox((0, 0), text, font=fnt)
        if (bbox[2] - bbox[0]) <= max_width:
            return text, fnt

    fnt = _get_font(min_size, bold=bold)
    trimmed = text
    while len(trimmed) > 3:
        trimmed = trimmed[:-1].rstrip()
        candidate = trimmed + "…"
        bbox = draw.textbbox((0, 0), candidate, font=fnt)
        if (bbox[2] - bbox[0]) <= max_width:
            return candidate, fnt
    return text[:3] + "…", fnt


def _build_circular_avatar(
    avatar_bytes: bytes | None,
    diameter: int,
    initials: str,
    grad_top: str,
    grad_bot: str,
    claimed: bool = True,
) -> Image.Image:
    """Create a crisp circular RGBA avatar of size `(diameter, diameter)`."""
    mask = Image.new("L", (diameter, diameter), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, diameter - 1, diameter - 1), fill=255)

    img: Image.Image | None = None
    if avatar_bytes and claimed:
        try:
            src = Image.open(io.BytesIO(avatar_bytes)).convert("RGBA")
            w, h = src.size
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            cropped = src.crop((left, top, left + side, top + side))
            img = cropped.resize(
                (diameter, diameter), resample=Image.Resampling.LANCZOS
            )
        except Exception:
            img = None

    if img is None:
        top_c = grad_top if claimed else "#1e293b"
        bot_c = grad_bot if claimed else "#0f172a"
        img = _make_vertical_gradient(diameter, diameter, top_c, bot_c)

        # Subtle glossy top sheen composited on a separate RGBA layer so base alpha stays 255
        sheen = Image.new("RGBA", (diameter, diameter), (0, 0, 0, 0))
        ImageDraw.Draw(sheen).ellipse(
            (
                int(diameter * 0.06),
                -int(diameter * 0.32),
                int(diameter * 0.94),
                int(diameter * 0.50),
            ),
            fill=(255, 255, 255, 26 if claimed else 10),
        )
        img = Image.alpha_composite(img, sheen)

        txt_layer = Image.new("RGBA", (diameter, diameter), (0, 0, 0, 0))
        td = ImageDraw.Draw(txt_layer)
        label = initials if claimed else "?"
        fnt = _get_font(int(diameter * (0.37 if len(label) <= 2 else 0.29)), bold=True)
        bbox = td.textbbox((0, 0), label, font=fnt)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = (diameter - tw) / 2 - bbox[0]
        ty = (diameter - th) / 2 - bbox[1]
        td.text(
            (tx, ty),
            label,
            font=fnt,
            fill=(255, 255, 255, 244 if claimed else 95),
        )
        img = Image.alpha_composite(img, txt_layer)

    out = Image.new("RGBA", (diameter, diameter), (0, 0, 0, 0))
    out.paste(img, (0, 0), mask=mask)
    return out


def _draw_crown(
    draw: ImageDraw.ImageDraw,
    cx: int,
    bottom_y: int,
    width: int,
    height: int,
):
    """Draw a crisp geometric 5-point golden crown centered at `(cx, bottom_y)`."""
    half_w = width // 2
    left = cx - half_w
    right = cx + half_w
    base_h = int(height * 0.24)
    body_bottom = bottom_y - base_h
    top_y = bottom_y - height

    pts = [
        (left + int(width * 0.08), body_bottom),
        (left, top_y + int(height * 0.32)),
        (left + int(width * 0.23), top_y + int(height * 0.58)),
        (cx, top_y),
        (right - int(width * 0.23), top_y + int(height * 0.58)),
        (right, top_y + int(height * 0.32)),
        (right - int(width * 0.08), body_bottom),
    ]
    draw.polygon(
        pts,
        fill=_hex_rgba("#facc15"),
        outline=_hex_rgba("#fef08a"),
        width=3,
    )

    band_rect = (
        left + int(width * 0.07),
        body_bottom - 2,
        right - int(width * 0.07),
        bottom_y,
    )
    draw.rounded_rectangle(
        band_rect,
        radius=6,
        fill=_hex_rgba("#eab308"),
        outline=_hex_rgba("#fef08a"),
        width=3,
    )

    for px, py, r_dot in [
        (left, top_y + int(height * 0.30), 6),
        (cx, top_y - 2, 8),
        (right, top_y + int(height * 0.30), 6),
    ]:
        draw.ellipse(
            (px - r_dot, py - r_dot, px + r_dot, py + r_dot),
            fill=_hex_rgba("#fef9c3"),
            outline=_hex_rgba("#ca8a04"),
            width=2,
        )

    jr = 5
    by_mid = (body_bottom + bottom_y) // 2
    draw.ellipse(
        (cx - jr, by_mid - jr, cx + jr, by_mid + jr),
        fill=_hex_rgba("#38bdf8"),
    )


def render_podium_png(
    title: str,
    entries: list[dict],
    *,
    is_timeframe: bool = False,
    subtitle_detail: str | None = None,
) -> io.BytesIO:
    """
    Render a sleek 1080x700 dark-mode Top 3 Podium PNG in memory (drawn at 2160x1400 2x supersampling).
    """
    W, H = 2160, 1400

    canvas = Image.new("RGBA", (W, H), _hex_rgba("#0b0f19"))

    # Ambient spotlights behind card
    _add_radial_glow(canvas, W // 2, int(H * 0.50), 650, "#f59e0b", max_alpha=42)
    _add_radial_glow(canvas, int(W * 0.24), int(H * 0.58), 480, "#38bdf8", max_alpha=30)
    _add_radial_glow(canvas, int(W * 0.76), int(H * 0.62), 460, "#fb923c", max_alpha=28)

    # Main inner card container
    card_x0, card_y0 = 72, 184
    card_x1, card_y1 = W - 72, H - 62
    card_radius = 36

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        (card_x0, card_y0 + 10, card_x1, card_y1 + 10),
        radius=card_radius,
        fill=(0, 0, 0, 115),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=18))
    canvas.alpha_composite(shadow)

    card_w = card_x1 - card_x0
    card_h = card_y1 - card_y0
    card_grad = _make_vertical_gradient(card_w, card_h, "#131c2e", "#0d1321")

    # Interior spotlight glow inside the card behind #1, #2, #3
    _add_radial_glow(card_grad, card_w // 2, int(card_h * 0.44), 530, "#fbbf24", max_alpha=40)
    _add_radial_glow(card_grad, int(card_w * 0.21), int(card_h * 0.55), 390, "#94a3b8", max_alpha=26)
    _add_radial_glow(card_grad, int(card_w * 0.79), int(card_h * 0.60), 370, "#fb923c", max_alpha=26)

    # Subtle translucent symmetrical grid overlay composited properly via alpha_composite
    grid_overlay = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
    g_draw = ImageDraw.Draw(grid_overlay)
    for gy in range(130, card_h - 70, 150):
        g_draw.line((52, gy, card_w - 52, gy), fill=(148, 163, 184, 22), width=2)
    for gx_frac in (0.215, 0.50, 0.785):
        gx = int(card_w * gx_frac)
        g_draw.line((gx, 56, gx, card_h - 56), fill=(148, 163, 184, 18), width=2)
    card_grad = Image.alpha_composite(card_grad, grid_overlay)

    card_mask = Image.new("L", (card_w, card_h), 0)
    ImageDraw.Draw(card_mask).rounded_rectangle(
        (0, 0, card_w - 1, card_h - 1), radius=card_radius, fill=255
    )
    canvas.paste(card_grad, (card_x0, card_y0), mask=card_mask)

    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (card_x0, card_y0, card_x1, card_y1),
        radius=card_radius,
        outline=_hex_rgba("#293548"),
        width=3,
    )

    # ── Top Header Banner ──
    f_brand = _get_font(44, bold=True)
    f_sub = _get_font(28, bold=False)
    f_right_main = _get_font(36, bold=True)
    f_right_sub = _get_font(25, bold=False)

    draw.text(
        (80, 48),
        "symm.city  •  Leaderboard Podium",
        font=f_brand,
        fill=_hex_rgba("#f8fafc"),
    )

    clean_title = normalize_podium_text(title, fallback="Leaderboard")
    if subtitle_detail:
        clean_title = f"{clean_title}  •  {normalize_podium_text(subtitle_detail, '')}"
    draw.text(
        (80, 112),
        clean_title,
        font=f_sub,
        fill=_hex_rgba("#94a3b8"),
    )

    now_str = datetime.now(timezone.utc).strftime("%b %d, %Y  •  UTC")
    if entries:
        top1_cxp = int(entries[0].get("cxp") or 0)
        prefix = "+" if (is_timeframe and top1_cxp > 0) else ""
        lead_str = f"#1 Lead: {prefix}{top1_cxp:,} CXP"
        rb = draw.textbbox((0, 0), lead_str, font=f_right_main)
        draw.text(
            (W - 80 - (rb[2] - rb[0]), 52),
            lead_str,
            font=f_right_main,
            fill=_hex_rgba("#fde047"),
        )
    db_box = draw.textbbox((0, 0), now_str, font=f_right_sub)
    draw.text(
        (W - 80 - (db_box[2] - db_box[0]), 114),
        now_str,
        font=f_right_sub,
        fill=_hex_rgba("#94a3b8"),
    )

    # ── Prepare 3 Podium Slots: Left = #2, Center = #1, Right = #3 ──
    by_rank = {int(e["rank"]): e for e in entries[:3]}

    c1 = max(1, abs(int(by_rank.get(1, {}).get("cxp") or 1)))
    c2 = abs(int(by_rank.get(2, {}).get("cxp") or 0))
    c3 = abs(int(by_rank.get(3, {}).get("cxp") or 0))

    r2 = min(0.88, max(0.55, (c2 / c1) if 2 in by_rank else 0.68))
    r3 = min(r2 - 0.10, max(0.42, (c3 / c1) if 3 in by_rank else 0.50))

    h1 = 486
    h2 = min(h1 - 92, max(352, int(round(320 + (r2 - 0.55) * 220))))
    h3 = min(h2 - 64, max(284, int(round(268 + (r3 - 0.42) * 160))))

    floor_y = card_y1 - 64

    slot_specs = [
        {
            "rank": 2,
            "ord_label": "2ND",
            "cx": int(W * 0.245),
            "width": 520,
            "height": h2,
            "avatar_d": 188,
            "ring_outer": "#e2e8f0",
            "ring_inner": "#94a3b8",
            "ped_top": "#334155",
            "ped_bot": "#131c2b",
            "lip_fill": "#475569",
            "rim_hex": "#e2e8f0",
            "score_hex": "#f8fafc",
            "badge_bg": "#475569",
            "badge_fg": "#f8fafc",
            "avatar_g1": "#475569",
            "avatar_g2": "#1e293b",
        },
        {
            "rank": 1,
            "ord_label": "1ST",
            "cx": W // 2,
            "width": 576,
            "height": h1,
            "avatar_d": 220,
            "ring_outer": "#fde047",
            "ring_inner": "#d97706",
            "ped_top": "#78350f",
            "ped_bot": "#291205",
            "lip_fill": "#b45309",
            "rim_hex": "#fde047",
            "score_hex": "#fde047",
            "badge_bg": "#eab308",
            "badge_fg": "#1c1917",
            "avatar_g1": "#d97706",
            "avatar_g2": "#451a03",
        },
        {
            "rank": 3,
            "ord_label": "3RD",
            "cx": int(W * 0.755),
            "width": 520,
            "height": h3,
            "avatar_d": 184,
            "ring_outer": "#fdba74",
            "ring_inner": "#c2410c",
            "ped_top": "#6c2710",
            "ped_bot": "#261008",
            "lip_fill": "#9a3412",
            "rim_hex": "#fdba74",
            "score_hex": "#fed7aa",
            "badge_bg": "#ea580c",
            "badge_fg": "#fff7ed",
            "avatar_g1": "#c2410c",
            "avatar_g2": "#431407",
        },
    ]

    for spec in [slot_specs[0], slot_specs[2], slot_specs[1]]:
        rank = spec["rank"]
        entry = by_rank.get(rank)
        claimed = entry is not None

        cx = spec["cx"]
        pw = spec["width"]
        ph = spec["height"]
        px0 = cx - pw // 2
        px1 = cx + pw // 2
        py0 = floor_y - ph
        py1 = floor_y
        lip_h = 26

        # 1. Pedestal drop shadow
        p_shad = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(p_shad).rounded_rectangle(
            (px0 - 6, py0 + 8, px1 + 6, py1 + 8),
            radius=20,
            fill=(0, 0, 0, 125 if rank == 1 else 88),
        )
        p_shad = p_shad.filter(ImageFilter.GaussianBlur(radius=16))
        canvas.alpha_composite(p_shad)

        # 2. 3D Isometric Top Bevel Deck on its own RGBA overlay for proper alpha blending
        bevel_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        bdraw = ImageDraw.Draw(bevel_layer)
        bevel_inset = 24
        deck_pts = [
            (px0 + bevel_inset, py0 - lip_h),
            (px1 - bevel_inset, py0 - lip_h),
            (px1, py0),
            (px0, py0),
        ]
        bdraw.polygon(
            deck_pts,
            fill=_hex_rgba(spec["lip_fill"] if claimed else "#1e293b", 230 if claimed else 130),
            outline=_hex_rgba(spec["rim_hex"] if claimed else "#334155", 210 if claimed else 95),
        )
        canvas.alpha_composite(bevel_layer)

        # 3. Pedestal front gradient face + true translucent watermark overlay
        ped_top_col = spec["ped_top"] if claimed else "#182234"
        ped_bot_col = spec["ped_bot"] if claimed else "#0f172a"
        ped_face = _make_vertical_gradient(
            pw,
            ph,
            ped_top_col,
            ped_bot_col,
            alpha_top=248 if claimed else 160,
            alpha_bottom=255 if claimed else 180,
        )

        wm_layer = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
        wm_draw = ImageDraw.Draw(wm_layer)
        f_water = _get_font(int(ph * 0.54), bold=True)
        wb = wm_draw.textbbox((0, 0), str(rank), font=f_water)
        wx = (pw - (wb[2] - wb[0])) / 2 - wb[0]
        wy = (ph - (wb[3] - wb[1])) / 2 - wb[1] + int(ph * 0.04)
        wm_draw.text(
            (wx, wy),
            str(rank),
            font=f_water,
            fill=(255, 255, 255, 20 if claimed else 12),
        )
        ped_face = Image.alpha_composite(ped_face, wm_layer)

        ped_mask = Image.new("L", (pw, ph), 0)
        ImageDraw.Draw(ped_mask).rounded_rectangle(
            (0, 0, pw - 1, ph - 1), radius=14, fill=255
        )
        ImageDraw.Draw(ped_mask).rectangle((0, 0, pw - 1, 20), fill=255)
        canvas.paste(ped_face, (px0, py0), mask=ped_mask)

        # Pedestal border + glowing top rim bar on RGBA overlay
        ped_fg = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        pf_draw = ImageDraw.Draw(ped_fg)
        pf_draw.rounded_rectangle(
            (px0, py0, px1, py1),
            radius=14,
            outline=_hex_rgba(spec["rim_hex"] if claimed else "#334155", 135 if claimed else 75),
            width=3,
        )
        pf_draw.line(
            (px0 + 3, py0 + 2, px1 - 3, py0 + 2),
            fill=_hex_rgba(spec["rim_hex"] if claimed else "#475569", 255 if claimed else 120),
            width=6,
        )

        # 4. Pedestal Interior Text
        f_ord = _get_font(25, bold=True)
        ord_txt = f"{spec['ord_label']} PLACE"
        ob = pf_draw.textbbox((0, 0), ord_txt, font=f_ord)
        pf_draw.text(
            (cx - (ob[2] - ob[0]) / 2 - ob[0], py0 + 24 - ob[1]),
            ord_txt,
            font=f_ord,
            fill=_hex_rgba(spec["rim_hex"] if claimed else "#64748b", 225 if claimed else 140),
        )

        if claimed and entry is not None:
            cxp_val = int(entry.get("cxp") or 0)
            total_cxp = int(entry.get("total_cxp", cxp_val) or 0)
            lvl = int(entry.get("level") or 1)
            lvl_title = normalize_podium_text(entry.get("level_title") or "", fallback="")

            sign_prefix = "+" if (is_timeframe and cxp_val > 0) else ""
            score_str = f"{sign_prefix}{cxp_val:,} CXP"
            score_str_fit, f_score = _fit_text(
                pf_draw,
                score_str,
                pw - 56,
                start_size=54 if rank == 1 else 45,
                min_size=32,
                bold=True,
            )
            sb = pf_draw.textbbox((0, 0), score_str_fit, font=f_score)
            score_y = py0 + (78 if rank == 1 else 66)
            pf_draw.text(
                (cx - (sb[2] - sb[0]) / 2 - sb[0], score_y - sb[1]),
                score_str_fit,
                font=f_score,
                fill=_hex_rgba(spec["score_hex"]),
            )

            if is_timeframe:
                tot_txt = f"Total: {total_cxp:,} CXP"
                f_tot = _get_font(23, bold=False)
                tb = pf_draw.textbbox((0, 0), tot_txt, font=f_tot)
                pf_draw.text(
                    (
                        cx - (tb[2] - tb[0]) / 2 - tb[0],
                        score_y + (sb[3] - sb[1]) + 14 - tb[1],
                    ),
                    tot_txt,
                    font=f_tot,
                    fill=_hex_rgba("#cbd5e1", 220),
                )

            pill_label = f"Lvl {lvl}  •  {lvl_title}" if lvl_title else f"Level {lvl}"
            pill_label_fit, f_pill = _fit_text(
                pf_draw, pill_label, pw - 84, start_size=24, min_size=19, bold=True
            )
            pb = pf_draw.textbbox((0, 0), pill_label_fit, font=f_pill)
            pill_w = (pb[2] - pb[0]) + 44
            pill_h = 46
            pill_x0 = cx - pill_w // 2
            pill_y0 = py1 - pill_h - 22
            pf_draw.rounded_rectangle(
                (pill_x0, pill_y0, pill_x0 + pill_w, pill_y0 + pill_h),
                radius=23,
                fill=_hex_rgba("#090d16", 215),
                outline=_hex_rgba(spec["rim_hex"], 135),
                width=2,
            )
            pf_draw.text(
                (
                    cx - (pb[2] - pb[0]) / 2 - pb[0],
                    pill_y0 + (pill_h - (pb[3] - pb[1])) / 2 - pb[1],
                ),
                pill_label_fit,
                font=f_pill,
                fill=_hex_rgba("#f1f5f9"),
            )
        else:
            f_empty = _get_font(29, bold=True)
            em_txt = "Unclaimed"
            eb = pf_draw.textbbox((0, 0), em_txt, font=f_empty)
            pf_draw.text(
                (cx - (eb[2] - eb[0]) / 2 - eb[0], py0 + ph // 2 - (eb[3] - eb[1]) // 2 - eb[1]),
                em_txt,
                font=f_empty,
                fill=_hex_rgba("#64748b", 190),
            )

        canvas.alpha_composite(ped_fg)
        draw = ImageDraw.Draw(canvas)

        # 5. Name, @username, Avatar, Medallion, & Crown above pedestal
        disp_raw = entry.get("name", "Member") if entry else "Open Spot"
        disp_clean = normalize_podium_text(disp_raw, fallback="Member")
        uname_raw = (entry.get("username") or "").lstrip("@") if entry else ""
        show_uname = (
            bool(uname_raw)
            and uname_raw.lower() != disp_clean.lstrip("@").lower()
        )

        cursor_bottom = py0 - lip_h - 18

        if show_uname:
            handle_txt = f"@{uname_raw}"
            handle_fit, f_handle = _fit_text(
                draw, handle_txt, pw - 30, start_size=23, min_size=19, bold=False
            )
            hb = draw.textbbox((0, 0), handle_fit, font=f_handle)
            hh = hb[3] - hb[1]
            draw.text(
                (cx - (hb[2] - hb[0]) / 2 - hb[0], cursor_bottom - hh - hb[1]),
                handle_fit,
                font=f_handle,
                fill=_hex_rgba("#94a3b8"),
            )
            cursor_bottom -= hh + 12

        name_fit, f_name = _fit_text(
            draw,
            disp_clean,
            pw - 24,
            start_size=36 if rank == 1 else 32,
            min_size=23,
            bold=True,
        )
        nb = draw.textbbox((0, 0), name_fit, font=f_name)
        nh = nb[3] - nb[1]
        draw.text(
            (cx - (nb[2] - nb[0]) / 2 - nb[0], cursor_bottom - nh - nb[1]),
            name_fit,
            font=f_name,
            fill=_hex_rgba("#f8fafc" if claimed else "#64748b"),
        )
        cursor_bottom -= nh + 40  # clean gap below medallion badge

        av_d = spec["avatar_d"]
        av_cy = cursor_bottom - av_d // 2
        av_x0 = cx - av_d // 2
        av_y0 = av_cy - av_d // 2

        if rank == 1 and claimed:
            _add_radial_glow(canvas, cx, av_cy, int(av_d * 0.85), "#fbbf24", max_alpha=68)
            draw = ImageDraw.Draw(canvas)

        ring_pad = 10
        draw.ellipse(
            (
                av_x0 - ring_pad,
                av_y0 - ring_pad,
                av_x0 + av_d + ring_pad,
                av_y0 + av_d + ring_pad,
            ),
            fill=_hex_rgba(spec["ring_outer"] if claimed else "#334155"),
            outline=_hex_rgba(spec["ring_inner"] if claimed else "#1e293b"),
            width=4,
        )
        inner_moat = 4
        draw.ellipse(
            (
                av_x0 - inner_moat,
                av_y0 - inner_moat,
                av_x0 + av_d + inner_moat,
                av_y0 + av_d + inner_moat,
            ),
            fill=_hex_rgba("#0b0f19"),
        )

        av_img = _build_circular_avatar(
            entry.get("avatar_bytes") if entry else None,
            diameter=av_d,
            initials=extract_initials(disp_raw),
            grad_top=spec["avatar_g1"],
            grad_bot=spec["avatar_g2"],
            claimed=claimed,
        )
        canvas.alpha_composite(av_img, dest=(av_x0, av_y0))
        draw = ImageDraw.Draw(canvas)

        med_r = 28 if rank == 1 else 24
        med_cy = av_y0 + av_d + 2
        draw.ellipse(
            (cx - med_r - 4, med_cy - med_r - 4, cx + med_r + 4, med_cy + med_r + 4),
            fill=_hex_rgba("#0b0f19"),
        )
        draw.ellipse(
            (cx - med_r, med_cy - med_r, cx + med_r, med_cy + med_r),
            fill=_hex_rgba(spec["badge_bg"] if claimed else "#334155"),
            outline=_hex_rgba(spec["ring_outer"] if claimed else "#64748b"),
            width=3,
        )
        f_med = _get_font(28 if rank == 1 else 24, bold=True)
        mb = draw.textbbox((0, 0), str(rank), font=f_med)
        draw.text(
            (cx - (mb[2] - mb[0]) / 2 - mb[0], med_cy - (mb[3] - mb[1]) / 2 - mb[1]),
            str(rank),
            font=f_med,
            fill=_hex_rgba(spec["badge_fg"] if claimed else "#cbd5e1"),
        )

        if rank == 1 and claimed:
            _draw_crown(
                draw,
                cx=cx,
                bottom_y=av_y0 - ring_pad - 8,
                width=108,
                height=68,
            )

    # Grounded stage plinth bar across the base of all 3 pedestals
    plinth_x0 = slot_specs[0]["cx"] - slot_specs[0]["width"] // 2 - 36
    plinth_x1 = slot_specs[2]["cx"] + slot_specs[2]["width"] // 2 + 36
    draw.rounded_rectangle(
        (plinth_x0, floor_y - 4, plinth_x1, floor_y + 18),
        radius=9,
        fill=_hex_rgba("#1e293b"),
        outline=_hex_rgba("#475569"),
        width=2,
    )

    final_img = canvas.convert("RGB").resize(
        (W // 2, H // 2), resample=Image.Resampling.LANCZOS
    )
    buf = io.BytesIO()
    final_img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf
