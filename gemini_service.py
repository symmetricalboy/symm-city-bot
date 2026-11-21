"""
Unified async Gemini API client wrapper for symm.city bot.
Uses:
  - `gemini-4-neon` via the GA Interactions API (`client.aio.interactions.create`)
  - `gemini-nano-banana-2.1` (Nano Banana 2.1) for contextual image generation & editing
  - `gemini-omni-1.2-flash` (Gemini Omni 1.2 Flash) for video & GIF generation, editing, and extension
  - `gemini-embedding-2` (`768` dimensions) via `client.aio.models.embed_content`
"""

import asyncio
import base64
import json
import logging
import os
import re
import tempfile
from typing import Any, Awaitable, Callable

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-4-neon")
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-nano-banana-2.1")
GEMINI_VIDEO_MODEL = os.getenv("GEMINI_VIDEO_MODEL", "gemini-omni-1.2-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
GEMINI_EMBEDDING_DIM = 768

VALID_ASPECT_RATIOS = {
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
    "1:4",
    "4:1",
    "1:8",
    "8:1",
}

VALID_VIDEO_ASPECT_RATIOS = {"16:9", "9:16"}
VALID_VIDEO_RESOLUTIONS = {"360p", "720p", "1080p", "4k"}

_client: genai.Client | None = None
_client_key: str | None = None
_video_client: genai.Client | None = None
_video_client_key: str | None = None


def get_gemini_client() -> genai.Client | None:
    """Return a cached `genai.Client` initialized with `GEMINI_API_KEY`, refreshing if env changed."""
    global _client, _client_key
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return None
    if _client is None or _client_key != api_key:
        try:
            _client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=90_000),
            )
            _client_key = api_key
        except Exception as e:
            logger.error("Failed to initialize Gemini client: %s", e)
            _client = None
    return _client


def get_video_gemini_client() -> genai.Client | None:
    """Return a cached `genai.Client` with an extended 420s HTTP timeout for Gemini Omni 1.2 Flash video generation."""
    global _video_client, _video_client_key
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return None
    if _video_client is None or _video_client_key != api_key:
        try:
            _video_client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=420_000),
            )
            _video_client_key = api_key
        except Exception as e:
            logger.error("Failed to initialize Gemini video client: %s", e)
            _video_client = None
    return _video_client


def reset_gemini_client():
    """Discard cached `genai.Client` instances so subsequent calls establish fresh HTTP connections."""
    global _client, _video_client
    _client = None
    _video_client = None


async def _create_interaction_with_timeout(
    kwargs: dict[str, Any], *, timeout_sec: float = 75.0
) -> Any:
    """
    Call `client.aio.interactions.create(**kwargs)` with an explicit asyncio deadline.
    `google-genai`'s async Interactions client defaults to `Timeout(timeout=None)`,
    so enforcing `asyncio.wait_for` prevents stalled sockets from ever hanging indefinitely.
    """
    for attempt in range(2):
        client = get_gemini_client()
        if not client:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        try:
            return await asyncio.wait_for(
                client.aio.interactions.create(**kwargs),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Gemini interactions.create(%s) timed out after %.1fs (attempt %d/2); resetting client pool.",
                kwargs.get("model"),
                timeout_sec,
                attempt + 1,
            )
            reset_gemini_client()
            if attempt == 1:
                raise
        except Exception as err:
            # Retry once on transient connection/pool transport errors
            err_name = type(err).__name__.lower()
            if attempt == 0 and any(
                k in err_name for k in ("connect", "pool", "protocol", "readerror", "remoteprotocol")
            ):
                logger.warning(
                    "Transient transport error on Gemini (%s); resetting client and retrying once.",
                    err,
                )
                reset_gemini_client()
                continue
            raise


def extract_interaction_text(interaction: Any) -> str:
    """Safely extract combined model text output from an Interaction object."""
    out = getattr(interaction, "output_text", None)
    if out:
        return out.strip()

    parts: list[str] = []
    for step in getattr(interaction, "steps", None) or []:
        if getattr(step, "type", None) == "model_output":
            for item in getattr(step, "content", None) or []:
                if getattr(item, "type", None) == "text" and getattr(item, "text", None):
                    parts.append(item.text)
    return "".join(parts).strip()


def _decode_image_part(img_obj: Any) -> tuple[bytes | None, str]:
    """Decode an `ImageContent` object's `data` field (`str` base64 or `bytes`) to `(bytes, mime_type)`."""
    if not img_obj:
        return None, "image/png"
    raw_data = getattr(img_obj, "data", None)
    mime = getattr(img_obj, "mime_type", None) or "image/png"
    if not raw_data:
        return None, mime
    if isinstance(raw_data, (bytes, bytearray)):
        return bytes(raw_data), mime
    if isinstance(raw_data, str):
        return base64.b64decode(raw_data), mime
    return None, mime


def extract_interaction_image(interaction: Any) -> tuple[bytes | None, str]:
    """Safely extract the last generated image `(bytes, mime_type)` from an Interaction object."""
    out_img = getattr(interaction, "output_image", None)
    if out_img:
        b, m = _decode_image_part(out_img)
        if b:
            return b, m

    for step in reversed(getattr(interaction, "steps", None) or []):
        if getattr(step, "type", None) == "model_output":
            for item in reversed(getattr(step, "content", None) or []):
                if getattr(item, "type", None) == "image":
                    b, m = _decode_image_part(item)
                    if b:
                        return b, m
    return None, "image/png"


async def generate_text(
    prompt: str,
    *,
    system_instruction: str | None = None,
    thinking_level: str = "low",
    model: str = GEMINI_CHAT_MODEL,
) -> str:
    """Generate text asynchronously with `gemini-4-neon` via the Interactions API."""
    client = get_gemini_client()
    if not client:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    kwargs: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "store": False,
        "generation_config": {
            "thinking_level": thinking_level,
        },
    }
    if system_instruction:
        kwargs["system_instruction"] = system_instruction

    interaction = await _create_interaction_with_timeout(kwargs, timeout_sec=45.0)
    return extract_interaction_text(interaction)


async def describe_images_with_neon(
    prompt: str,
    images: list[tuple[bytes, str]],
    *,
    thinking_level: str = "low",
    model: str = GEMINI_CHAT_MODEL,
) -> str:
    """Inspect one or more images with `gemini-4-neon` and answer `prompt`."""
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img_bytes, mime_type in images[:6]:
        content.append(
            {
                "type": "image",
                "data": base64.b64encode(img_bytes).decode("ascii"),
                "mime_type": mime_type or "image/jpeg",
            }
        )

    interaction = await _create_interaction_with_timeout(
        {
            "model": model,
            "input": [{"type": "user_input", "content": content}],
            "store": False,
            "generation_config": {"thinking_level": thinking_level},
        },
        timeout_sec=60.0,
    )
    return extract_interaction_text(interaction)


async def generate_or_edit_image_bytes(
    prompt: str,
    *,
    reference_images: list[tuple[bytes, str]] | None = None,
    aspect_ratio: str | None = None,
    model: str = GEMINI_IMAGE_MODEL,
) -> dict[str, Any]:
    """
    Generate a new image or edit/combine up to 14 reference images using
    `gemini-nano-banana-2.1` via the GA Interactions API.
    """
    content_blocks: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img_bytes, mime_type in (reference_images or [])[:14]:
        content_blocks.append(
            {
                "type": "image",
                "data": base64.b64encode(img_bytes).decode("ascii"),
                "mime_type": mime_type or "image/jpeg",
            }
        )

    kwargs: dict[str, Any] = {
        "model": model,
        "input": content_blocks,
        "store": False,
    }
    if aspect_ratio and aspect_ratio in VALID_ASPECT_RATIOS:
        kwargs["response_format"] = {
            "type": "image",
            "aspect_ratio": aspect_ratio,
        }

    interaction = await _create_interaction_with_timeout(kwargs, timeout_sec=90.0)
    img_bytes, img_mime = extract_interaction_image(interaction)
    model_text = extract_interaction_text(interaction)
    return {
        "image_bytes": img_bytes,
        "mime_type": img_mime,
        "model_text": model_text,
    }


def _normalize_gemini_file_uri(uri: str | None) -> str | None:
    """Normalize any Files API URI to `https://generativelanguage.googleapis.com/files/{id}`."""
    if not uri:
        return None
    m = re.search(r"files/([a-zA-Z0-9_\-]+)", uri)
    if m:
        return f"https://generativelanguage.googleapis.com/files/{m.group(1)}"
    return uri.split("?", 1)[0]


async def _upload_bytes_to_files_api(
    client: genai.Client,
    raw_bytes: bytes,
    mime_type: str,
    display_name: str = "symm_asset",
) -> tuple[str, str, str]:
    """
    Upload in-memory media bytes via `client.aio.files.upload` and poll until `ACTIVE`.
    Returns `(normalized_uri, resolved_mime_type, file_resource_name)`.
    """
    ext_map = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
        "video/mp4": ".mp4",
    }
    ext = ext_map.get((mime_type or "").lower(), ".bin")

    with tempfile.TemporaryDirectory(prefix="symm_upload_") as tmpdir:
        tmp_path = os.path.join(tmpdir, f"{display_name}{ext}")
        with open(tmp_path, "wb") as f:
            f.write(raw_bytes)

        file_obj = await asyncio.wait_for(
            client.aio.files.upload(
                file=tmp_path,
                config=types.UploadFileConfig(
                    display_name=display_name,
                    mime_type=mime_type,
                ),
            ),
            timeout=120.0,
        )

    file_name = getattr(file_obj, "name", None) or ""
    interval = 2.0
    deadline = asyncio.get_running_loop().time() + 150.0

    while True:
        state = getattr(file_obj, "state", None)
        state_str = getattr(state, "name", None) or str(state or "")
        if state == types.FileState.ACTIVE or "ACTIVE" in state_str.upper():
            norm_uri = _normalize_gemini_file_uri(getattr(file_obj, "uri", None))
            resolved_mime = getattr(file_obj, "mime_type", None) or mime_type
            if not norm_uri:
                raise RuntimeError("Uploaded Gemini File is ACTIVE but missing URI.")
            return norm_uri, resolved_mime, file_name

        if state == types.FileState.FAILED or "FAILED" in state_str.upper():
            raise RuntimeError(f"Gemini Files API processing failed for {display_name}.")

        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(f"Timed out waiting for uploaded file {file_name} to become ACTIVE.")

        await asyncio.sleep(interval)
        interval = min(interval * 1.25, 8.0)
        file_obj = await asyncio.wait_for(
            client.aio.files.get(name=file_name),
            timeout=25.0,
        )


async def _extract_or_download_output_video(
    interaction: Any, client: genai.Client
) -> bytes | None:
    """Extract inline video bytes OR poll & download `output_video.uri` via `client.aio.files.download`."""
    vid_obj = getattr(interaction, "output_video", None)
    if not vid_obj:
        for step in reversed(getattr(interaction, "steps", None) or []):
            if getattr(step, "type", None) == "model_output":
                for item in reversed(getattr(step, "content", None) or []):
                    if getattr(item, "type", None) == "video":
                        vid_obj = item
                        break
            if vid_obj:
                break

    if not vid_obj:
        return None

    raw_data = getattr(vid_obj, "data", None)
    if raw_data:
        if isinstance(raw_data, (bytes, bytearray)):
            return bytes(raw_data)
        if isinstance(raw_data, str):
            return base64.b64decode(raw_data)

    vid_uri = getattr(vid_obj, "uri", None)
    if not vid_uri:
        return None

    m = re.search(r"files/([a-zA-Z0-9_\-]+)", vid_uri)
    if m:
        res_name = f"files/{m.group(1)}"
        for _ in range(30):
            try:
                f_info = await asyncio.wait_for(
                    client.aio.files.get(name=res_name), timeout=20.0
                )
                st = getattr(f_info, "state", None)
                st_str = getattr(st, "name", None) or str(st or "")
                if st == types.FileState.ACTIVE or "ACTIVE" in st_str.upper():
                    break
                if st == types.FileState.FAILED or "FAILED" in st_str.upper():
                    raise RuntimeError("Generated video file reported FAILED state.")
            except RuntimeError:
                raise
            except Exception:
                # Some signed URIs download directly without needing files.get polling
                break
            await asyncio.sleep(3.0)

    downloaded = await asyncio.wait_for(
        client.aio.files.download(file=vid_uri),
        timeout=150.0,
    )
    return bytes(downloaded) if downloaded else None


def _append_role_hint(text: str, hint: str) -> str:
    clean = (text or "").strip()
    if not clean:
        return hint
    sep = "" if clean[-1] in ".!?" else "."
    return f"{clean}{sep} {hint}"


async def generate_omni_video_bytes(
    prompt: str,
    *,
    aspect_ratio: str = "16:9",
    duration_sec: int = 8,
    resolution: str = "720p",
    first_frame: tuple[bytes, str] | None = None,
    last_frame: tuple[bytes, str] | None = None,
    loop_same_first_last: bool = False,
    reference_images: list[tuple[bytes, str]] | None = None,
    source_video: bytes | None = None,
    mode: str = "create",
    model: str = GEMINI_VIDEO_MODEL,
) -> dict[str, Any]:
    """
    Generate, edit, or extend a video with `gemini-omni-1.2-flash` via the GA Interactions API.
    Automatically uploads any input images/videos via the Gemini Files API, binds media roles
    using natural language, downloads the resulting MP4 bytes, and cleans up uploaded temp files.
    """
    client = get_video_gemini_client()
    if not client:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    norm_ar = aspect_ratio if aspect_ratio in VALID_VIDEO_ASPECT_RATIOS else "16:9"
    norm_res = (resolution or "720p").lower()
    if norm_res not in VALID_VIDEO_RESOLUTIONS:
        norm_res = "720p"

    has_two_keyframes = bool(first_frame and (last_frame or loop_same_first_last))
    max_dur = 10 if (has_two_keyframes or mode == "extend") else 15
    clamped_dur = max(3, min(max_dur, int(duration_sec or 8)))

    uploaded_file_names: list[str] = []
    try:
        input_parts: list[dict[str, Any]] = []

        # 1. Optional first frame (+ optional identical or distinct last frame)
        f1_uri = None
        f1_mime = None
        if first_frame:
            f1_uri, f1_mime, f1_name = await _upload_bytes_to_files_api(
                client, first_frame[0], first_frame[1] or "image/png", "first_frame"
            )
            if f1_name:
                uploaded_file_names.append(f1_name)
            input_parts.append({"type": "image", "uri": f1_uri, "mime_type": f1_mime})

            if loop_same_first_last:
                # Re-use uploaded URI for seamless start==end loop
                input_parts.append({"type": "image", "uri": f1_uri, "mime_type": f1_mime})
            elif last_frame:
                f2_uri, f2_mime, f2_name = await _upload_bytes_to_files_api(
                    client, last_frame[0], last_frame[1] or "image/png", "last_frame"
                )
                if f2_name:
                    uploaded_file_names.append(f2_name)
                input_parts.append({"type": "image", "uri": f2_uri, "mime_type": f2_mime})

        # 2. Optional reference images (up to 6 per Omni 1.2 Flash spec)
        ref_list = list(reference_images or [])[:6]
        for idx, (r_bytes, r_mime) in enumerate(ref_list):
            r_uri, r_resolved_mime, r_name = await _upload_bytes_to_files_api(
                client, r_bytes, r_mime or "image/png", f"ref_img_{idx + 1}"
            )
            if r_name:
                uploaded_file_names.append(r_name)
            input_parts.append({"type": "image", "uri": r_uri, "mime_type": r_resolved_mime})

        # 3. Optional source video for editing or extension
        if source_video:
            v_uri, v_mime, v_name = await _upload_bytes_to_files_api(
                client, source_video, "video/mp4", "source_video"
            )
            if v_name:
                uploaded_file_names.append(v_name)
            input_parts.append({"type": "video", "uri": v_uri, "mime_type": v_mime})

        # 4. Natural-language media role bindings (per Gemini Omni 1.2 Flash prompt spec)
        prompt_text = (prompt or "").strip()
        low_p = prompt_text.lower()
        if mode == "extend" and source_video:
            if not any(k in low_p for k in ("extend", "continue", "continuation", "keep going")):
                prompt_text = f"Extend this video. {prompt_text}".strip()
        elif first_frame and loop_same_first_last:
            if "starting frame" not in low_p and "first frame" not in low_p:
                first_ref = "the first image" if ref_list else "this image"
                prompt_text = _append_role_hint(
                    prompt_text,
                    f"Use {first_ref} as both the starting frame and the final frame.",
                )
        elif first_frame and last_frame:
            if "starting frame" not in low_p and "first frame" not in low_p:
                prompt_text = _append_role_hint(
                    prompt_text,
                    "Use the first image as the starting frame and the second image as the final frame.",
                )
        elif first_frame:
            if "starting frame" not in low_p and "first frame" not in low_p:
                first_ref = "the first image" if ref_list else "this image"
                prompt_text = _append_role_hint(
                    prompt_text,
                    f"Use {first_ref} as the starting frame.",
                )

        input_parts.append({"type": "text", "text": prompt_text})

        video_format: dict[str, Any] = {
            "type": "video",
            "delivery": "uri",
            "duration": f"{clamped_dur}s",
            "resolution": norm_res,
        }
        if mode != "extend":
            video_format["aspect_ratio"] = norm_ar

        req_kwargs: dict[str, Any] = {
            "model": model,
            "input": input_parts,
            "response_format": video_format,
        }

        try:
            interaction = await asyncio.wait_for(
                client.aio.interactions.create(**req_kwargs),
                timeout=360.0,
            )
        except asyncio.TimeoutError:
            logger.warning("Gemini Omni video generation timed out after 360s; resetting client.")
            reset_gemini_client()
            raise

        video_bytes = await _extract_or_download_output_video(interaction, client)
        model_text = extract_interaction_text(interaction)
        return {
            "video_bytes": video_bytes,
            "model_text": model_text,
            "interaction_id": getattr(interaction, "id", None),
            "duration_sec": clamped_dur,
            "aspect_ratio": norm_ar,
        }
    finally:
        # Best-effort cleanup of uploaded temporary input files
        for fname in uploaded_file_names:
            try:
                await asyncio.wait_for(client.aio.files.delete(name=fname), timeout=10.0)
            except Exception:
                pass


async def generate_json(
    prompt: str,
    schema: dict[str, Any],
    *,
    system_instruction: str | None = None,
    thinking_level: str = "medium",
    model: str = GEMINI_CHAT_MODEL,
) -> Any:
    """Generate schema-constrained structured JSON asynchronously with `gemini-4-neon`."""
    kwargs: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "store": False,
        "generation_config": {
            "thinking_level": thinking_level,
        },
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": schema,
        },
    }
    if system_instruction:
        kwargs["system_instruction"] = system_instruction

    interaction = await _create_interaction_with_timeout(kwargs, timeout_sec=75.0)
    raw_text = extract_interaction_text(interaction)
    if not raw_text:
        return None

    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return json.loads(cleaned.strip())


async def run_agentic_interaction(
    input_text: str,
    *,
    system_instruction: str,
    tools: list[dict[str, Any]],
    tool_executor: Callable[[str, dict[str, Any]], Awaitable[Any]],
    input_images: list[tuple[bytes, str]] | None = None,
    thinking_level: str = "medium",
    max_tool_rounds: int = 3,
    model: str = GEMINI_CHAT_MODEL,
) -> str:
    """
    Execute an async `gemini-4-neon` interaction supporting multimodal vision inputs
    and multi-step function calling in stateless mode (`store=False`).
    """
    client = get_gemini_client()
    if not client:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    user_content: list[dict[str, Any]] = [{"type": "text", "text": input_text}]
    for img_bytes, mime_type in (input_images or [])[:6]:
        user_content.append(
            {
                "type": "image",
                "data": base64.b64encode(img_bytes).decode("ascii"),
                "mime_type": mime_type or "image/jpeg",
            }
        )

    history: list[dict[str, Any]] = [
        {
            "type": "user_input",
            "content": user_content,
        }
    ]

    for _round in range(max_tool_rounds + 1):
        req_kwargs: dict[str, Any] = {
            "model": model,
            "system_instruction": system_instruction,
            "input": history,
            "store": False,
            "generation_config": {
                "thinking_level": thinking_level,
            },
        }
        if tools and _round < max_tool_rounds:
            req_kwargs["tools"] = tools

        interaction = await _create_interaction_with_timeout(req_kwargs, timeout_sec=75.0)
        steps = getattr(interaction, "steps", None) or []

        function_calls = [s for s in steps if getattr(s, "type", None) == "function_call"]
        if not function_calls or _round >= max_tool_rounds:
            return extract_interaction_text(interaction)

        for step in steps:
            if hasattr(step, "model_dump"):
                history.append(step.model_dump(exclude_none=True))
            elif isinstance(step, dict):
                history.append(step)

        for fc in function_calls:
            fn_name = getattr(fc, "name", "") or ""
            raw_args = getattr(fc, "arguments", None) or {}
            if isinstance(raw_args, str):
                try:
                    fn_args = json.loads(raw_args)
                except Exception:
                    fn_args = {}
            elif isinstance(raw_args, dict):
                fn_args = raw_args
            else:
                fn_args = {}

            try:
                tool_res = await tool_executor(fn_name, fn_args)
            except Exception as e:
                logger.warning("Tool %s execution error: %s", fn_name, e)
                tool_res = {"error": str(e)}

            res_str = (
                tool_res
                if isinstance(tool_res, str)
                else json.dumps(tool_res, ensure_ascii=False, default=str)
            )
            history.append(
                {
                    "type": "function_result",
                    "name": fn_name,
                    "call_id": getattr(fc, "id", ""),
                    "result": [{"type": "text", "text": res_str}],
                }
            )

    return ""


async def embed_texts(
    texts: list[str],
    *,
    model: str = GEMINI_EMBEDDING_MODEL,
    dimensionality: int = GEMINI_EMBEDDING_DIM,
) -> list[list[float] | None]:
    """Generate 768-dim embeddings via `gemini-embedding-2` (`embed_content`)."""
    if not texts:
        return []
    client = get_gemini_client()
    if not client:
        return [None] * len(texts)

    try:
        resp = await asyncio.wait_for(
            client.aio.models.embed_content(
                model=model,
                contents=texts,
                config=types.EmbedContentConfig(output_dimensionality=dimensionality),
            ),
            timeout=20.0,
        )
        embeddings = getattr(resp, "embeddings", None) or []
        results: list[list[float] | None] = []
        for idx in range(len(texts)):
            if idx < len(embeddings) and getattr(embeddings[idx], "values", None):
                vals = list(embeddings[idx].values)
                if len(vals) > dimensionality:
                    vals = vals[:dimensionality]
                results.append(vals)
            else:
                results.append(None)
        return results
    except Exception as e:
        logger.warning("Embedding generation failed: %s", e)
        return [None] * len(texts)
