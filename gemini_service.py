"""
Unified async Gemini API client wrapper for symm.city bot.
Uses:
  - `gemini-4-neon` via the GA Interactions API (`client.aio.interactions.create`)
  - `gemini-nano-banana-2.1` (Nano Banana 2.1) for contextual image generation & editing
  - `gemini-embedding-2` (`768` dimensions) via `client.aio.models.embed_content`
"""

import asyncio
import base64
import json
import logging
import os
from typing import Any, Awaitable, Callable

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-4-neon")
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-nano-banana-2.1")
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

_client: genai.Client | None = None
_client_key: str | None = None


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


def reset_gemini_client():
    """Discard cached `genai.Client` so subsequent calls establish fresh HTTP connections."""
    global _client
    _client = None


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
