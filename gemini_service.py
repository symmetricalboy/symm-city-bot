"""
Unified async Gemini API client wrapper for symm.city bot.
Uses `gemini-4-neon` via the GA Interactions API (`client.aio.interactions.create`)
and `gemini-embedding-2` (`768` dimensions) via `client.aio.models.embed_content`.
"""

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
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
GEMINI_EMBEDDING_DIM = 768

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
            _client = genai.Client(api_key=api_key)
            _client_key = api_key
        except Exception as e:
            logger.error("Failed to initialize Gemini client: %s", e)
            _client = None
    return _client


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

    interaction = await client.aio.interactions.create(**kwargs)
    return extract_interaction_text(interaction)


async def generate_json(
    prompt: str,
    schema: dict[str, Any],
    *,
    system_instruction: str | None = None,
    thinking_level: str = "medium",
    model: str = GEMINI_CHAT_MODEL,
) -> Any:
    """Generate schema-constrained structured JSON asynchronously with `gemini-4-neon`."""
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
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": schema,
        },
    }
    if system_instruction:
        kwargs["system_instruction"] = system_instruction

    interaction = await client.aio.interactions.create(**kwargs)
    raw_text = extract_interaction_text(interaction)
    if not raw_text:
        return None

    # Strip accidental markdown fences if present
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
    thinking_level: str = "medium",
    max_tool_rounds: int = 3,
    model: str = GEMINI_CHAT_MODEL,
) -> str:
    """
    Execute an async `gemini-4-neon` interaction supporting multi-step function calling
    in stateless mode (`store=False`).
    """
    client = get_gemini_client()
    if not client:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    history: list[dict[str, Any]] = [
        {
            "type": "user_input",
            "content": [{"type": "text", "text": input_text}],
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

        interaction = await client.aio.interactions.create(**req_kwargs)
        steps = getattr(interaction, "steps", None) or []

        function_calls = [s for s in steps if getattr(s, "type", None) == "function_call"]
        if not function_calls or _round >= max_tool_rounds:
            return extract_interaction_text(interaction)

        # Append all returned model steps (including thought + function_call steps) verbatim
        for step in steps:
            if hasattr(step, "model_dump"):
                history.append(step.model_dump(exclude_none=True))
            elif isinstance(step, dict):
                history.append(step)

        # Execute requested tool calls and append function_result steps
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
        resp = await client.aio.models.embed_content(
            model=model,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=dimensionality),
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
