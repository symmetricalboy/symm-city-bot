"""
Long-term Continuous Memory & Floating Context Engine for symm.city bot.
Inspired by Wristy's continuous memory architecture (memory_documents + memory_keyword_index
+ 768-dim gemini-embedding-2 pgvector embeddings + conversation archive search), expanded
for multi-user group chat dynamics, per-member living dossiers, and a 250-message floating window.
"""

import asyncio
import html
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from ai_library import get_relevant_context
from database import db
from gemini_service import (
    embed_texts,
    generate_json,
    get_gemini_client,
    run_agentic_interaction,
)
from handlers.cxp import _get_member_tag_string, calculate_level

logger = logging.getLogger(__name__)

FLOATING_WINDOW_MESSAGES = 250

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same",
    "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with", "would", "you",
    "your", "yours", "yourself", "yourselves", "im", "ive", "dont", "cant", "wont",
    "like", "yeah", "yes", "lol", "lmao", "haha", "also", "really", "think", "know",
}

VALID_CATEGORIES = {
    "USER_FACT",
    "INTEREST",
    "PREFERENCE_TASTE",
    "COMMUNICATION_STYLE",
    "SOCIAL_TIE",
    "EVENT_PAST",
    "EVENT_FUTURE",
    "COMMUNITY_LORE",
    "PROJECT_KNOWLEDGE",
    "DIRECTIVE",
}


class KeywordIndexer:
    """Multilingual tokenizer generating unigrams, bigrams, and trigrams for fast <5ms lookup."""

    TOKEN_RE = re.compile(r"[\w]{2,}", re.UNICODE)

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        if not text:
            return []
        cleaned = text.lower().replace("'", "").replace("’", "")
        raw_tokens = [t.strip("_") for t in cls.TOKEN_RE.findall(cleaned)]
        tokens = [t for t in raw_tokens if len(t) >= 2]

        result: list[str] = []
        seen: set[str] = set()

        def _add(tok: str):
            if tok and len(tok) <= 120 and tok not in seen:
                seen.add(tok)
                result.append(tok)

        for t in tokens:
            if t not in STOP_WORDS:
                _add(t)

        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i + 1]
            if w1 not in STOP_WORDS or w2 not in STOP_WORDS:
                _add(f"{w1} {w2}")
            if i < len(tokens) - 2:
                w3 = tokens[i + 2]
                if w1 not in STOP_WORDS and w3 not in STOP_WORDS:
                    _add(f"{w1} {w2} {w3}")

        return result

    @classmethod
    def build_document_keywords(
        cls,
        *,
        title: str,
        summary: str,
        content: str,
        category: str,
        subject_name: str | None = None,
        extra_keywords: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        weights: dict[str, float] = {}

        def _put(phrase: str, w: float):
            norm = phrase.strip().lower().lstrip("@")[:120]
            if norm and len(norm) >= 2 and norm not in STOP_WORDS:
                weights[norm] = max(weights.get(norm, 0.0), w)

        for kw in extra_keywords or []:
            _put(kw, 3.8)
            for sub in cls.tokenize(kw):
                _put(sub, 3.2)

        if subject_name:
            _put(subject_name, 3.5)
            for sub in cls.tokenize(subject_name):
                _put(sub, 3.0)

        for t in cls.tokenize(title):
            _put(t, 3.2)

        _put(category.lower(), 1.8)

        for t in cls.tokenize(summary)[:35]:
            _put(t, 1.8)

        for t in cls.tokenize(content)[:45]:
            _put(t, 1.1)

        return list(weights.items())


# ---------------------------------------------------------------------------
# Structured JSON Schema for Gemini 4 Neon Chat Batch Digestion
# ---------------------------------------------------------------------------
DIGESTION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": ["USER", "COMMUNITY"],
                    },
                    "subject_user_id": {
                        "type": "integer",
                        "description": "Numeric Telegram user_id when scope is USER; 0 when scope is COMMUNITY.",
                    },
                    "subject_name": {
                        "type": "string",
                        "description": "Display name of the member if USER scope, or 'symm.city' if COMMUNITY.",
                    },
                    "category": {
                        "type": "string",
                        "enum": sorted(VALID_CATEGORIES),
                    },
                    "title": {
                        "type": "string",
                        "description": "Concise canonical title (max 80 chars) suitable for deduplication/updating.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "1-sentence high-signal summary of the fact or memory.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Detailed 1-3 sentence context including specifics, models/names, dates, or quotes.",
                    },
                    "importance_score": {
                        "type": "number",
                        "description": "1.0 (minor) to 10.0 (core identity, major lore, or explicit directive).",
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "scope",
                    "subject_user_id",
                    "subject_name",
                    "category",
                    "title",
                    "summary",
                    "content",
                    "importance_score",
                    "keywords",
                ],
            },
        },
        "profile_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer"},
                    "canonical_name": {"type": "string"},
                    "username": {"type": "string"},
                    "bio_summary": {
                        "type": "string",
                        "description": "Updated 1-3 sentence biographical summary merging known & newly learned facts.",
                    },
                    "personality_and_style": {
                        "type": "string",
                        "description": "How this person talks, jokes, and vibes in chat.",
                    },
                    "new_interests": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "new_facts": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "user_id",
                    "canonical_name",
                    "bio_summary",
                    "personality_and_style",
                    "new_interests",
                    "new_facts",
                ],
            },
        },
    },
    "required": ["memories", "profile_updates"],
}

DIGESTION_SYSTEM_PROMPT = """You are the long-term memory consolidation engine for the **symm.city** Telegram group chat (@symmcity / symm.city), founded by **symmetricalboy** (symm / Dylan).

Analyze the chronological slice of group chat messages and extract durable, high-value memories about:
1. **Individual Community Members (`scope = "USER"`)** — use the exact numeric `user_id` shown in the speaker tag:
   - `USER_FACT`: Real name, age/birth year, country/city/timezone, languages spoken, job/studies, family/partner/pets, hardware/devices (phones, smartwatches, guitars, consoles, PCs), projects they build.
   - `INTEREST`: Games, music, programming languages, AI models/tools, crypto/TON, hobbies, philosophical topics.
   - `PREFERENCE_TASTE`: Strong likes, pet peeves/hatreds (e.g. hating Apple/ads/emails), favorite foods/media.
   - `COMMUNICATION_STYLE`: Distinct slang, humor style (dry, chaotic, sarcastic, multilingual), banter habits.
   - `SOCIAL_TIE`: Friendships, rivalries, playful feuds, or collaborations with other group members.
   - `EVENT_PAST` / `EVENT_FUTURE`: Memorable life updates, launches, exams, trips, or milestones.
2. **Shared Community Lore & Projects (`scope = "COMMUNITY"`, `subject_user_id = 0`)**:
   - `COMMUNITY_LORE`: Inside jokes, legendary CXP gambling wins/wipeouts, memorable group debates, funny running gags.
   - `PROJECT_KNOWLEDGE`: Facts discussed about Symmetris, Wristy, Ask Everything, Comail, FlappySky, symm.city bot features, or member projects.
   - `DIRECTIVE`: Explicit group instructions for how the bot should behave or remember things.

RULES:
- Ignore trivial small talk ("ok", "lol", single emojis) unless part of a notable joke or fact.
- NEVER confuse speakers! Attribute facts strictly to the exact `user_id` who stated or owns them.
- Do NOT extract facts from bot messages (`[BOT]`) unless capturing a legendary community event (such as a massive CXP slot win/wipeout).
- Give canonical, reusable `title`s (e.g. "SnotGoblin's Ibanez RGA42FM Guitar", "Amir's Life & Languages in Iran") so updates merge cleanly.
- If a slice contains nothing worth remembering long-term, return empty arrays `{"memories": [], "profile_updates": []}`.
"""


# ---------------------------------------------------------------------------
# Tool Declarations for `/ask` Agentic Retrieval & Live Memory Writing
# ---------------------------------------------------------------------------
ASK_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "search_chat_history",
        "description": (
            "Search the complete historical & live message archive of the symm.city Telegram group chat "
            "(all 10,000+ messages since Feb 2025). Call this whenever a question asks about something said earlier "
            "beyond the recent 250-message window, exact quotes, earliest/first messages, specific dates, or "
            "past discussions about a topic or member."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords or phrase to search for (e.g. 'Ibanez', 'Costco', 'first watchface', '300k'). Leave empty to browse by user/date/order.",
                },
                "username": {
                    "type": "string",
                    "description": "Optional member display name or @username to filter messages by author (e.g. 'Zeno', 'SnotGoblin', 'Julia', 'Kostis', 'symmetricalboy').",
                },
                "order": {
                    "type": "string",
                    "enum": ["newest", "oldest"],
                    "description": "Use 'oldest' for earliest/first messages or origins; 'newest' for most recent mentions.",
                },
                "time_range": {
                    "type": "string",
                    "description": "Optional filter: 'today', 'yesterday', 'week', 'month', 'all_time', or 'YYYY-MM' / 'YYYY-MM-DD'.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max matches to return (default 15, max 30).",
                },
            },
        },
    },
    {
        "type": "function",
        "name": "get_member_memories",
        "description": (
            "Look up the full learned profile dossier and stored long-term memories for a specific "
            "symm.city group member by their name, @username, or nickname."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "member": {
                    "type": "string",
                    "description": "Member name, @username, or alias (e.g. 'SnotGoblin', '@Hexaspells', 'Jason', 'Zeno', 'Amir', 'Kostis', 'Julia', 'Ewan').",
                }
            },
            "required": ["member"],
        },
    },
    {
        "type": "function",
        "name": "remember_fact",
        "description": (
            "Immediately store a new durable memory about a user or the symm.city community into long-term memory. "
            "Call this when a user shares a new personal detail during /ask or explicitly asks you to remember something."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "enum": ["USER", "COMMUNITY"],
                },
                "category": {
                    "type": "string",
                    "enum": sorted(VALID_CATEGORIES),
                },
                "subject_name": {
                    "type": "string",
                    "description": "Member's name/@username (for USER scope) or 'symm.city' (for COMMUNITY scope).",
                },
                "title": {
                    "type": "string",
                    "description": "Concise canonical title for this memory.",
                },
                "summary": {
                    "type": "string",
                    "description": "1-sentence summary of what to remember.",
                },
                "content": {
                    "type": "string",
                    "description": "Full details and context.",
                },
                "importance_score": {
                    "type": "number",
                    "description": "Importance from 1.0 to 10.0.",
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "scope",
                "category",
                "subject_name",
                "title",
                "summary",
                "content",
            ],
        },
    },
]


def _format_profile_block(prof: dict, memories: list[dict] | None = None) -> str:
    cname = prof.get("canonical_name") or f"User {prof.get('user_id')}"
    uname = f" (@{prof['username']})" if prof.get("username") else ""
    raw_aliases = prof.get("known_aliases")
    aliases = (
        json.loads(raw_aliases)
        if isinstance(raw_aliases, str)
        else (raw_aliases or [])
    )
    alias_str = f" | Aliases: {', '.join(aliases[:5])}" if aliases else ""

    raw_int = prof.get("interests")
    interests = (
        json.loads(raw_int) if isinstance(raw_int, str) else (raw_int or [])
    )
    raw_facts = prof.get("notable_facts")
    facts = (
        json.loads(raw_facts) if isinstance(raw_facts, str) else (raw_facts or [])
    )

    lines = [f"• {cname}{uname} [id:{prof.get('user_id')}]{alias_str}"]
    if prof.get("bio_summary"):
        lines.append(f"  Bio: {prof['bio_summary']}")
    if prof.get("personality_and_style"):
        lines.append(f"  Vibe/Style: {prof['personality_and_style']}")
    if interests:
        lines.append(f"  Interests: {', '.join(str(x) for x in interests[:12])}")
    if facts:
        lines.append(f"  Key Facts: {' | '.join(str(x) for x in facts[:10])}")
    if memories:
        for m in memories[:8]:
            lines.append(f"  - [{m['category']}] {m['title']}: {m['summary']}")
    return "\n".join(lines)


def _format_floating_window(messages: list[dict]) -> str:
    if not messages:
        return "(No recent group messages recorded yet.)"

    lines: list[str] = []
    for m in messages:
        dt = m.get("created_at")
        ts_str = (
            dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
            if isinstance(dt, datetime)
            else ""
        )
        mid = m.get("message_id")
        reply_id = m.get("reply_to_message_id")
        reply_tag = f" ->#{reply_id}" if reply_id else ""
        bot_tag = "[BOT] " if m.get("is_bot") else ""
        sname = m.get("sender_name") or (
            f"@{m['sender_username']}" if m.get("sender_username") else f"User {m.get('user_id')}"
        )
        uname_part = (
            f" (@{m['sender_username']})"
            if m.get("sender_username")
            and m["sender_username"].lower() not in sname.lower()
            else ""
        )
        text = (m.get("content") or "").replace("\n", " ⏎ ").strip()
        if len(text) > 420:
            text = text[:417] + "..."
        lines.append(
            f"[{ts_str} #{mid}{reply_tag}] {bot_tag}{sname}{uname_part}: {text}"
        )
    return "\n".join(lines)


def sanitize_telegram_html(raw: str) -> str:
    """
    Clean model output so Telegram's strict HTML parser accepts it reliably:
    strips markdown code fences around the whole reply, converts stray markdown bold
    if no HTML tags were used, and preserves only Telegram-supported HTML tags.
    """
    text = (raw or "").strip()
    if text.startswith("```html"):
        text = text[7:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

    # Replace unsupported block/list HTML tags with clean Telegram equivalents
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*/?\s*p\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*li\s*>", "• ", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*/\s*li\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*/?\s*(ul|ol|div|h[1-6])\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


class MemoryEngine:
    """Orchestrates `/ask` context + tools and background continuous chat memory digestion."""

    def __init__(self):
        self._worker_task: asyncio.Task | None = None
        self._wake_event = asyncio.Event()
        self._stop_requested = False
        self._digest_lock = asyncio.Lock()

    def notify_new_message(self):
        """Signal the background worker that new chat activity arrived."""
        self._wake_event.set()

    def start_background_worker(self):
        if self._worker_task is None or self._worker_task.done():
            self._stop_requested = False
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info("MemoryEngine background worker started.")

    async def stop_background_worker(self):
        self._stop_requested = True
        self._wake_event.set()
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        self._worker_task = None

    async def _worker_loop(self):
        # Give bot startup a few seconds to settle
        await asyncio.sleep(5)

        while not self._stop_requested:
            try:
                if get_gemini_client() and db.pool:
                    # 1. Backfill any missing pgvector embeddings on memory_documents
                    embedded_n = await self.backfill_missing_embeddings(batch_size=32)
                    if embedded_n > 0:
                        await asyncio.sleep(1.5)
                        continue

                    # 2. Check if main_group_id has undigested messages ready to consolidate
                    config = await db.get_config()
                    main_group_id = config.get("main_group_id") if config else None
                    if main_group_id:
                        stats = await db.count_undigested_messages(main_group_id)
                        cnt = stats["count"]
                        oldest_ts = stats["oldest_ts"]
                        age_secs = (
                            (datetime.now(timezone.utc) - oldest_ts).total_seconds()
                            if oldest_ts
                            else 0
                        )
                        if cnt >= 25 or (cnt >= 6 and age_secs >= 900):
                            ok = await self.digest_next_batch(
                                main_group_id, batch_size=80
                            )
                            if ok:
                                # Keep chewing through backlog steadily without rate-limiting
                                await asyncio.sleep(3.0)
                                continue
                            else:
                                # Back off on transient API/auth error
                                await asyncio.sleep(180.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("MemoryEngine background loop error: %s", e)
                await asyncio.sleep(60.0)

            self._wake_event.clear()
            try:
                await asyncio.wait_for(self._wake_event.wait(), timeout=60.0)
            except asyncio.TimeoutError:
                pass

    async def backfill_missing_embeddings(self, batch_size: int = 32) -> int:
        """Embed any `memory_documents` rows whose `embedding` is currently NULL."""
        if not get_gemini_client() or not db.has_pgvector:
            return 0
        docs = await db.get_unembedded_memory_documents(limit=batch_size)
        if not docs:
            return 0

        texts = [
            f"[{d['scope']} | {d.get('subject_name') or 'symm.city'} | {d['category']}] "
            f"{d['title']}: {d['summary']} {d['content']}"
            for d in docs
        ]
        vectors = await embed_texts(texts)
        updated = 0
        for doc, vec in zip(docs, vectors):
            if vec:
                await db.update_memory_document_embedding(int(doc["id"]), vec)
                updated += 1
        if updated:
            logger.info("Embedded %d memory_documents with gemini-embedding-2.", updated)
        return updated

    async def digest_next_batch(self, chat_id: int, batch_size: int = 80) -> bool:
        """
        Digest the oldest undigested batch of messages in `group_chat_messages` using `gemini-4-neon`,
        storing extracted `memory_documents`, `memory_keyword_index`, `pgvector` embeddings, and
        updated `user_memory_profiles`.
        """
        if not get_gemini_client():
            return False

        async with self._digest_lock:
            batch = await db.get_undigested_message_batch(chat_id, batch_size=batch_size)
            if not batch:
                return False

            all_msg_ids = [int(m["message_id"]) for m in batch]

            # Filter out noisy automated game/cooldown lines from the LLM prompt while still advancing cursor
            meaningful = [
                m
                for m in batch
                if not (
                    m.get("is_bot")
                    and (m.get("content") or "").startswith(
                        ("🎰", "⏳", "📊 Statistics", "🏆 ", "⚠️ Unlinked")
                    )
                )
                and not (
                    re.match(
                        r"^/(gamble|g|bet|slots|slot|spin|steal|s|rob|yoink|heist|mug|level|lvl|xp|cxp|stats|rank|bal|balance|me|leaderboard|lb|top|ranks|leaders|history|h|hist|chart|charts|graph|ledger|pnl|ping|status|latency|pong)(\s|$|@)",
                        (m.get("content") or "").strip(),
                        flags=re.IGNORECASE,
                    )
                )
            ]

            if len(meaningful) < 2:
                await db.mark_messages_digested(chat_id, all_msg_ids)
                return True

            speaker_ids = sorted(
                {
                    int(m["user_id"])
                    for m in meaningful
                    if m.get("user_id") and not m.get("is_bot") and int(m["user_id"]) > 0
                }
            )
            existing_profiles = await db.get_user_memory_profiles_by_ids(speaker_ids)

            profiles_context_lines = []
            for uid in speaker_ids:
                if uid in existing_profiles:
                    profiles_context_lines.append(_format_profile_block(existing_profiles[uid]))
            profiles_block = (
                "\n".join(profiles_context_lines)
                if profiles_context_lines
                else "(No prior profiles recorded for these speakers yet.)"
            )

            transcript_lines = []
            for m in meaningful:
                dt = m.get("created_at")
                ts_str = (
                    dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
                    if isinstance(dt, datetime)
                    else ""
                )
                uid = m.get("user_id") or 0
                sname = m.get("sender_name") or f"User {uid}"
                uname = f" @{m['sender_username']}" if m.get("sender_username") else ""
                role_prefix = "[BOT]" if m.get("is_bot") else f"[user_id={uid}]"
                reply_part = (
                    f" (replying to #{m['reply_to_message_id']})"
                    if m.get("reply_to_message_id")
                    else ""
                )
                txt = (m.get("content") or "").replace("\n", " ").strip()
                transcript_lines.append(
                    f"[{ts_str} #{m['message_id']}]{reply_part} {role_prefix} {sname}{uname}: {txt}"
                )

            prompt = (
                f"EXISTING KNOWN PROFILES FOR SPEAKERS IN THIS BATCH:\n{profiles_block}\n\n"
                f"CHRONOLOGICAL CHAT TRANSCRIPT ({len(meaningful)} messages, IDs #{all_msg_ids[0]}..#{all_msg_ids[-1]}):\n"
                + "\n".join(transcript_lines)
            )

            try:
                parsed = await generate_json(
                    prompt,
                    DIGESTION_RESPONSE_SCHEMA,
                    system_instruction=DIGESTION_SYSTEM_PROMPT,
                    thinking_level="medium",
                )
            except Exception as e:
                logger.warning("Memory digestion Gemini call failed: %s", e)
                return False

            if isinstance(parsed, dict):
                raw_memories = parsed.get("memories") or []
                profile_updates = parsed.get("profile_updates") or []

                # Batch-embed all extracted memories in one call
                emb_inputs = [
                    f"[{m.get('scope', 'USER')} | {m.get('subject_name', 'symm.city')} | {m.get('category', 'USER_FACT')}] "
                    f"{m.get('title', '')}: {m.get('summary', '')} {m.get('content', '')}"
                    for m in raw_memories
                ]
                vectors = await embed_texts(emb_inputs) if emb_inputs else []

                for idx, mem in enumerate(raw_memories):
                    cat = mem.get("category") or "USER_FACT"
                    if cat not in VALID_CATEGORIES:
                        cat = "USER_FACT"
                    scope = (mem.get("scope") or "USER").upper()
                    subj_id = int(mem.get("subject_user_id") or 0)
                    if scope == "COMMUNITY" or subj_id <= 0:
                        scope = "COMMUNITY"
                        subj_id_val = None
                    else:
                        subj_id_val = subj_id

                    title = (mem.get("title") or "").strip()
                    summary = (mem.get("summary") or "").strip()
                    content = (mem.get("content") or summary).strip()
                    subj_name = (mem.get("subject_name") or "").strip() or None
                    if not title or not summary:
                        continue

                    kws = KeywordIndexer.build_document_keywords(
                        title=title,
                        summary=summary,
                        content=content,
                        category=cat,
                        subject_name=subj_name,
                        extra_keywords=mem.get("keywords") or [],
                    )
                    vec = vectors[idx] if idx < len(vectors) else None

                    await db.upsert_memory_document(
                        scope=scope,
                        category=cat,
                        title=title,
                        summary=summary,
                        content=content,
                        subject_user_id=subj_id_val,
                        subject_name=subj_name,
                        importance_score=float(mem.get("importance_score") or 5.5),
                        source_start_msg_id=all_msg_ids[0],
                        source_end_msg_id=all_msg_ids[-1],
                        embedding=vec,
                        keywords=kws,
                    )

                for pu in profile_updates:
                    uid = int(pu.get("user_id") or 0)
                    if uid <= 0:
                        continue
                    await db.upsert_user_memory_profile(
                        user_id=uid,
                        canonical_name=pu.get("canonical_name") or f"User {uid}",
                        username=pu.get("username"),
                        bio_summary=pu.get("bio_summary"),
                        personality_and_style=pu.get("personality_and_style"),
                        interests=pu.get("new_interests") or [],
                        notable_facts=pu.get("new_facts") or [],
                    )

            await db.mark_messages_digested(chat_id, all_msg_ids)
            logger.info(
                "Digested chat batch #%d..#%d (%d msgs).",
                all_msg_ids[0],
                all_msg_ids[-1],
                len(all_msg_ids),
            )
            return True

    async def answer_ask_query(
        self,
        *,
        chat_id: int,
        asker_id: int,
        asker_name: str,
        asker_username: str | None,
        question: str,
        reply_context: dict | None = None,
    ) -> str:
        """
        Generate a context-rich, memory-grounded answer for `/ask` using `gemini-4-neon`,
        injecting:
          - 250-message floating group chat context window
          - Learned user profiles & long-term memories (hybrid keyword + pgvector retrieval)
          - Bluesky archive & symm.city core knowledge
          - Live DB search & memory tools (`search_chat_history`, `get_member_memories`, `remember_fact`)
        """
        # Clean accidental literal "<question>" copied from /help examples
        clean_q = re.sub(r"^\s*<question>\s*[.:,-]?\s*", "", question.strip(), flags=re.IGNORECASE).strip()
        if not clean_q:
            clean_q = question.strip()

        # 1. Core knowledge + Bluesky post archive BM25 context
        core_and_bluesky = get_relevant_context(clean_q)

        # 2. Fetch floating context window of the 250 most recent conversational messages
        recent_250 = await db.get_recent_group_messages(
            chat_id=chat_id,
            limit=FLOATING_WINDOW_MESSAGES,
            exclude_bot_noise=True,
        )

        # 3. Identify focused users (the asker + replied-to author + any members mentioned in question)
        all_profiles = await db.get_all_user_memory_profiles(limit=40)
        profiles_by_id = {int(p["user_id"]): p for p in all_profiles}

        focus_user_ids: list[int] = [asker_id]
        if (
            reply_context
            and reply_context.get("author_id")
            and not reply_context.get("is_self_bot")
        ):
            rid = int(reply_context["author_id"])
            if rid not in focus_user_ids:
                focus_user_ids.append(rid)

        mention_haystack = clean_q.lower()
        if reply_context and reply_context.get("text"):
            mention_haystack += " " + str(reply_context["text"][:400]).lower()

        for prof in all_profiles:
            uid = int(prof["user_id"])
            if uid in focus_user_ids:
                continue
            cname = (prof.get("canonical_name") or "").lower()
            uname = (prof.get("username") or "").lower()
            raw_al = prof.get("known_aliases")
            aliases = (
                json.loads(raw_al) if isinstance(raw_al, str) else (raw_al or [])
            )
            candidates = [c for c in [cname, uname, *[str(a).lower() for a in aliases]] if len(c) >= 3]
            if any(c in mention_haystack for c in candidates):
                focus_user_ids.append(uid)

        # 4. Hybrid memory retrieval (keyword inverted index + gemini-embedding-2 pgvector)
        search_seed = clean_q
        if reply_context and reply_context.get("text"):
            search_seed = f"{clean_q} {reply_context['text'][:200]}"
        query_keywords = KeywordIndexer.tokenize(search_seed)

        query_vec_list = await embed_texts([search_seed])
        query_vec = query_vec_list[0] if query_vec_list else None

        matched_memories = await db.retrieve_hybrid_memories(
            query_keywords=query_keywords,
            query_embedding=query_vec,
            focus_user_ids=focus_user_ids,
            limit=22,
        )

        # 5. Live asker stats (CXP / level / location)
        asker_db_user = await db.get_user(asker_id)
        asker_cxp = int(asker_db_user.get("cxp") or 0) if asker_db_user else 0
        asker_lvl = calculate_level(asker_cxp)
        asker_tag = _get_member_tag_string(asker_lvl)
        asker_loc_raw = asker_db_user.get("location") if asker_db_user else None
        asker_loc = ""
        if asker_loc_raw:
            try:
                loc_obj = json.loads(asker_loc_raw)
                asker_loc = f" | Location: {loc_obj.get('location_name')} ({loc_obj.get('timezone')})"
            except Exception:
                asker_loc = f" | Location: {asker_loc_raw}"

        # Build Focused Member Profiles section
        focused_blocks: list[str] = []
        for f_uid in focus_user_ids[:5]:
            prof = profiles_by_id.get(f_uid)
            u_mems = await db.get_memories_for_user(f_uid, limit=10)
            if prof:
                focused_blocks.append(_format_profile_block(prof, u_mems))
            elif u_mems:
                lines = [f"• User {f_uid}:"] + [
                    f"  - [{m['category']}] {m['title']}: {m['summary']}" for m in u_mems
                ]
                focused_blocks.append("\n".join(lines))

        # Build Active Community Roster (compact 1-liners for speakers in the 250-message window)
        window_speaker_ids = []
        seen_sp: set[int] = set()
        for m in reversed(recent_250):
            uid = m.get("user_id")
            if uid and not m.get("is_bot") and int(uid) not in seen_sp and int(uid) not in focus_user_ids:
                seen_sp.add(int(uid))
                window_speaker_ids.append(int(uid))

        roster_lines: list[str] = []
        for sp_id in window_speaker_ids[:15]:
            p = profiles_by_id.get(sp_id)
            if p:
                un = f" (@{p['username']})" if p.get("username") else ""
                bio = (p.get("bio_summary") or "")[:180]
                roster_lines.append(f"• {p['canonical_name']}{un}: {bio}")

        # Format retrieved long-term memory documents
        memory_lines: list[str] = []
        for m in matched_memories:
            who = (
                f"[Member: {m.get('subject_name') or m.get('subject_user_id')}]"
                if m.get("scope") == "USER"
                else "[Community Lore]"
            )
            memory_lines.append(
                f"• {who} ({m['category']}) {m['title']}: {m['summary']} — {m['content']}"
            )

        floating_transcript = _format_floating_window(recent_250)

        # Define live tool executor bound to this chat & asker
        async def _execute_tool(fn_name: str, fn_args: dict[str, Any]) -> Any:
            if fn_name == "search_chat_history":
                hits = await db.search_group_chat_messages(
                    chat_id=chat_id,
                    query_str=fn_args.get("query"),
                    user_filter=fn_args.get("username"),
                    order=fn_args.get("order") or "newest",
                    time_range=fn_args.get("time_range"),
                    limit=int(fn_args.get("limit") or 15),
                )
                if not hits:
                    return {"match_count": 0, "messages": []}
                return {
                    "match_count": len(hits),
                    "messages": [
                        {
                            "message_id": h["message_id"],
                            "timestamp_utc": (
                                h["created_at"].astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
                                if isinstance(h.get("created_at"), datetime)
                                else str(h.get("created_at"))
                            ),
                            "sender": h.get("sender_name") or h.get("sender_username") or str(h.get("user_id")),
                            "username": h.get("sender_username"),
                            "is_bot": bool(h.get("is_bot")),
                            "text": h.get("content"),
                        }
                        for h in hits
                    ],
                }

            if fn_name == "get_member_memories":
                q_member = (fn_args.get("member") or "").strip()
                profs = await db.find_member_profiles(q_member)
                if not profs:
                    u_row = await db.get_user_by_username(q_member)
                    if u_row:
                        uid = int(u_row["user_id"])
                        mems = await db.get_memories_for_user(uid, limit=15)
                        return {
                            "found": True,
                            "user": {
                                "user_id": uid,
                                "display_name": u_row.get("display_name"),
                                "username": u_row.get("username"),
                                "cxp": u_row.get("cxp"),
                            },
                            "memories": [
                                {
                                    "category": m["category"],
                                    "title": m["title"],
                                    "summary": m["summary"],
                                    "content": m["content"],
                                }
                                for m in mems
                            ],
                        }
                    return {"found": False, "message": f"No member matching '{q_member}' found."}

                results = []
                for p in profs[:3]:
                    uid = int(p["user_id"])
                    mems = await db.get_memories_for_user(uid, limit=15)
                    results.append(
                        {
                            "user_id": uid,
                            "canonical_name": p.get("canonical_name"),
                            "username": p.get("username"),
                            "known_aliases": p.get("known_aliases"),
                            "bio_summary": p.get("bio_summary"),
                            "personality_and_style": p.get("personality_and_style"),
                            "interests": p.get("interests"),
                            "notable_facts": p.get("notable_facts"),
                            "memories": [
                                {
                                    "category": m["category"],
                                    "title": m["title"],
                                    "summary": m["summary"],
                                    "content": m["content"],
                                }
                                for m in mems
                            ],
                        }
                    )
                return {"found": True, "members": results}

            if fn_name == "remember_fact":
                scope = (fn_args.get("scope") or "USER").upper()
                cat = fn_args.get("category") or "USER_FACT"
                if cat not in VALID_CATEGORIES:
                    cat = "USER_FACT"
                subj_name = (fn_args.get("subject_name") or "").strip()
                subj_id = None
                if scope == "USER":
                    profs = await db.find_member_profiles(subj_name)
                    if profs:
                        subj_id = int(profs[0]["user_id"])
                        subj_name = profs[0]["canonical_name"]
                    else:
                        subj_id = asker_id
                        subj_name = asker_name

                title = (fn_args.get("title") or "").strip()
                summary = (fn_args.get("summary") or "").strip()
                content = (fn_args.get("content") or summary).strip()
                kws = KeywordIndexer.build_document_keywords(
                    title=title,
                    summary=summary,
                    content=content,
                    category=cat,
                    subject_name=subj_name,
                    extra_keywords=fn_args.get("keywords") or [],
                )
                vecs = await embed_texts(
                    [f"[{scope} | {subj_name} | {cat}] {title}: {summary} {content}"]
                )
                doc_id = await db.upsert_memory_document(
                    scope=scope,
                    category=cat,
                    title=title,
                    summary=summary,
                    content=content,
                    subject_user_id=subj_id,
                    subject_name=subj_name,
                    importance_score=float(fn_args.get("importance_score") or 7.0),
                    embedding=vecs[0] if vecs else None,
                    keywords=kws,
                )
                return {"saved": bool(doc_id), "document_id": doc_id}

            return {"error": f"Unknown tool: {fn_name}"}

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        system_instruction = f"""You are <b>symm.city bot</b>, the resident AI companion and community brain for the <b>symm.city</b> Telegram group chat (<code>@symmcity</code> / https://symm.city), created by <b>symm</b> (<code>@symmetricalboy</code> / Dylan).
Current Time: {now_str}

PERSONA & VIBE:
- Warm, sharp, witty, grounded, and genuinely part of the @symmcity hangout — like a longtime regular who knows everyone in the group, remembers their stories, projects, gear, hot takes, and inside jokes.
- Never robotic or sycophantic. Match the playful, candid, eclectic energy of symm.city while staying helpful, accurate, and kind.
- Never fabricate personal facts about members or symm. Rely on your Long-Term Memories, Member Profiles, Floating 250-Message Chat Window, Bluesky Archive, and your `search_chat_history` / `get_member_memories` tools!
- Whenever someone asks about an older conversation, specific quote, first message, or a member detail not already shown below, proactively call `search_chat_history` or `get_member_memories` before answering.
- Whenever someone shares a new durable personal fact or asks you to remember something, call `remember_fact` so it persists forever.

=== CORE SYMM.CITY KNOWLEDGE ===
{core_and_bluesky}

=== FOCUSED MEMBER PROFILES ===
{chr(10).join(focused_blocks) if focused_blocks else "(No prior profile stored yet.)"}

=== OTHER ACTIVE MEMBERS IN RECENT CHAT ===
{chr(10).join(roster_lines) if roster_lines else "(None)"}

=== RETRIEVED LONG-TERM MEMORIES & LORE ===
{chr(10).join(memory_lines) if memory_lines else "(No specific long-term memory documents matched yet.)"}

=== FLOATING CONTEXT WINDOW (MOST RECENT {len(recent_250)} GROUP MESSAGES, OLDEST -> NEWEST) ===
{floating_transcript}

OUTPUT FORMATTING RULES (STRICT):
1. Reply in the SAME language that the asking user used in their question.
2. Keep responses concise, natural, and engaging (MUST NEVER exceed 3,500 characters).
3. Format strictly using Telegram-supported HTML tags ONLY:
   <code>&lt;b&gt;</code>, <code>&lt;i&gt;</code>, <code>&lt;u&gt;</code>, <code>&lt;s&gt;</code>, <code>&lt;code&gt;</code>, <code>&lt;pre&gt;</code>, <code>&lt;blockquote&gt;</code>, <code>&lt;tg-spoiler&gt;</code>, <code>&lt;a href="..."&gt;</code>.
   NEVER use Markdown (`**bold**`, `*italic*`, `### headers`, or ``` fences) and escape any literal `&`, `<`, `>` that are not one of the allowed HTML tags above!"""

        asker_handle = f" (@{asker_username})" if asker_username else ""
        reply_block = ""
        if reply_context and reply_context.get("text"):
            if reply_context.get("is_self_bot"):
                reply_block = (
                    f"\n[Directly replying to YOUR (symm.city bot's) message #{reply_context.get('message_id')}: "
                    f"\"{reply_context['text']}\"]\n"
                )
            else:
                reply_block = (
                    f"\n[Directly replying to message #{reply_context.get('message_id')} "
                    f"from {reply_context.get('author_name')}: \"{reply_context['text']}\"]\n"
                )

        user_prompt = (
            f"Asking Member: {asker_name}{asker_handle} [user_id={asker_id} | Level {asker_lvl} ({asker_tag}) | {asker_cxp:,} CXP{asker_loc}]"
            f"{reply_block}\n"
            f"Question / Prompt: {clean_q}"
        )

        raw_answer = await run_agentic_interaction(
            user_prompt,
            system_instruction=system_instruction,
            tools=ASK_TOOLS,
            tool_executor=_execute_tool,
            thinking_level="medium",
            max_tool_rounds=3,
        )

        cleaned_html = sanitize_telegram_html(raw_answer)
        if not cleaned_html:
            return html.escape("I drew a blank on that one — try asking me again in a sec!")
        return cleaned_html


memory_engine = MemoryEngine()
