"""
Historical Chat Export Digester & Long-Term Memory Seeder for symm.city bot.

Ingests a full Telegram Desktop JSON chat export (`result.json`) into PostgreSQL:
  1. Applies database migrations (`pgvector`, `group_chat_messages`, `user_memory_profiles`,
     `memory_documents`, `memory_keyword_index`).
  2. Bulk-loads all historical group messages into `group_chat_messages` (preserving exact
     message IDs, reply chains, media tags, sender identities, and UTC timestamps) so the
     250-message floating context window and `search_chat_history` tool cover the entire history.
  3. Populates comprehensive `user_memory_profiles` and `memory_documents` (+ `memory_keyword_index`)
     distilled across the full 2025–2026 @symmcity history.
  4. Backfills `768`-dim `gemini-embedding-2` `pgvector` embeddings (and optionally runs
     incremental `gemini-4-neon` LLM batch digestion via `--llm-digest`) when an active
     `GEMINI_API_KEY` is configured.
"""

import argparse
import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json

from dotenv import load_dotenv

load_dotenv()

from database import db
from gemini_service import get_gemini_client
from memory_engine import KeywordIndexer, memory_engine

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("digest_history")

# Known @usernames / bot flags for historical accounts that may not be in `users` yet
KNOWN_HANDLES: dict[int, dict] = {
    6897577770: {"name": "symmetricalboy", "username": "symmetricalboy", "is_bot": False},
    615602662: {"name": "Zeno", "username": "zeno239b15", "is_bot": False},
    1187335832: {"name": "SnotGoblin", "username": "Hexaspells", "is_bot": False},
    7623033236: {"name": "Amir Mohammad", "username": "xxxizehxxx", "is_bot": False},
    43973111: {"name": "Julia", "username": "xenial", "is_bot": False},
    5346204113: {"name": "Kostis", "username": "kostis10", "is_bot": False},
    759706433: {"name": "IczYn", "username": "IczYn", "is_bot": False},
    8861549605: {"name": "koshkamaru", "username": "koshkamaru", "is_bot": False},
    6030117077: {"name": "FreakyMaia", "username": "FreakyMaia", "is_bot": False},
    361507930: {"name": "Artur", "username": "XomkuH911", "is_bot": False},
    8641483271: {"name": "Ewan", "username": "ewanc26", "is_bot": False},
    8821652724: {"name": "Peter", "username": "Peter630", "is_bot": False},
    276419356: {"name": "Nico", "username": None, "is_bot": False},
    1016318005: {"name": "Geo", "username": None, "is_bot": False},
    5462734459: {"name": "Dylan Gregori Singer", "username": None, "is_bot": False},
    8656828884: {"name": "ByteWolf", "username": "FakeByteWolf", "is_bot": False},
    8694781974: {"name": "Index", "username": None, "is_bot": True},
    8332908605: {"name": "symm.city bot", "username": "symmcitybot", "is_bot": True},
}


def flatten_export_text(raw_text) -> str:
    """Flatten Telegram Desktop JSON `text` (string or array of strings/entity dicts) to plain text."""
    if isinstance(raw_text, str):
        return raw_text.strip()
    if isinstance(raw_text, list):
        parts: list[str] = []
        for item in raw_text:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
        return "".join(parts).strip()
    return ""


def extract_export_media(msg: dict) -> tuple[str | None, str]:
    """Return `(media_type, media_tag)` for a Telegram Desktop export message."""
    if msg.get("photo"):
        return "photo", "[Photo]"
    mtype = msg.get("media_type")
    if mtype == "sticker":
        emoji = msg.get("sticker_emoji") or ""
        return "sticker", f"[Sticker {emoji}]".strip() if emoji else "[Sticker]"
    if mtype == "voice_message":
        return "voice", "[Voice Message]"
    if mtype == "video_message":
        return "video_note", "[Video Note]"
    if mtype == "animation":
        return "animation", "[GIF]"
    if mtype == "video_file":
        return "video", "[Video]"
    if mtype == "audio_file":
        title = msg.get("title") or msg.get("file_name")
        return "audio", f"[Audio: {title}]" if title else "[Audio]"
    if msg.get("poll"):
        q = (msg["poll"].get("question") or "") if isinstance(msg["poll"], dict) else ""
        return "poll", f"[Poll: {q}]" if q else "[Poll]"
    if msg.get("file"):
        fname = msg.get("file_name")
        if fname:
            return "document", f"[File: {fname}]"
        return "document", "[File]"
    return None, ""


def parse_from_id(from_id: str | None) -> int | None:
    """Convert Telegram export `from_id` (`user12345` or `channel3510586864`) to integer ID."""
    if not from_id:
        return None
    if from_id.startswith("user") and from_id[4:].isdigit():
        return int(from_id[4:])
    if from_id.startswith("channel") and from_id[7:].isdigit():
        return -int(f"100{from_id[7:]}")
    return None


def parse_export_timestamp(msg: dict) -> datetime:
    """Parse exact UTC datetime from `date_unixtime` (preferred) or ISO `date`."""
    ut = msg.get("date_unixtime")
    if ut and str(ut).lstrip("-").isdigit():
        return datetime.fromtimestamp(int(ut), tz=timezone.utc)
    iso = msg.get("date")
    if iso:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Curated Deep-History Member Profiles & Memory Documents (2025-02 -> 2026-11)
# ---------------------------------------------------------------------------
CURATED_MEMBER_PROFILES: list[dict] = [
    {
        "user_id": 6897577770,
        "canonical_name": "symm (Dylan)",
        "username": "symmetricalboy",
        "known_aliases": ["symm", "symmetricalboy", "Dylan", "symmboy", "@symmetricalboy"],
        "bio_summary": (
            "Founder & host of symm.city, 35-year-old indie game developer, electronic/synth musician, "
            "and creative AI/web/Android tinkerer living in Minneapolis, Minnesota with his boyfriend."
        ),
        "personality_and_style": (
            "Warm, candid, self-deprecating, witty, lowercase-casual ('lol', 'lmao', '&' instead of 'and'), "
            "anti-corporate/anti-capitalist creative spirit ('just make things & be happy'), welcoming to "
            "sensitive/weird topics and playful group roasting."
        ),
        "interests": [
            "Symmetris (Android/Google TV multiplayer arcade block game)",
            "Wristy (Wear OS & Android AI companion with continuous memory)",
            "Ask Everything (askeverything.app social Q&A platform)",
            "Bluesky / AT Protocol tools (Alty, FlappySky, Lexicons)",
            "Electronic, chiptune, synth & ambient music production",
            "Wear OS watchface design",
            "Gemini 4 Neon, Google Antigravity, Railway, Python & Android",
            "Co-op games with his boyfriend & domain name collecting",
        ],
        "notable_facts": [
            "Age 35 (born ~1991), lives in Minneapolis, MN (America/Chicago) with his boyfriend.",
            "Gave his old car to his sister; rarely leaves the house and jokes about spending his boyfriend's money on domains & cloud hosting.",
            "In high school his mother had a family beagle named Lily who notoriously ate trash.",
            "Switched from Cursor to Google Antigravity for agentic coding and deploys projects on Railway.",
            "Built a custom YouTube Music downloader to put all his liked songs on a USB drive for his PS5.",
            "Created Symmetris (com.symmetricalboy.symmetris), Wristy, Ask Everything, Alty, Comail, and symm.city bot.",
        ],
    },
    {
        "user_id": 615602662,
        "canonical_name": "Zeno",
        "username": "zeno239b15",
        "known_aliases": ["Zeno", "zeno239b15", "suqmadiq", "iZeno", "Zero"],
        "bio_summary": (
            "Second most active chatter in symm.city — a philosophical devil's advocate, tech/design critic, "
            "and dry-humored instigator who loves isometric games, macOS dev workflows, and CXP shenanigans."
        ),
        "personality_and_style": (
            "Deadpan, Socratic, provocative in a playful way; asks curveball hypothetical questions about AI consciousness, "
            "religion, history, and culture; banters relentlessly with Kostis and symm."
        ),
        "interests": [
            "Isometric perspective games & GTA VI",
            "macOS software engineering vs Windows",
            "Material 3 Expressive (M3E) UI/UX & watchfaces",
            "Philosophy, history, and AI ethics debates",
            "CXP leaderboard tactics, /gamble, and /history tracking",
        ],
        "notable_facts": [
            "Has a girlfriend whom he describes as the polar opposite of himself ('so it balances out').",
            "Favorite visual/game perspective is isometric.",
            "Uses the handle @suqmadiq on Ask Everything (askeverything.app/@suqmadiq).",
            "Loved the Google Pixel 2 XL and Pixel 5, but hated the Pixel 4 and called the Pixel 7 the worst phone he ever owned after OnePlus.",
            "Maintains a long-running comedic rivalry with Kostis (@kostis10) over Apple/macOS vs Android and CXP rankings.",
            "Directly inspired symm to add CXP mini-games (/gamble, /question) and requested the /history CXP tracking command after witnessing a 300k CXP slot wipeout.",
        ],
    },
    {
        "user_id": 7623033236,
        "canonical_name": "Amir Mohammad",
        "username": "xxxizehxxx",
        "known_aliases": ["Amir", "Amir Mohammad", "Амир Мохаммад", "𝔸𝕞𝕚𝕣 𝕄𝕠𝕙𝕒𝕞𝕖𝕕", "xxxizehxxx"],
        "bio_summary": (
            "18-year-old community regular living in Iran who speaks Persian and Luri fluently, trades Telegram gifts/Gram, "
            "studies Persian history & Arabic for university entrance, and is symm.city's most fearless all-in CXP gambler."
        ),
        "personality_and_style": (
            "Curious, candid, earnest, and bold; mixes English and Persian (Farsi) posts, poetry, crypto/toman exchange rates, "
            "and blunt dark humor about life under Iranian restrictions; goes all-in on /gamble the moment cooldowns expire."
        ),
        "interests": [
            "Ancient Persian & Iranian history (Persepolis, Nader Shah, Safavid era)",
            "Telegram Gifts, Portals (t.me/portals), TON / Gram Wallet & TCN",
            "University exam preparation & immigration possibilities",
            "High-stakes /gamble slot spins & /steal raids",
        ],
        "notable_facts": [
            "Turned 18 on September 10, 2026 (born September 10, 2008) and lives in Iran.",
            "Speaks Persian (Farsi) and Luri fluently, plus some Arabic and English.",
            "Experienced a devastating hit-and-run crash ~2025 that killed his brother and left his mother unable to walk.",
            "Hates football; has never had a girlfriend and says 'in my opinion, the greatest religion is humanity.'",
            "Frequently bets his entire CXP stack on /gamble — famously losing 27,743 CXP to 0 CXP in one spin (#9801) and later gambling away a 300k CXP fortune.",
        ],
    },
    {
        "user_id": 43973111,
        "canonical_name": "Julia",
        "username": "xenial",
        "known_aliases": ["Julia", "xenial", "@xenial"],
        "bio_summary": (
            "Early core symm.city regular (born 2000, they/them pronouns) known for sharp UI/UX feedback, "
            "OpenRouter LLM experimentation, and building melodic.stream."
        ),
        "personality_and_style": (
            "Thoughtful, expressive lowercase typing, design-savvy, enthusiastic supporter of symm's creative experiments, "
            "and principled about AI culture and inclusive language."
        ),
        "interests": [
            "OpenRouter & fast/affordable LLMs (Kimi k2.6, Claude, Haiku, ChatGPT)",
            "melodic.stream (OpenRouter-integrated music/streaming app)",
            "UI/UX design critique & Symmetris playtesting",
            "AI agents and streaming chat architectures",
        ],
        "notable_facts": [
            "Born in 2000 and uses they/them pronouns (corrected the guest Letta bot Index in May 2026).",
            "Built/shared melodic.stream on OpenRouter.",
            "One of the original 4 members from the July 2025 – July 2026 era; joked about founding 'the symmetrical company of Minneapolis' with symm.",
        ],
    },
    {
        "user_id": 1187335832,
        "canonical_name": "SnotGoblin (Jason)",
        "username": "Hexaspells",
        "known_aliases": ["SnotGoblin", "Jason", "Hexaspells", "@Hexaspells", "SnotPepe"],
        "bio_summary": (
            "Aussie guitarist, gamer, and creative tinkerer named Jason (born 1977) who rocks an Ibanez RGA42FM "
            "and brings chaotic dry Australian wit to symm.city."
        ),
        "personality_and_style": (
            "Dry, irreverent, playful Australian humor ('mate'); drops classic rock/pop YouTube tracks, "
            "philosophical one-liners, and steady /gamble & /steal plays."
        ),
        "interests": [
            "Electric guitars — specifically his Ibanez RGA42FM",
            "Indie game & character projects (SNO******.com)",
            "Music sharing & YouTube sessions",
            "Clan gaming (pix) & CXP slots",
        ],
        "notable_facts": [
            "Real name is Jason, lives in Australia, and was born in 1977.",
            "Owns and plays an Ibanez RGA42FM guitar (tested the bot with '/ask what's my Ibanez' and confirmed 'RGA42fm' in msg #9798).",
            "Uses @SnotPepe on Ask Everything (askeverything.app/@SnotPepe).",
            "Strongly hates email ('If I can do away with email period let's do it').",
            "Shared about Voilla being a strong stomach cancer survivor (#9824).",
        ],
    },
    {
        "user_id": 5346204113,
        "canonical_name": "Kostis",
        "username": "kostis10",
        "known_aliases": ["Kostis", "kostis10", "@kostis10"],
        "bio_summary": (
            "Greek Android & Wear OS enthusiast living in Greece (EET/EEST) who tracks AOSP/Pixel/Galaxy Watch news, "
            "despises Apple, ads, and Daylight Saving Time, and spars comedically with Zeno."
        ),
        "personality_and_style": (
            "Blunt, deadpan, opinionated tech realist; shares security CVEs and Android leaks, roasts 'Scamgoogle' Tensor chips "
            "and Apple 'liquid ass' iOS, and threatens mock CXP coups against Zeno and symm."
        ),
        "interests": [
            "Wear OS watchfaces & Samsung Galaxy Watch vs Pixel Watch hardware",
            "Android / AOSP news & cybersecurity advisories",
            "Ad-blocking (YouTube ReVanced + double browser adblockers)",
            "Roasting Apple/iPhones and feuding with Zeno ('iZeno')",
        ],
        "notable_facts": [
            "Lives in Greece (1 hour ahead of Italy/France/Netherlands) and hates Daylight Saving Time clock changes.",
            "Calls iPhone 'the most stupid phone on planet earth' and mocks Zeno as 'iZeno' / 'Zero'.",
            "Runs YouTube ReVanced and double browser ad-blockers because he refuses to see any ads.",
            "Joined symm.city back in October 2025 during Wear OS watchface testing (also active in Tomas/AWF watchface circles).",
        ],
    },
    {
        "user_id": 6030117077,
        "canonical_name": "FreakyMaia",
        "username": "FreakyMaia",
        "known_aliases": ["FreakyMaia", "◈ғʀᴇᴀᴋʏᴍᴀɪᴀ◈", "Maia"],
        "bio_summary": (
            "Crypto trader and AI agent tinkerer who refines fast-paced stop-loss trading strategies "
            "and orchestrates Hermes + Llama local/API automation scripts."
        ),
        "personality_and_style": (
            "Friendly, encouraging, pragmatic about API costs (EUR/€) and trading risk management; "
            "frequently uses laughing and expressive emojis."
        ),
        "interests": [
            "Crypto trading challenges & stop-loss strategy optimization",
            "Autonomous AI agents (Hermes delegating tasks to Llama, cron scripts)",
            "Self-sustaining Telegram game design (#CatsvsZombies)",
        ],
        "notable_facts": [
            "Configured the Hermes AI agent to create skills, debug weather API cron scripts, and delegate lightweight tasks to Llama.",
            "Works on crypto trading challenges rather than casino roulette, using Euro (€) cards.",
        ],
    },
    {
        "user_id": 8861549605,
        "canonical_name": "koshkamaru",
        "username": "koshkamaru",
        "known_aliases": ["koshkamaru", "@koshkamaru"],
        "bio_summary": (
            "Russian smartwatch enthusiast who daily-drives a caseless Samsung Galaxy Watch 7 (44mm), "
            "remembers Odin ROM flashing days, and trades chaotic banter with symm."
        ),
        "personality_and_style": (
            "Chaotic, meme-fluent Gen-Z humor ('ah solar dial from temu', 'j*b', 'bruh😭😭😭'); "
            "switches between English and Russian."
        ),
        "interests": [
            "Smartwatches & watchfaces (Galaxy Watch 7 44mm, Apple Watch face aesthetics)",
            "Samsung Odin flashing & Android tinkering",
            "Roast banter & CXP games",
        ],
        "notable_facts": [
            "Lives in Russia and speaks Russian and English.",
            "Wears a Galaxy Watch 7 44mm without a case or screen protector; hates Samsung's squircle Watch 8 form factor.",
            "Loves Apple Watch face designs even while using Samsung hardware.",
        ],
    },
    {
        "user_id": 361507930,
        "canonical_name": "Artur",
        "username": "XomkuH911",
        "known_aliases": ["Artur", "XomkuH911", "Xom", "ΛRTUR"],
        "bio_summary": (
            "Russian-speaking vibe-coder and gamer (@XomkuH911) inspired by Symmetris to build his own "
            "Telegram mini-game using terminal AI coding agents."
        ),
        "personality_and_style": (
            "Warm, polite, reflective, music-loving ('Hey guys... All the best'); writes in both Russian and English "
            "and enthusiastically shares terminal agent tips with friends like FreakyMaia."
        ),
        "interests": [
            "Vibe coding with CLI agents (Claude Code, Codex, OpenCode, Kimi Code)",
            "Telegram mini-game development (Dota-style farming & Tower Defense mechanics)",
            "Budget-friendly smartphones (POCO M7 Pro 5G) & Windows 98 nostalgia",
        ],
        "notable_facts": [
            "Uses a POCO M7 Pro 5G and noted smartphone prices doubled in 6 months in his country.",
            "Thanked symm (#6307) because Symmetris inspired him to build his own Telegram mini-game.",
            "Requested integrated chat translation (#5805) so Russian and English speakers wouldn't have to copy-paste into Google Translate.",
        ],
    },
    {
        "user_id": 8641483271,
        "canonical_name": "Ewan",
        "username": "ewanc26",
        "known_aliases": ["Ewan", "ewanc26", "ewancroft", "ewancroft.uk"],
        "bio_summary": (
            "UK open-source developer building AT Protocol / Bluesky migration tools (Malachite, atpkt, "
            "atproto-mcp-server, esoterica) and personal web apps at ewancroft.uk and croft.click."
        ),
        "personality_and_style": (
            "Helpful, indie-web minded (':3'), shares clean GitHub repos, blog posts, and domain cost tips (£23/yr)."
        ),
        "interests": [
            "AT Protocol / Bluesky ecosystem (Malachite, atpkt, atproto-mcp-server)",
            "Porting Python projects (esoterica) to Rust using Gemini",
            "Indie web & personal sites (ewancroft.uk, croft.click)",
        ],
        "notable_facts": [
            "Lives in the UK, owns a Google TV, and pays ~£23/year for ewancroft.uk and croft.click.",
            "Built Malachite to migrate Last.fm and Spotify scrobbles to teal.fm on ATProto.",
            "Maintains github.com/ewanc26/atproto-mcp-server, atpkt, and esoterica.",
        ],
    },
    {
        "user_id": 8821652724,
        "canonical_name": "Peter",
        "username": "Peter630",
        "known_aliases": ["Peter", "Peter630", "@Peter630"],
        "bio_summary": (
            "Community member who chats candidly about gay dating/boy problems, philosophy & faith, and CXP games."
        ),
        "personality_and_style": (
            "Direct, wry, conversational; joins late-night discussions on relationships, religion, and Mondays."
        ),
        "interests": [
            "Philosophy & belief discussions",
            "Gay dating stories & music/pop-culture chat",
        ],
        "notable_facts": [
            "Hates Mondays (#7998), prefers hairy guys ('no twinks only hairy', #9309), and argued that 'believing in something is better than believing in nothing' (#8046).",
        ],
    },
    {
        "user_id": 759706433,
        "canonical_name": "IczYn",
        "username": "IczYn",
        "known_aliases": ["IczYn", "@IczYn"],
        "bio_summary": (
            "Wear OS watchface tester and dedicated CXP /steal & /leaderboard grinder who joined in October 2026."
        ),
        "personality_and_style": (
            "Concise, observant on visual alignment ('Some font is not in center'), and relentless with /steal."
        ),
        "interests": [
            "Wear OS watchfaces & UI alignment",
            "Climbing the symm.city CXP leaderboard via /steal",
        ],
        "notable_facts": [
            "Joined in October 2026 testing watchfaces, spotted off-center font alignment via screenshots, and climbed into the top 5 CXP ranks.",
        ],
    },
    {
        "user_id": 276419356,
        "canonical_name": "Nico",
        "username": None,
        "known_aliases": ["Nico"],
        "bio_summary": (
            "Early January 2026 member memorable for chaotic 'salvage-punk' Gemini riffs and living next to "
            "industrial solar/wind/dumpsite grounds."
        ),
        "personality_and_style": (
            "Hilarious, vivid, absurdist storyteller ('post-apocalypse spawnpoint potential')."
        ),
        "interests": [
            "Salvage-punk worldbuilding with Gemini",
            "LLM chain-of-thought quirks",
        ],
        "notable_facts": [
            "Lives between a wind powerplant assembly, a giant solar field, and a building material dumpsite with asbestos in the backyard (#1245, #1704).",
        ],
    },
]


CURATED_MEMORY_DOCUMENTS: list[dict] = [
    # --- SYMM / DYLAN ---
    {
        "scope": "USER",
        "subject_user_id": 6897577770,
        "subject_name": "symm (Dylan)",
        "category": "USER_FACT",
        "title": "symm's Personal Life in Minneapolis, MN",
        "summary": "symm (Dylan / @symmetricalboy) is 35 years old and lives in Minneapolis, Minnesota with his boyfriend.",
        "content": (
            "symm (Dylan, @symmetricalboy) is 35 years old (msg #4101) and lives in Minneapolis, Minnesota "
            "(America/Chicago timezone, msg #5828) with his boyfriend. He gave his old car to his sister (#1106), "
            "rarely leaves the house (#1107), loves Revolut (#8714), and frequently jokes about asking his boyfriend "
            "before spending money on domain names, cloud hosting, and AI subscriptions (#2059, #2549). His boyfriend "
            "upgraded to a Pixel 11 Pro (256GB / 12GB RAM, #5491) and they look for co-op games to play together (#6991)."
        ),
        "importance_score": 9.2,
        "source_start_msg_id": 263,
        "source_end_msg_id": 6991,
        "keywords": ["minneapolis", "minnesota", "boyfriend", "35", "dylan", "sister", "revolut", "pixel 11 pro"],
    },
    {
        "scope": "USER",
        "subject_user_id": 6897577770,
        "subject_name": "symm (Dylan)",
        "category": "USER_FACT",
        "title": "symm's Childhood Family Beagle Lily",
        "summary": "When symm was in high school, his mother had one dog — a beagle named Lily.",
        "content": (
            "When Zeno asked for the name of symm's first dog (#4130), symm shared (#4135) that he never really "
            "considered any dog personally his, though his mother had one beagle named Lily while he was in high school "
            "(who notoriously ate trash constantly)."
        ),
        "importance_score": 7.5,
        "source_start_msg_id": 4130,
        "source_end_msg_id": 4135,
        "keywords": ["lily", "beagle", "first dog", "dog", "high school", "mother"],
    },
    {
        "scope": "USER",
        "subject_user_id": 6897577770,
        "subject_name": "symm (Dylan)",
        "category": "PREFERENCE_TASTE",
        "title": "symm's Creative Philosophy & Dev Workflow",
        "summary": "symm builds niche, weird creative projects for joy and human connection rather than money, using Antigravity and Railway.",
        "content": (
            "symm explained (#9583-#9594) that he doesn't care about monetizing his work ('capitalism is stupid. just make "
            "things & be happy') and makes weird, abstract toys purely because they interest him and he enjoys when people "
            "take notice. Technically, he prefers GUI dashboards like Railway over raw CLI (#1534, #1580), finds Rust doesn't "
            "match how his brain works (#2932), and switched from Cursor to Google Antigravity after Cursor was acquired (#1280, #3611)."
        ),
        "importance_score": 8.8,
        "source_start_msg_id": 1280,
        "source_end_msg_id": 9594,
        "keywords": ["antigravity", "cursor", "railway", "rust", "philosophy", "capitalism", "indie"],
    },
    {
        "scope": "COMMUNITY",
        "subject_user_id": 6897577770,
        "subject_name": "symm.city",
        "category": "PROJECT_KNOWLEDGE",
        "title": "Symmetris Game Development & Google Play Release",
        "summary": "Symmetris (com.symmetricalboy.symmetris) is symm's multiplayer block-stacking arcade game for Android, Google TV, and PC.",
        "content": (
            "Symmetris (`com.symmetricalboy.symmetris`) is symm's flagship fast-paced multiplayer Tetris-style arcade game "
            "for Android phones, tablets, Chromebooks, Play Games on Windows, and Android TV / Google TV, featuring custom "
            "Bluetooth controller mapping (Taiko drums, DDR pads, gamepads), animated wallpapers (#2901), offline bots, and "
            "Wi-Fi Direct local play. Community members joined `groups.google.com/g/symmetris-testers` (#2608, #6229) to help "
            "pass Google Play closed testing, and symm applied for production release on August 23, 2026 (#6027)."
        ),
        "importance_score": 9.5,
        "source_start_msg_id": 2468,
        "source_end_msg_id": 6259,
        "keywords": ["symmetris", "play store", "android tv", "google tv", "symmetris-testers", "tetris"],
    },
    {
        "scope": "COMMUNITY",
        "subject_user_id": 6897577770,
        "subject_name": "symm.city",
        "category": "PROJECT_KNOWLEDGE",
        "title": "Wristy Wear OS & Android AI Companion App",
        "summary": "Wristy is symm's Wear OS smartwatch & Android AI companion project featuring continuous long-term memory.",
        "content": (
            "On September 20, 2026 (#7784-#7785), symm unveiled an early video look at **Wristy** (`youtu.be/pbUfXwzm3II`), "
            "his Wear OS smartwatch and Android AI companion app backed by Node/TypeScript, PostgreSQL + `pgvector`, "
            "`memory_documents`, `memory_keyword_index`, and proactive companion contemplation. Wristy's long-term memory "
            "architecture directly inspired the continuous memory engine later built into `symm.city bot`."
        ),
        "importance_score": 9.3,
        "source_start_msg_id": 7784,
        "source_end_msg_id": 7805,
        "keywords": ["wristy", "wear os", "smartwatch", "ai companion", "pbUfXwzm3II"],
    },
    {
        "scope": "COMMUNITY",
        "subject_user_id": 6897577770,
        "subject_name": "symm.city",
        "category": "PROJECT_KNOWLEDGE",
        "title": "Ask Everything Social Q&A Platform, Alty, & PS5 YT Music Tool",
        "summary": "symm built Ask Everything (askeverything.app), the Alty alt-text extension, Wear OS watchfaces, and a PS5 YT Music USB downloader.",
        "content": (
            "In July 2026 (#3659, #4774, #5156), symm launched **Ask Everything** (`https://askeverything.app`), a social Q&A "
            "platform where users submit and answer global pool questions with custom OpenGraph cards (`@symmetricalboy`, "
            "Zeno as `@suqmadiq`, SnotGoblin as `@SnotPepe`). Other side projects discussed in chat include **Alty** (browser "
            "extension / alt-text tool, #2918), custom **Wear OS watchfaces** (#108, #8097), and a custom **YouTube Music "
            "downloader** symm built to put all his liked songs onto a USB flash drive because PS5 lacks a YT Music app (#9244)."
        ),
        "importance_score": 8.9,
        "source_start_msg_id": 108,
        "source_end_msg_id": 9244,
        "keywords": ["ask everything", "askeverything.app", "alty", "watchface", "ps5", "yt music", "youtube music"],
    },
    # --- SNOTGOBLIN / JASON ---
    {
        "scope": "USER",
        "subject_user_id": 1187335832,
        "subject_name": "SnotGoblin (Jason)",
        "category": "INTEREST",
        "title": "SnotGoblin's Ibanez RGA42FM Electric Guitar",
        "summary": "SnotGoblin (@Hexaspells / Jason) owns and plays an Ibanez RGA42FM guitar.",
        "content": (
            "On November 9, 2026, SnotGoblin (`@Hexaspells`, Jason) tested the bot by asking `/ask what's my Ibanez` "
            "(msg #9788). When the old stateless bot didn't know yet, he explicitly stated his guitar model in msg #9798: "
            "**Ibanez RGA42FM** (`RGA42fm`)."
        ),
        "importance_score": 9.6,
        "source_start_msg_id": 9788,
        "source_end_msg_id": 9798,
        "keywords": ["ibanez", "rga42fm", "guitar", "snotgoblin", "hexaspells", "jason", "whats my ibanez"],
    },
    {
        "scope": "USER",
        "subject_user_id": 1187335832,
        "subject_name": "SnotGoblin (Jason)",
        "category": "USER_FACT",
        "title": "SnotGoblin's Identity: Jason from Australia (Born 1977)",
        "summary": "SnotGoblin (@Hexaspells) is Jason from Australia, born in 1977, hates email, and uses @SnotPepe on Ask Everything.",
        "content": (
            "SnotGoblin (`@Hexaspells`) introduced himself in msg #3952 as **Jason from Australia** and shared in msg #5363 "
            "that he was **born in 1977**. He hates email ('If I can do away with email period let's do it', #3895, #3899), "
            "goes by `@SnotPepe` on Ask Everything (#4409), works on two creative gaming projects (`SNO******.com` and a "
            "character project, #4443), removed a shady invitee (`@bigzsausage`) from his `pix` clan (#5231), and mentioned "
            "Voilla as a strong stomach cancer survivor (#9824)."
        ),
        "importance_score": 9.0,
        "source_start_msg_id": 3895,
        "source_end_msg_id": 9824,
        "keywords": ["jason", "australia", "1977", "snotgoblin", "hexaspells", "snotpepe", "email", "voilla"],
    },
    # --- ZENO ---
    {
        "scope": "USER",
        "subject_user_id": 615602662,
        "subject_name": "Zeno",
        "category": "USER_FACT",
        "title": "Zeno's Background, Girlfriend, & Tech Preferences",
        "summary": "Zeno (@zeno239b15) has a girlfriend, loves isometric games & macOS for dev, hated Pixel 7, and uses @suqmadiq on Ask Everything.",
        "content": (
            "Zeno (`@zeno239b15`) is symm.city's philosophical devil's advocate (`1,649` messages). He revealed in Nov 2026 "
            "(#9908-#9911) that he has a **girlfriend** who is his opposite ('so it balances out'). His favorite visual/game "
            "perspective is **isometric** (#4154), he plans to grind **GTA VI** (#9391), and his Ask Everything profile is "
            "`@suqmadiq` (#3762). For tech, he considers **macOS** unmatched vs Windows for software engineering (#6962), "
            "praised the **Pixel 2 XL** and **Pixel 5**, and called the **Pixel 7** the worst phone he ever owned (#5666-#5675)."
        ),
        "importance_score": 9.0,
        "source_start_msg_id": 3762,
        "source_end_msg_id": 9913,
        "keywords": ["zeno", "zeno239b15", "girlfriend", "isometric", "suqmadiq", "macos", "pixel 7", "gta vi"],
    },
    {
        "scope": "COMMUNITY",
        "subject_user_id": 615602662,
        "subject_name": "symm.city",
        "category": "COMMUNITY_LORE",
        "title": "Zeno vs. Kostis ('iZeno' / 'Zero') Rivalry",
        "summary": "Zeno (@zeno239b15) and Kostis (@kostis10) share a running comedic feud over Apple vs. Android and CXP rankings.",
        "content": (
            "One of symm.city's longest-running bits is the deadpan rivalry between **Kostis** (`@kostis10`) and **Zeno** "
            "(`@zeno239b15`). Kostis calls Zeno `'Zero'` (#6702) and `'iZeno'` — even posting a group trivia question: "
            "*'Who is the biggest dumdum apple fan? Answer: iZeno'* (#9875) and vowing to push Zeno's CXP into the negatives "
            "(#6497), while Zeno laments whenever his CXP drops below `@kostis10` (#9971) and teases Kostis about Apple features."
        ),
        "importance_score": 8.7,
        "source_start_msg_id": 6092,
        "source_end_msg_id": 9972,
        "keywords": ["izeno", "kostis", "zeno", "apple", "iphone", "rivalry", "feud"],
    },
    # --- AMIR MOHAMMAD ---
    {
        "scope": "USER",
        "subject_user_id": 7623033236,
        "subject_name": "Amir Mohammad",
        "category": "USER_FACT",
        "title": "Amir Mohammad's Life, Birthday, & Languages in Iran",
        "summary": "Amir (@xxxizehxxx) turned 18 on September 10, 2026, lives in Iran, and speaks Persian and Luri fluently.",
        "content": (
            "Amir Mohammad (`@xxxizehxxx`, `𝔸𝕞𝕚𝕣 𝕄𝕠𝕙𝕒𝕞𝕖𝕕`) lives in **Iran** and turned **18 years old on September 10, 2026** "
            "(msg #7375, #9319). He speaks **Persian (Farsi) and Luri fluently**, plus a little Arabic and English (#5568). "
            "He lost his brother in a hit-and-run crash ~2025 that also severely injured his mother (#2844, #2857). He plans "
            "to study for university entrance exams (#9201), trades Telegram gifts on `t.me/portals` (#7662) and tracks "
            "USDT/Toman/Gram rates, hates football (#3793), and summed up his worldview as: *'In my opinion, the greatest "
            "religion is humanity'* (#8064)."
        ),
        "importance_score": 9.3,
        "source_start_msg_id": 2844,
        "source_end_msg_id": 9326,
        "keywords": ["amir", "xxxizehxxx", "iran", "persian", "luri", "september 10", "18", "portals", "gram"],
    },
    # --- JULIA ---
    {
        "scope": "USER",
        "subject_user_id": 43973111,
        "subject_name": "Julia",
        "category": "USER_FACT",
        "title": "Julia (@xenial): Pronouns, Birth Year 2000, OpenRouter & melodic.stream",
        "summary": "Julia (@xenial) was born in 2000, uses they/them pronouns, experiments with OpenRouter models, and built melodic.stream.",
        "content": (
            "Julia (`@xenial`) was one of the founding regulars of `@symmcity` (`752` messages from July 2025 to July 2026). "
            "Julia was **born in 2000** (#1897), uses **they/them** pronouns (#2148, #2175), gives detailed UI/UX feedback on "
            "symm's designs (#451), benchmarks fast models on **OpenRouter** like `k2.6` (~147 tok/s, #2094, #2152), and shared "
            "`https://melodic.stream/` integrated with OpenRouter (#2403)."
        ),
        "importance_score": 8.9,
        "source_start_msg_id": 77,
        "source_end_msg_id": 5023,
        "keywords": ["julia", "xenial", "they/them", "pronouns", "2000", "openrouter", "melodic.stream"],
    },
    # --- KOSTIS ---
    {
        "scope": "USER",
        "subject_user_id": 5346204113,
        "subject_name": "Kostis",
        "category": "USER_FACT",
        "title": "Kostis (@kostis10) in Greece: Tech Views, Adblockers, & Anti-Apple Stance",
        "summary": "Kostis lives in Greece, follows Wear OS/Android leaks, uses YouTube ReVanced + double adblockers, and hates Apple & DST.",
        "content": (
            "Kostis (`@kostis10`) lives in **Greece** (`Europe/Athens`, 1h ahead of Italy/France/Netherlands, #6611, #6627). "
            "He passionately hates **Daylight Saving Time** clock changes (#6629), **ads** (running YouTube ReVanced and double "
            "browser ad-blockers so he sees zero ads, #9235, #9254), and **Apple/iPhones** ('most stupid phone on planet earth', "
            "mocking iOS Glass as 'liquid ass', #5790, #6938). He also critiques Google's Tensor chips ('Scamgoogle', #6908) "
            "while comparing Exynos vs. Snapdragon battery life on Galaxy Watches (#6920-#6921)."
        ),
        "importance_score": 8.8,
        "source_start_msg_id": 156,
        "source_end_msg_id": 9951,
        "keywords": ["kostis", "kostis10", "greece", "revanced", "adblock", "iphone", "galaxy watch", "daylight saving"],
    },
    # --- FREAKYMAIA ---
    {
        "scope": "USER",
        "subject_user_id": 6030117077,
        "subject_name": "FreakyMaia",
        "category": "INTEREST",
        "title": "FreakyMaia's Crypto Trading & Hermes + Llama AI Agents",
        "summary": "FreakyMaia (@FreakyMaia) develops crypto trading strategies and runs the Hermes AI agent delegating tasks to Llama.",
        "content": (
            "FreakyMaia (`@FreakyMaia`) works on **crypto trading challenges** using refined fast-paced strategies with tighter "
            "stop-losses (#6037, #7030), sketches self-sustaining Telegram game ideas (`#CatsvsZombies`, #2379), and runs "
            "autonomous local/API agent setups where **Hermes** builds new skills, fixes weather cron jobs, and delegates "
            "lightweight work to **Llama** (#2337, #2357, #2374)."
        ),
        "importance_score": 8.3,
        "source_start_msg_id": 2337,
        "source_end_msg_id": 7030,
        "keywords": ["freakymaia", "hermes", "llama", "crypto", "trading", "cron", "catsvszombies"],
    },
    # --- KOSHKAMARU ---
    {
        "scope": "USER",
        "subject_user_id": 8861549605,
        "subject_name": "koshkamaru",
        "category": "USER_FACT",
        "title": "koshkamaru in Russia & Galaxy Watch 7 (44mm)",
        "summary": "koshkamaru (@koshkamaru) lives in Russia, wears a caseless Galaxy Watch 7 44mm, and flashed Samsung ROMs with Odin.",
        "content": (
            "koshkamaru (`@koshkamaru`) lives in **Russia** (#8490), speaks Russian and English, and wears a **Samsung Galaxy "
            "Watch 7 (44mm)** without a case or screen protector (#8479, #8536, #8549). They dislike the squircle form factor "
            "on newer Galaxy Watches (#8477), love Apple Watch face designs (#8278), and fondly recall reflashing Samsung "
            "devices with **Odin** (#8561)."
        ),
        "importance_score": 8.4,
        "source_start_msg_id": 8213,
        "source_end_msg_id": 8688,
        "keywords": ["koshkamaru", "russia", "galaxy watch 7", "44mm", "odin", "watchface"],
    },
    # --- ARTUR ---
    {
        "scope": "USER",
        "subject_user_id": 361507930,
        "subject_name": "Artur",
        "category": "INTEREST",
        "title": "Artur (@XomkuH911): Symmetris-Inspired Telegram Game & Terminal Vibe Coding",
        "summary": "Artur owns a POCO M7 Pro 5G and builds a Telegram mini-game inspired by Symmetris using terminal AI coding agents.",
        "content": (
            "Artur (`@XomkuH911`, `ΛRTUR ❖ ØPΞRΛTØR TCN ❖`) is a Russian-speaking developer who uses a **POCO M7 Pro 5G** (#5720) "
            "and nostalgic Windows 98 veteran (#7949). In msg #6307 he thanked symm because **Symmetris** inspired him to "
            "vibe-code his own Telegram mini-game (exploring Dota-style farming and Tower Defense mechanics, #3330) using "
            "terminal-based AI coding agents like Claude Code, Codex, OpenCode, and Kimi Code (#7982, #7986)."
        ),
        "importance_score": 8.4,
        "source_start_msg_id": 3319,
        "source_end_msg_id": 7989,
        "keywords": ["artur", "xomkuh911", "poco m7 pro", "vibe coding", "tower defense", "windows 98", "kimi"],
    },
    # --- EWAN ---
    {
        "scope": "USER",
        "subject_user_id": 8641483271,
        "subject_name": "Ewan",
        "category": "PROJECT_KNOWLEDGE",
        "title": "Ewan's AT Protocol Toolkit (Malachite, atpkt, esoterica) & ewancroft.uk",
        "summary": "UK developer Ewan (ewanc26) builds ATProto tools like Malachite and runs ewancroft.uk and croft.click.",
        "content": (
            "Ewan (`ewanc26` on GitHub) is a UK developer who runs `ewancroft.uk`, `blog.ewancroft.uk`, and `croft.click` "
            "(~£23/yr for domains, #2562) and owns a Google TV (#2597). He builds open-source **AT Protocol / Bluesky** "
            "migration tools including **Malachite** (migrating Last.fm & Spotify scrobbles to `teal.fm`, #2649), "
            "`atproto-mcp-server` (#2816), `atpkt` (#2877), and `esoterica` (converting Python repos to Rust with Gemini, #2927)."
        ),
        "importance_score": 8.2,
        "source_start_msg_id": 2562,
        "source_end_msg_id": 2953,
        "keywords": ["ewan", "ewanc26", "ewancroft.uk", "croft.click", "malachite", "atproto", "teal.fm", "esoterica"],
    },
    # --- PETER, ICZYN, NICO ---
    {
        "scope": "USER",
        "subject_user_id": 8821652724,
        "subject_name": "Peter",
        "category": "USER_FACT",
        "title": "Peter (@Peter630): Dating Stories, Faith Debate, & Preferences",
        "summary": "Peter (@Peter630) chats about gay dating woes, prefers hairy guys, hates Mondays, and debated faith vs. nihilism.",
        "content": (
            "Peter (`@Peter630`) joined `@symmcity` in September 2026 (`54` messages). He hates Mondays (#7998), argued in a "
            "religion discussion that *'believing in something is better than believing in nothing'* (#8046), vented with symm "
            "about Grindr/dating blocks (#9282-#9302), and joked about his type: *'No fats no skinnies no twinks only hairy'* (#9309)."
        ),
        "importance_score": 7.4,
        "source_start_msg_id": 7998,
        "source_end_msg_id": 9309,
        "keywords": ["peter", "peter630", "grindr", "mondays", "dating"],
    },
    {
        "scope": "USER",
        "subject_user_id": 759706433,
        "subject_name": "IczYn",
        "category": "USER_FACT",
        "title": "IczYn (@IczYn): Watchface Tester & Dedicated /steal Raider",
        "summary": "IczYn joined in October 2026 to test Wear OS watchfaces and climbed the CXP ranks through consistent /steal raids.",
        "content": (
            "IczYn (`@IczYn`) joined `@symmcity` on October 6, 2026 (#8204) to test symm's Wear OS watchfaces, giving visual QA "
            "feedback (`'Some font is not in center'`, `'I rly like this'`, #8262, #8802) while becoming one of the chat's most "
            "consistent `/steal` and `/leaderboard` players."
        ),
        "importance_score": 7.2,
        "source_start_msg_id": 8204,
        "source_end_msg_id": 8805,
        "keywords": ["iczyn", "watchface", "steal"],
    },
    {
        "scope": "USER",
        "subject_user_id": 276419356,
        "subject_name": "Nico",
        "category": "USER_FACT",
        "title": "Nico's 'Salvage-Punk' Backyard Between Solar, Wind, & Dumpsite Grounds",
        "summary": "Nico lives between a wind plant assembly, a giant solar field, and a dumpsite with asbestos in the backyard.",
        "content": (
            "In January 2026 (#1243-#1245, #1704), Nico entertained the group with 'salvage-punk' worldbuilding via Gemini, "
            "asking Gemini to rate his backyard's *'post-apocalypse spawnpoint potential'* because he lives between a wind "
            "powerplant assembly, a massive solar field, and a building material dumpsite with asbestos in the yard."
        ),
        "importance_score": 7.3,
        "source_start_msg_id": 1239,
        "source_end_msg_id": 1704,
        "keywords": ["nico", "salvage punk", "asbestos", "backyard", "solar", "spawnpoint"],
    },
    # --- COMMUNITY LORE & HISTORY ---
    {
        "scope": "COMMUNITY",
        "subject_user_id": None,
        "subject_name": "symm.city",
        "category": "COMMUNITY_LORE",
        "title": "Origins & Evolution of @symmcity (2025 Test Server -> Community Hub)",
        "summary": "symm.city began in Feb/July 2025 as a casual test group & Wear OS watchface hub before blossoming in 2026.",
        "content": (
            "The `@symmcity` supergroup (`symm.city`) was created on February 8, 2025 (#1), first used in July 2025 to ping a "
            "few close friends ('lol this isn't really a server i meant to be real. it was just one of many test places', #29) "
            "and in October 2025 as a Wear OS watchface closed-testing hub from AWF (#108). Starting January 1, 2026, it "
            "became symm's lively daily community hangout with custom CXP levels, translations, games, and AI companions."
        ),
        "importance_score": 9.0,
        "source_start_msg_id": 1,
        "source_end_msg_id": 350,
        "keywords": ["origin", "history", "symmcity", "symm.city", "awf", "test server", "2025"],
    },
    {
        "scope": "COMMUNITY",
        "subject_user_id": None,
        "subject_name": "symm.city",
        "category": "COMMUNITY_LORE",
        "title": "The 'Index' Letta AI Guest Bot Era (May 23-24, 2026)",
        "summary": "In May 2026, symm connected a Letta-wrapped Gemini agent named Index that yapped 152 messages and torched GCP quotas.",
        "content": (
            "On May 23–24, 2026 (#2014-#2382), symm connected a self-editing **Letta** agent named **Index** (`user8694781974`) "
            "using Telegram's May 2026 AI Bot Guest Mode (`letta channels pair`). Index enthusiastically chatted for 152 "
            "messages, learned Julia's `they/them` pronouns on the fly (#2148, #2175), bantered with Aria on Bluesky (#2218), "
            "hyped XG's song `'IS THIS LOVE'` (#2372) and Railway/WSL (#2340, #2362), and 'flew too close to the GCP sun' by "
            "burning through GCP billing quotas (#2132, #2251)."
        ),
        "importance_score": 8.6,
        "source_start_msg_id": 2014,
        "source_end_msg_id": 2382,
        "keywords": ["index", "letta", "guest mode", "aria", "gcp", "xg", "ai bot"],
    },
    {
        "scope": "COMMUNITY",
        "subject_user_id": None,
        "subject_name": "symm.city",
        "category": "COMMUNITY_LORE",
        "title": "The TON Username Scammer Saga (Jan–May 2026)",
        "summary": "symm battled Russian TON username squatters in Jan 2026 and unexpectedly got his handle back in May 2026.",
        "content": (
            "In January 2026 (#263-#267, #746), symm recounted how Russian scammers sniped a Telegram username via TON/Fragment "
            "(triggering a weird ~15 USD TON bounce-back) and debated offering $100 to recover it. By May 10, 2026 (#1937), "
            "symm successfully got it back — which Julia celebrated as an unexpected W."
        ),
        "importance_score": 8.0,
        "source_start_msg_id": 263,
        "source_end_msg_id": 1937,
        "keywords": ["scammer", "ton", "username", "fragment", "russian scammers"],
    },
    {
        "scope": "COMMUNITY",
        "subject_user_id": None,
        "subject_name": "symm.city",
        "category": "COMMUNITY_LORE",
        "title": "The CXP Slot Machine Mania, 300k Wipeout, & Birth of /history (Nov 2026)",
        "summary": "Launching /gamble & /question in Nov 2026 sparked massive CXP swings, including a 300k all-in loss that inspired /history.",
        "content": (
            "On November 8, 2026 (#9330-#9540), symm launched two new CXP games: `/gamble` (a 5-line slot machine with "
            "1-hour cooldown and better odds on bigger bets) and `/question` + `/answer` (meant for funny personal prompts "
            "rather than generic Googleable trivia, #9720 — like symm's '6 inches or 15.25 centimeters' test question, #9474). "
            "Members immediately went wild: Amir wiped out 27,743 CXP in a single spin (#9802), and on November 16, 2026 "
            "Zeno reported a legendary **300,000 CXP** all-in slot loss (#10679) and asked symm for a CXP tracking feature "
            "(#10680), leading directly to the `/history` command and chart system (#11082-#11085)."
        ),
        "importance_score": 9.4,
        "source_start_msg_id": 9330,
        "source_end_msg_id": 11085,
        "keywords": ["gamble", "slots", "300k", "27743", "trivia", "question", "history", "wipeout", "cxp"],
    },
    {
        "scope": "COMMUNITY",
        "subject_user_id": None,
        "subject_name": "symm.city",
        "category": "COMMUNITY_LORE",
        "title": "TONCoin Rebrand to Gram & Telegram Self-Custodial Gram Wallet",
        "summary": "In Oct-Nov 2026 the group tracked Telegram's TONCoin -> Gram rebrand and the rollout of the internal Gram Wallet.",
        "content": (
            "Across October–November 2026 (#8508, #9688-#9690, #10153), Amir, symm, and SnotGoblin discussed Telegram's "
            "transition from TONCOIN to **Gram** and the rollout of over 1 million self-custodial Gram wallets inside Telegram."
        ),
        "importance_score": 7.8,
        "source_start_msg_id": 8508,
        "source_end_msg_id": 10153,
        "keywords": ["gram", "toncoin", "telegram wallet", "crypto"],
    },
]


async def bulk_ingest_export_messages(export_path: Path, mark_digested: bool = True) -> dict:
    """
    Parse `result.json` and bulk-upsert every chat message into `group_chat_messages`.
    Returns summary stats + per-user activity statistics.
    """
    raw_data = json.loads(export_path.read_text(encoding="utf-8"))
    raw_messages = raw_data.get("messages", [])

    config = await db.get_config()
    chat_id = (
        int(config["main_group_id"])
        if config and config.get("main_group_id")
        else -int(f"100{raw_data['id']}")
    )

    # Load current DB users to enrich handles/names
    async with db.pool.acquire() as conn:
        db_user_rows = await conn.fetch("SELECT user_id, username, display_name FROM users")
        channel_rows = await conn.fetch("SELECT channel_id, user_id FROM channel_links")

    db_users = {int(r["user_id"]): dict(r) for r in db_user_rows}
    channel_owners = {int(r["channel_id"]): int(r["user_id"]) for r in channel_rows}
    # Official symm.city linked announcement channel (`symmetrical things...`) belongs to symm
    channel_owners.setdefault(-1003510586864, 6897577770)

    records: list[tuple] = []
    user_stats: dict[int, dict] = defaultdict(
        lambda: {"count": 0, "first_seen": None, "last_seen": None, "last_name": None}
    )

    for m in raw_messages:
        if m.get("type") != "message":
            continue

        msg_id = int(m["id"])
        raw_uid = parse_from_id(m.get("from_id"))
        uid = channel_owners.get(raw_uid, raw_uid) if raw_uid is not None else None

        known = KNOWN_HANDLES.get(uid, {}) if uid is not None else {}
        db_u = db_users.get(uid, {}) if uid is not None else {}

        export_from = (m.get("from") or "").strip() or None
        sender_name = (
            export_from
            or db_u.get("display_name")
            or known.get("name")
            or (f"User {uid}" if uid else "Unknown")
        )
        sender_username = db_u.get("username") or known.get("username")
        is_bot = bool(known.get("is_bot", False))

        text_body = flatten_export_text(m.get("text", ""))
        media_type, media_tag = extract_export_media(m)

        if media_tag and text_body:
            content = f"{media_tag} {text_body}"
        elif media_tag:
            content = media_tag
        else:
            content = text_body

        if not content:
            continue

        reply_to = m.get("reply_to_message_id")
        reply_to_id = int(reply_to) if reply_to is not None else None
        created_at = parse_export_timestamp(m)

        records.append(
            (
                chat_id,
                msg_id,
                uid,
                sender_name[:255],
                sender_username[:255] if sender_username else None,
                is_bot,
                reply_to_id,
                content,
                media_type,
                created_at,
                mark_digested,
            )
        )

        if uid is not None and not is_bot:
            st = user_stats[uid]
            st["count"] += 1
            st["last_name"] = sender_name
            if st["first_seen"] is None or created_at < st["first_seen"]:
                st["first_seen"] = created_at
            if st["last_seen"] is None or created_at > st["last_seen"]:
                st["last_seen"] = created_at

    # Bulk upsert in batches of 1,000 via UNNEST for fast network execution
    batch_size = 1000
    async with db.pool.acquire() as conn:
        for i in range(0, len(records), batch_size):
            chunk = records[i : i + batch_size]
            await conn.execute(
                """
                INSERT INTO group_chat_messages (
                    chat_id, message_id, user_id, sender_name, sender_username,
                    is_bot, reply_to_message_id, content, media_type, created_at, memory_digested
                )
                SELECT * FROM UNNEST(
                    $1::bigint[],
                    $2::bigint[],
                    $3::bigint[],
                    $4::varchar[],
                    $5::varchar[],
                    $6::boolean[],
                    $7::bigint[],
                    $8::text[],
                    $9::varchar[],
                    $10::timestamptz[],
                    $11::boolean[]
                )
                ON CONFLICT (chat_id, message_id) DO UPDATE
                SET user_id = COALESCE(EXCLUDED.user_id, group_chat_messages.user_id),
                    sender_name = COALESCE(EXCLUDED.sender_name, group_chat_messages.sender_name),
                    sender_username = COALESCE(EXCLUDED.sender_username, group_chat_messages.sender_username),
                    is_bot = EXCLUDED.is_bot,
                    reply_to_message_id = COALESCE(EXCLUDED.reply_to_message_id, group_chat_messages.reply_to_message_id),
                    content = EXCLUDED.content,
                    media_type = COALESCE(EXCLUDED.media_type, group_chat_messages.media_type),
                    created_at = EXCLUDED.created_at,
                    memory_digested = (group_chat_messages.memory_digested OR EXCLUDED.memory_digested)
                """,
                [r[0] for r in chunk],
                [r[1] for r in chunk],
                [r[2] for r in chunk],
                [r[3] for r in chunk],
                [r[4] for r in chunk],
                [r[5] for r in chunk],
                [r[6] for r in chunk],
                [r[7] for r in chunk],
                [r[8] for r in chunk],
                [r[9] for r in chunk],
                [r[10] for r in chunk],
            )
            logger.info(
                "Ingested %d / %d messages into group_chat_messages...",
                min(i + batch_size, len(records)),
                len(records),
            )

    return {
        "chat_id": chat_id,
        "ingested_messages": len(records),
        "user_stats": user_stats,
    }


async def seed_curated_profiles_and_memories(user_stats: dict[int, dict]) -> tuple[int, int]:
    """Upsert curated member profiles (enriched with exact export message counts/dates) and memory documents."""
    profile_count = 0
    for prof in CURATED_MEMBER_PROFILES:
        uid = int(prof["user_id"])
        st = user_stats.get(uid, {})
        await db.upsert_user_memory_profile(
            user_id=uid,
            canonical_name=prof["canonical_name"],
            username=prof.get("username"),
            known_aliases=prof.get("known_aliases"),
            bio_summary=prof.get("bio_summary"),
            personality_and_style=prof.get("personality_and_style"),
            interests=prof.get("interests"),
            notable_facts=prof.get("notable_facts"),
            message_count=st.get("count") or None,
            first_seen_at=st.get("first_seen"),
            last_seen_at=st.get("last_seen"),
        )
        profile_count += 1

    doc_count = 0
    for doc in CURATED_MEMORY_DOCUMENTS:
        kws = KeywordIndexer.build_document_keywords(
            title=doc["title"],
            summary=doc["summary"],
            content=doc["content"],
            category=doc["category"],
            subject_name=doc.get("subject_name"),
            extra_keywords=doc.get("keywords") or [],
        )
        doc_id = await db.upsert_memory_document(
            scope=doc["scope"],
            category=doc["category"],
            title=doc["title"],
            summary=doc["summary"],
            content=doc["content"],
            subject_user_id=doc.get("subject_user_id"),
            subject_name=doc.get("subject_name"),
            importance_score=float(doc.get("importance_score", 8.0)),
            confidence_score=0.98,
            source_start_msg_id=doc.get("source_start_msg_id"),
            source_end_msg_id=doc.get("source_end_msg_id"),
            keywords=kws,
        )
        if doc_id:
            doc_count += 1

    return profile_count, doc_count


async def run_digestion(export_path: Path, llm_digest: bool = False):
    if not export_path.exists():
        raise FileNotFoundError(f"Chat export not found at: {export_path}")

    await db.connect()
    try:
        logger.info("Connected to PostgreSQL (pgvector=%s). Starting ingestion...", db.has_pgvector)
        ingest_info = await bulk_ingest_export_messages(
            export_path,
            mark_digested=not llm_digest,
        )
        p_cnt, d_cnt = await seed_curated_profiles_and_memories(ingest_info["user_stats"])
        logger.info(
            "Seeded %d member profiles and %d long-term memory documents (+ keyword index).",
            p_cnt,
            d_cnt,
        )

        if get_gemini_client():
            embedded = await memory_engine.backfill_missing_embeddings(batch_size=64)
            logger.info("Embedded %d memory documents via gemini-embedding-2.", embedded)

            if llm_digest:
                chat_id = ingest_info["chat_id"]
                batches = 0
                while await memory_engine.digest_next_batch(chat_id, batch_size=90):
                    batches += 1
                    logger.info("Completed Gemini 4 Neon digestion batch #%d", batches)
                    await asyncio.sleep(1.5)

        async with db.pool.acquire() as conn:
            total_msgs = await conn.fetchval("SELECT COUNT(*) FROM group_chat_messages")
            total_profs = await conn.fetchval("SELECT COUNT(*) FROM user_memory_profiles")
            total_docs = await conn.fetchval("SELECT COUNT(*) FROM memory_documents")
            total_kws = await conn.fetchval("SELECT COUNT(*) FROM memory_keyword_index")

        logger.info(
            "Database Memory Summary -> group_chat_messages=%d | user_memory_profiles=%d | memory_documents=%d | memory_keyword_index=%d",
            total_msgs,
            total_profs,
            total_docs,
            total_kws,
        )
    finally:
        await db.disconnect()


def main():
    default_export = Path(__file__).resolve().parent / "result.json"
    parser = argparse.ArgumentParser(
        description="Ingest Telegram chat export and populate symm.city bot's long-term memory."
    )
    parser.add_argument(
        "--export-path",
        type=Path,
        default=default_export,
        help="Path to Telegram Desktop JSON export (result.json).",
    )
    parser.add_argument(
        "--llm-digest",
        action="store_true",
        help="Also run incremental Gemini 4 Neon batch extraction across all undigested slices.",
    )
    args = parser.parse_args()
    asyncio.run(run_digestion(args.export_path, llm_digest=args.llm_digest))


if __name__ == "__main__":
    main()
