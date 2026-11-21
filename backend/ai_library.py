"""
AI Knowledge Library and Bluesky Post History Search for symm.city Bot.
Integrates symmetricalboy's full Bluesky post archive with BM25 search and core community knowledge.
"""

import json
import logging
import math
import os
import re
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core Knowledge Profile
# ---------------------------------------------------------------------------
SYMM_CITY_KNOWLEDGE = """
**About symm & symm.city:**
- **Creator / Host**: symm (also known as symmetricalboy or Dylan).
- **Official Website**: https://symmetricalboy.com/
- **Community Chat**: symm.city (points to https://symm.city on Telegram).
- **Bluesky Profile**: @symmetricalboy.com
- **YouTube / Music**: symmetricalboy on YouTube, Spotify, and other streaming platforms. Produces electronic, ambient, chiptune, synth, and cyberpunk music, as well as original soundtracks.
- **Game Development (Symmetris)**:
  - Developer of **Symmetris**, a fast-paced multiplayer block-stacking / Tetris-style arcade game.
  - Built for Android (phone, tablet, Play Games for Windows, Chromebook, Android TV) with custom Bluetooth controller mapping (e.g. Taiko drums, DDR mats, gamepads).
  - Features real-time multiplayer versus modes, ranking systems, leaderboards, stats, sound effects & original music, offline AI bots, and Wi-Fi Direct local multiplayer.
- **Tech, AI & Software Interests**:
  - Passionate developer experimenting with AI (Gemini 4 Neon, Claude, Antigravity, local models, Live API, image/video generation, wearable companions like Wristy).
  - Creates tools for Bluesky (Lexicons, Alty alt-text bot, Ask Everything, FlappySky, AT Protocol tools), Wear OS apps & watchfaces, Telegram bots, and web applications (Comail, symmetricalboy.com).
  - Enjoys creative coding, robotics/automation, Android ecosystem, and UI/UX design.

**symm.city Community Rules:**
1. Be kind.
2. Assume good intent.
3. Be symmetrical.
4. Be patient.
5. Be *very* symmetrical.

**symm.city Bot Features:**
- **AI Assistant (/ask <question> or reply to the bot)**: Powered by `gemini-4-neon` with a 250-message floating group chat context window, full chat archive search, continuous long-term member & community memories, and symm's Bluesky post archive. Replying directly to any message from the bot continues the AI conversation automatically.
- **CXP System (Community XP)**: Earn 50 CXP per chat message (1 min cooldown). Positive emoji reactions give +50 CXP; negative emoji reactions give -50 CXP. Higher-level members have higher reaction multipliers!
- **Level Titles**: Budding Symmetry (1-4), Bilateral Balance (5-9), Rotational Harmony (10-19), Crystalline Order (20-29), Exquisitely Symmetrical (30-39), Supersymmetric (40-49), Very Symmetrical (50-59), Pure Symmetry (60+).
- **CXP Commands**: `/level` (view stats & rank), `/history` (view CXP history charts, high/low points, and breakdowns across messages, reactions, gambling, stealing, trivia, and admin gives for any user or timeframe), `/leaderboard` (top community members), `/contest` (active event rankings), `/steal` (attempt to steal 25-100 CXP from a random member, 1hr cooldown), `/gamble <amount>` (5-line CXP slot machine, 1hr cooldown), `/question` (DM bot to create CXP trivia questions) & `/answer <guess>` (answer active trivia).
- **Translation Commands**: Quick translation with `/en`, `/es`, `/fr`, `/de`, `/it`, `/nl`, `/pt`, `/ru`, `/uk`, `/el`, `/sk`, `/tr`, `/fa`, `/id`, `/hi`, `/da`, `/ja`, `/ar`, or `/translate` for an interactive menu.
- **Time Zone Commands**: `/settime <location>` (save your time zone), `/time <location | @username>` (check current time anywhere or for another member; defaults to symm.city time).
- **Moderation & Alerts**: Mention `@admin` in the group to ping all administrators. Moderation tools include `/mute`, `/unmute`, `/kick`, `/ban`.
- **Channel Post Drafting**: Authorized channel admins can use `/draft` in DM to format and preview posts with custom URL buttons.
"""

# ---------------------------------------------------------------------------
# In-Memory Bluesky Post History BM25 Search Engine
# ---------------------------------------------------------------------------
class BlueskyPostArchive:
    """In-memory search engine using BM25 ranking over symmetricalboy's Bluesky export."""

    def __init__(self):
        self.posts = []
        self.doc_lengths = []
        self.avg_doc_len = 0
        self.doc_count = 0
        self.inverted_index = {}  # term -> list of (doc_index, term_freq)
        self.idf = {}  # term -> idf score
        self.is_loaded = False
        self._load_posts()

    def _tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        return re.findall(r"\w+", text.lower())

    def _find_json_path(self) -> Path | None:
        """Find the Bluesky posts JSON file in standard locations."""
        filename = "symmetricalboy.com_posts_20260822_224732.json"
        custom_path = os.getenv("BLUESKY_POSTS_JSON")
        if custom_path and Path(custom_path).exists():
            return Path(custom_path)

        candidates = [
            Path(filename),
            Path(__file__).parent / filename,
            Path(__file__).parent.parent / filename,
            Path.cwd() / filename,
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def _load_posts(self):
        json_path = self._find_json_path()
        if not json_path:
            logger.warning("Bluesky post history JSON not found. Search will be inactive.")
            return

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                raw_posts = json.load(f)

            self.posts = []
            for item in raw_posts:
                text = item.get("text", "")
                alt_texts = [
                    img.get("alt_text", "")
                    for img in item.get("images", [])
                    if img.get("alt_text")
                ]
                full_text = text
                if alt_texts:
                    full_text += " [Image Alt: " + " | ".join(alt_texts) + "]"

                self.posts.append({
                    "created_at": item.get("created_at", ""),
                    "text": text,
                    "full_text": full_text,
                })

            self.doc_count = len(self.posts)
            if self.doc_count == 0:
                return

            # Build Inverted Index and calculate Doc Lengths
            total_len = 0
            doc_term_freqs = []
            doc_freq = {}

            for doc_idx, p in enumerate(self.posts):
                tokens = self._tokenize(p["full_text"])
                doc_len = len(tokens)
                self.doc_lengths.append(doc_len)
                total_len += doc_len

                tf_map = {}
                for t in tokens:
                    tf_map[t] = tf_map.get(t, 0) + 1
                doc_term_freqs.append(tf_map)

                for t in tf_map.keys():
                    doc_freq[t] = doc_freq.get(t, 0) + 1

            self.avg_doc_len = total_len / self.doc_count if self.doc_count else 1

            # Compute BM25 IDF
            for term, df in doc_freq.items():
                # Standard Lucene/BM25 IDF formula
                self.idf[term] = math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))

            # Build inverted index
            for doc_idx, tf_map in enumerate(doc_term_freqs):
                for term, tf in tf_map.items():
                    if term not in self.inverted_index:
                        self.inverted_index[term] = []
                    self.inverted_index[term].append((doc_idx, tf))

            self.is_loaded = True
            logger.info("Successfully loaded and indexed %d Bluesky posts.", self.doc_count)

        except Exception as e:
            logger.error("Failed to load Bluesky posts archive: %s", e)

    def search(self, query: str, top_k: int = 15) -> list[dict]:
        """Search the post history using BM25 scoring."""
        if not self.is_loaded or not query:
            return []

        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        k1 = 1.5
        b = 0.75
        scores = {}

        # Check exact phrase match boost
        clean_query = query.strip().lower()

        for term in q_tokens:
            if term not in self.inverted_index:
                continue
            term_idf = self.idf.get(term, 0)
            postings = self.inverted_index[term]

            for doc_idx, tf in postings:
                doc_len = self.doc_lengths[doc_idx]
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / self.avg_doc_len))
                score = term_idf * (numerator / denominator)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + score

        # Apply exact phrase bonus if query has multiple words
        if len(q_tokens) > 1:
            for doc_idx in list(scores.keys()):
                if clean_query in self.posts[doc_idx]["text"].lower():
                    scores[doc_idx] += 8.0

        if not scores:
            return []

        # Sort by score descending
        sorted_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_idx, score in sorted_indices:
            p = self.posts[doc_idx]
            results.append({
                "created_at": p["created_at"],
                "text": p["text"],
                "score": round(score, 2),
            })
        return results

    def get_recent(self, limit: int = 10) -> list[dict]:
        """Get the most recent posts."""
        if not self.is_loaded:
            return []
        return [
            {"created_at": p["created_at"], "text": p["text"]}
            for p in self.posts[:limit]
        ]


# Singleton instance of post archive
archive = BlueskyPostArchive()


def get_relevant_context(question: str) -> str:
    """Retrieve grounded context combining core knowledge and matching Bluesky posts."""
    results = archive.search(question, top_k=15)

    posts_section = ""
    if results:
        formatted_posts = []
        for r in results:
            date_str = r["created_at"][:10] if r.get("created_at") else "Recent"
            formatted_posts.append(f"- [{date_str}] {r['text']}")
        posts_section = "### Relevant Posts & Statements by symm (@symmetricalboy.com):\n" + "\n".join(formatted_posts)
    else:
        # If no specific match, inject a few recent posts
        recent = archive.get_recent(limit=5)
        if recent:
            formatted_posts = [f"- [{r['created_at'][:10]}] {r['text']}" for r in recent]
            posts_section = "### Recent Posts by symm (@symmetricalboy.com):\n" + "\n".join(formatted_posts)

    return f"{SYMM_CITY_KNOWLEDGE}\n\n{posts_section}".strip()


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------
ANSWER_GENERATION_PROMPT = """
You are **symm.city bot**, the AI assistant for the **symm.city** Telegram community (hosted at https://symm.city), founded by **symm** (symmetricalboy / Dylan).

Your goal is to assist community members by accurately answering questions about symm, his creative projects (like Symmetris, music, software), his thoughts from his Bluesky post history, the symm.city community, and the bot's features.

Tone & Persona:
- Warm, witty, intelligent, chill, direct, and helpful.
- Channel symm's authentic perspective: enthusiast of indie games, creative tech, AI, good music, clean coding, and community vibes.
- Never make up false facts about symm. Base your knowledge on the provided context and post history.
- If answering general questions (coding, tech, trivia, bot help), provide clear, insightful, and well-structured answers.

DOCUMENTATION & POST HISTORY CONTEXT:
{context}

RULES FOR ANSWERING:
1. Base answers about symm, Symmetris, his music, and his opinions on the provided context and Bluesky posts above.
2. Keep answers concise, engaging, and clear. Avoid unnecessary fluff.
3. If asked about the bot's features, explain the relevant commands (e.g. `/level`, `/leaderboard`, `/ask`, `/settime`, `/time`, `/translate`, etc.).
4. YOUR RESPONSE MUST NEVER EXCEED 3500 CHARACTERS.
5. ALWAYS USE HTML FORMATTING FOR EMPHASIS. You MUST use ONLY the following allowed Telegram HTML tags:
   - <b>bold</b>, <i>italic</i>, <u>underline</u>, <s>strikethrough</s>
   - <tg-spoiler>spoiler</tg-spoiler>
   - <code>monospace string</code>, <pre>monospace block</pre>
   - <a href="http://www.example.com/">embedded link</a>
   - <blockquote>quote</blockquote>
   DO NOT use Markdown (like **bold** or *italic*). ONLY use the HTML tags listed above.
6. IMPORTANT: Reply in the EXACT same language that the user used to ask their question.

User Inquiry:
User: {username}
Question: {question}
"""
