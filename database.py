import asyncpg
import json
import os
import logging
import time
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


class Database:
    def __init__(self):
        self.pool = None
        self._config_cache = None
        self._config_cache_time = 0
        self.has_pgvector = False

    async def connect(self):
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL is not set in the environment.")

        self.pool = await asyncpg.create_pool(database_url)
        await self.init_db()
        logger.info("Connected to the database successfully.")

    async def disconnect(self):
        if self.pool:
            await self.pool.close()

    async def init_db(self):
        if not self.pool:
            return

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bot_config (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    main_group_id BIGINT,
                    channel_id BIGINT,
                    admin_group_id BIGINT,
                    welcome_message TEXT
                );
                
                ALTER TABLE bot_config ADD COLUMN IF NOT EXISTS cxp_topic_id BIGINT;
                ALTER TABLE bot_config ADD COLUMN IF NOT EXISTS channel_forward_topic_id BIGINT;
                ALTER TABLE bot_config ADD COLUMN IF NOT EXISTS contest_start DATE;
                ALTER TABLE bot_config ADD COLUMN IF NOT EXISTS contest_end DATE;
                ALTER TABLE bot_config ADD COLUMN IF NOT EXISTS rules_message TEXT;
                
                CREATE TABLE IF NOT EXISTS users (
                    user_id BIGINT PRIMARY KEY,
                    cxp INTEGER DEFAULT 0,
                    last_message_time TIMESTAMP,
                    username VARCHAR(255),
                    is_admin BOOLEAN DEFAULT FALSE
                );
                
                ALTER TABLE users ADD COLUMN IF NOT EXISTS username VARCHAR(255);
                ALTER TABLE users ADD COLUMN IF NOT EXISTS display_name VARCHAR(255);
                ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN DEFAULT FALSE;
                ALTER TABLE users ADD COLUMN IF NOT EXISTS is_channel_admin BOOLEAN DEFAULT FALSE;
                ALTER TABLE users ADD COLUMN IF NOT EXISTS is_blacklisted BOOLEAN DEFAULT FALSE;
                ALTER TABLE users ADD COLUMN IF NOT EXISTS last_steal_time TIMESTAMP;
                ALTER TABLE users ADD COLUMN IF NOT EXISTS last_gamble_time TIMESTAMP;
                ALTER TABLE users ADD COLUMN IF NOT EXISTS location VARCHAR(255);

                CREATE TABLE IF NOT EXISTS messages (
                    chat_id BIGINT,
                    message_id BIGINT,
                    user_id BIGINT,
                    PRIMARY KEY (chat_id, message_id)
                );
                
                CREATE TABLE IF NOT EXISTS translations (
                    chat_id BIGINT,
                    message_id BIGINT,
                    original_text TEXT,
                    lang_en TEXT,
                    lang_es TEXT,
                    lang_fr TEXT,
                    lang_pt TEXT,
                    lang_id TEXT,
                    lang_fa TEXT,
                    lang_ru TEXT,
                    lang_uk TEXT,
                    lang_tr TEXT,
                    lang_nl TEXT,
                    lang_el TEXT,
                    lang_sk TEXT,
                    lang_de TEXT,
                    lang_it TEXT,
                    lang_hi TEXT,
                    lang_da TEXT,
                    lang_ja TEXT,
                    lang_ar TEXT,
                    PRIMARY KEY (chat_id, message_id)
                );
                
                ALTER TABLE translations ADD COLUMN IF NOT EXISTS lang_nl TEXT;
                ALTER TABLE translations ADD COLUMN IF NOT EXISTS lang_el TEXT;
                ALTER TABLE translations ADD COLUMN IF NOT EXISTS lang_sk TEXT;
                ALTER TABLE translations ADD COLUMN IF NOT EXISTS lang_de TEXT;
                ALTER TABLE translations ADD COLUMN IF NOT EXISTS lang_it TEXT;
                ALTER TABLE translations ADD COLUMN IF NOT EXISTS lang_hi TEXT;
                ALTER TABLE translations ADD COLUMN IF NOT EXISTS lang_da TEXT;
                ALTER TABLE translations ADD COLUMN IF NOT EXISTS lang_ja TEXT;
                ALTER TABLE translations ADD COLUMN IF NOT EXISTS lang_ar TEXT;
                
                CREATE TABLE IF NOT EXISTS translated_messages (
                    chat_id BIGINT,
                    message_id BIGINT,
                    original_message_id BIGINT,
                    author_id BIGINT,
                    author_name TEXT,
                    PRIMARY KEY (chat_id, message_id)
                );
                
                CREATE TABLE IF NOT EXISTS user_daily_cxp (
                    user_id BIGINT,
                    date DATE,
                    cxp_gained INTEGER DEFAULT 0,
                    PRIMARY KEY (user_id, date)
                );
                
                CREATE TABLE IF NOT EXISTS channel_links (
                    channel_id BIGINT PRIMARY KEY,
                    user_id BIGINT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS questions (
                    id SERIAL PRIMARY KEY,
                    asker_id BIGINT NOT NULL,
                    asker_name VARCHAR(255),
                    question_text TEXT NOT NULL,
                    answer_text TEXT NOT NULL,
                    prize_cxp INTEGER NOT NULL,
                    guess_cost INTEGER NOT NULL,
                    collected_cxp INTEGER DEFAULT 0,
                    guess_count INTEGER DEFAULT 0,
                    status VARCHAR(32) NOT NULL DEFAULT 'queued',
                    winner_id BIGINT,
                    group_message_id BIGINT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    closed_at TIMESTAMP
                );

                ALTER TABLE questions ALTER COLUMN answer_text TYPE TEXT;

                CREATE INDEX IF NOT EXISTS idx_questions_status_id ON questions (status, id);

                CREATE TABLE IF NOT EXISTS cxp_history (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    delta_cxp INTEGER NOT NULL,
                    balance_after INTEGER NOT NULL,
                    category VARCHAR(32) NOT NULL,
                    sub_type VARCHAR(32),
                    counterparty_id BIGINT,
                    reference_id BIGINT,
                    metadata JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_cxp_history_user_time ON cxp_history (user_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_cxp_history_user_cat_time ON cxp_history (user_id, category, created_at);

                CREATE TABLE IF NOT EXISTS group_chat_messages (
                    chat_id BIGINT NOT NULL,
                    message_id BIGINT NOT NULL,
                    thread_id BIGINT,
                    user_id BIGINT,
                    sender_name VARCHAR(255),
                    sender_username VARCHAR(255),
                    is_bot BOOLEAN DEFAULT FALSE,
                    reply_to_message_id BIGINT,
                    content TEXT NOT NULL,
                    media_type VARCHAR(64),
                    media_file_id TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    memory_digested BOOLEAN NOT NULL DEFAULT FALSE,
                    PRIMARY KEY (chat_id, message_id)
                );

                ALTER TABLE group_chat_messages ADD COLUMN IF NOT EXISTS media_file_id TEXT;

                CREATE INDEX IF NOT EXISTS idx_gcm_chat_created_desc
                    ON group_chat_messages (chat_id, created_at DESC, message_id DESC);
                CREATE INDEX IF NOT EXISTS idx_gcm_undigested
                    ON group_chat_messages (chat_id, created_at ASC, message_id ASC)
                    WHERE memory_digested = FALSE;
                CREATE INDEX IF NOT EXISTS idx_gcm_user_created
                    ON group_chat_messages (user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_gcm_fts
                    ON group_chat_messages USING GIN (to_tsvector('simple', content));

                CREATE TABLE IF NOT EXISTS user_memory_profiles (
                    user_id BIGINT PRIMARY KEY,
                    canonical_name VARCHAR(255) NOT NULL,
                    username VARCHAR(255),
                    known_aliases JSONB DEFAULT '[]'::jsonb,
                    bio_summary TEXT DEFAULT '',
                    personality_and_style TEXT DEFAULT '',
                    interests JSONB DEFAULT '[]'::jsonb,
                    notable_facts JSONB DEFAULT '[]'::jsonb,
                    message_count INTEGER DEFAULT 0,
                    first_seen_at TIMESTAMPTZ,
                    last_seen_at TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );

                INSERT INTO bot_config (id, welcome_message, rules_message) 
                VALUES (1, 'Welcome {mention} to symm.city! 🌆\n\n📜 <b>symm.city Community Rules</b>:\n\n1. Be kind.\n2. Assume good intent.\n3. Be symmetrical.\n4. Be patient.\n5. Be <i>very</i> symmetrical.\n\nWe''re glad to have you here! 💚', '📜 <b>symm.city Community Rules</b>:\n\n1. Be kind.\n2. Assume good intent.\n3. Be symmetrical.\n4. Be patient.\n5. Be <i>very</i> symmetrical.') 
                ON CONFLICT (id) DO NOTHING;
            """
            )

            # Initialize pgvector extension and long-term memory tables
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                self.has_pgvector = True
            except Exception as e:
                logger.warning("pgvector extension unavailable, falling back to JSONB embeddings: %s", e)
                self.has_pgvector = False

            emb_col_type = "vector(768)" if self.has_pgvector else "JSONB"
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS memory_documents (
                    id BIGSERIAL PRIMARY KEY,
                    scope VARCHAR(16) NOT NULL DEFAULT 'USER',
                    subject_user_id BIGINT,
                    subject_name VARCHAR(255),
                    category VARCHAR(32) NOT NULL,
                    title VARCHAR(255) NOT NULL,
                    summary TEXT NOT NULL,
                    content TEXT NOT NULL,
                    attributes JSONB DEFAULT '{{}}'::jsonb,
                    importance_score DOUBLE PRECISION DEFAULT 5.0,
                    confidence_score DOUBLE PRECISION DEFAULT 0.9,
                    source_start_msg_id BIGINT,
                    source_end_msg_id BIGINT,
                    embedding {emb_col_type},
                    last_accessed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_mem_docs_subject_cat
                    ON memory_documents (subject_user_id, category);
                CREATE INDEX IF NOT EXISTS idx_mem_docs_scope_imp
                    ON memory_documents (scope, importance_score DESC, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_mem_docs_subject_title
                    ON memory_documents (COALESCE(subject_user_id, 0), LOWER(title));

                CREATE TABLE IF NOT EXISTS memory_keyword_index (
                    id BIGSERIAL PRIMARY KEY,
                    document_id BIGINT NOT NULL REFERENCES memory_documents(id) ON DELETE CASCADE,
                    subject_user_id BIGINT,
                    keyword_or_phrase VARCHAR(128) NOT NULL,
                    weight DOUBLE PRECISION DEFAULT 1.0,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_mem_kw_phrase
                    ON memory_keyword_index (keyword_or_phrase);
                CREATE INDEX IF NOT EXISTS idx_mem_kw_doc
                    ON memory_keyword_index (document_id);
                CREATE INDEX IF NOT EXISTS idx_mem_kw_user_phrase
                    ON memory_keyword_index (subject_user_id, keyword_or_phrase);
                """
            )
            if self.has_pgvector:
                try:
                    await conn.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_mem_docs_embedding_hnsw
                        ON memory_documents USING hnsw (embedding vector_cosine_ops);
                        """
                    )
                except Exception as e:
                    logger.warning("Could not create HNSW vector index: %s", e)

    async def update_user_steal_time(self, user_id: int):
        if not self.pool:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO users (user_id, last_steal_time)
                VALUES ($1, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id) DO UPDATE
                SET last_steal_time = CURRENT_TIMESTAMP
                """,
                user_id,
            )

    async def get_config(self):
        if not self.pool:
            return None
        if self._config_cache and time.time() - self._config_cache_time < 60:
            return self._config_cache

        async with self.pool.acquire() as conn:
            config = await conn.fetchrow("SELECT * FROM bot_config WHERE id = 1")
            self._config_cache = config
            self._config_cache_time = time.time()
            return config

    async def update_config(self, **kwargs):
        if not self.pool:
            logger.error("update_config returning False because self.pool is None")
            return False

        valid_keys = {
            "main_group_id",
            "channel_id",
            "admin_group_id",
            "welcome_message",
            "rules_message",
            "cxp_topic_id",
            "channel_forward_topic_id",
            "contest_start",
            "contest_end",
        }
        updates = []
        values = []

        for k, v in kwargs.items():
            if k in valid_keys:
                values.append(v)
                updates.append(f"{k} = ${len(values)}")
            else:
                logger.warning(f"update_config ignoring invalid key: {k}")

        if not updates:
            logger.error(
                f"update_config returning False because updates is empty. kwargs: {kwargs}"
            )
            return False

        set_clause = ", ".join(updates)
        query = f"UPDATE bot_config SET {set_clause} WHERE id = 1"
        logger.info(f"update_config executing: {query} with values: {values}")

        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(query, *values)
                logger.info(f"update_config execute result: {result}")
                if result == "UPDATE 0":
                    raise Exception("Database configuration row id=1 is missing.")
                self._config_cache = None
                return True
        except Exception as e:
            logger.error(f"update_config encountered an unexpected error: {e}")
            raise

    async def get_user(self, user_id):
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            user = await conn.fetchrow(
                "SELECT * FROM users WHERE user_id = $1", user_id
            )
            if not user:
                await conn.execute(
                    "INSERT INTO users (user_id) VALUES ($1) ON CONFLICT (user_id) DO NOTHING",
                    user_id,
                )
                user = await conn.fetchrow(
                    "SELECT * FROM users WHERE user_id = $1", user_id
                )
            return user

    async def get_user_by_username(self, username: str):
        if not self.pool:
            return None
        # remove @ if present and strip whitespace
        clean = username.lstrip("@").strip()
        if not clean:
            return None
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(
                """
                SELECT * FROM users 
                WHERE username ILIKE $1 
                   OR username ILIKE ('@' || $1)
                   OR display_name ILIKE $1
                   OR display_name ILIKE ('%' || $1 || '%')
                ORDER BY (username ILIKE $1) DESC, (username ILIKE ('@' || $1)) DESC, (display_name ILIKE $1) DESC
                LIMIT 1
                """,
                clean,
            )

    async def get_random_user(self, exclude_user_id: int):
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(
                """
                SELECT u.* FROM users u
                LEFT JOIN channel_links cl ON u.user_id = cl.channel_id
                WHERE u.user_id != $1
                  AND u.user_id > 0
                  AND u.user_id NOT IN (777000, 136817688, 1087968824)
                  AND cl.channel_id IS NULL
                  AND u.cxp >= 1
                ORDER BY RANDOM()
                LIMIT 1
                """,
                exclude_user_id,
            )

    async def update_user_username(self, user_id: int, username: str):
        if not self.pool or not username:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO users (user_id, username)
                VALUES ($1, $2)
                ON CONFLICT (user_id) DO UPDATE
                SET username = $2
                """,
                user_id,
                username,
            )

    async def update_user_display_name(
        self, user_id: int, display_name: str, username: str = None
    ):
        if not self.pool or not display_name:
            return
        async with self.pool.acquire() as conn:
            if username:
                await conn.execute(
                    """
                    INSERT INTO users (user_id, display_name, username)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (user_id) DO UPDATE
                    SET display_name = $2, username = $3
                    """,
                    user_id,
                    display_name,
                    username,
                )
            else:
                await conn.execute(
                    """
                    INSERT INTO users (user_id, display_name)
                    VALUES ($1, $2)
                    ON CONFLICT (user_id) DO UPDATE
                    SET display_name = $2
                    """,
                    user_id,
                    display_name,
                )

    async def get_user_location(self, user_id: int):
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT location FROM users WHERE user_id = $1", user_id
            )

    async def update_user_location(self, user_id: int, location: str):
        if not self.pool or not location:
            return
        # Basic input validation, ensuring it's not too long
        location = location[:255]
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO users (user_id, location)
                VALUES ($1, $2)
                ON CONFLICT (user_id) DO UPDATE
                SET location = $2
                """,
                user_id,
                location,
            )

    async def update_user_admin_status(self, user_id: int, is_admin: bool):
        if not self.pool:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO users (user_id, is_admin)
                VALUES ($1, $2)
                ON CONFLICT (user_id) DO UPDATE
                SET is_admin = $2
                """,
                user_id,
                is_admin,
            )

    async def get_admin_users(self):
        if not self.pool:
            return []
        async with self.pool.acquire() as conn:
            return await conn.fetch("SELECT * FROM users WHERE is_admin = TRUE")

    async def update_user_channel_admin_status(self, user_id: int, is_channel_admin: bool):
        if not self.pool:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO users (user_id, is_channel_admin)
                VALUES ($1, $2)
                ON CONFLICT (user_id) DO UPDATE
                SET is_channel_admin = $2
                """,
                user_id,
                is_channel_admin,
            )

    async def get_channel_admins(self):
        if not self.pool:
            return []
        async with self.pool.acquire() as conn:
            return await conn.fetch("SELECT * FROM users WHERE is_channel_admin = TRUE")

    async def update_user_blacklist_status(self, user_id: int, is_blacklisted: bool):
        if not self.pool:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO users (user_id, is_blacklisted)
                VALUES ($1, $2)
                ON CONFLICT (user_id) DO UPDATE
                SET is_blacklisted = $2
                """,
                user_id,
                is_blacklisted,
            )

    async def get_blacklisted_users(self):
        if not self.pool:
            return []
        async with self.pool.acquire() as conn:
            return await conn.fetch("SELECT * FROM users WHERE is_blacklisted = TRUE")

    async def _record_cxp_history_in_tx(
        self,
        conn,
        user_id: int,
        delta_cxp: int,
        balance_after: int,
        category: str,
        sub_type: str | None = None,
        counterparty_id: int | None = None,
        reference_id: int | None = None,
        metadata: dict | None = None,
    ):
        meta_json = json.dumps(metadata) if metadata is not None else None
        await conn.execute(
            """
            INSERT INTO cxp_history (
                user_id, delta_cxp, balance_after, category, sub_type,
                counterparty_id, reference_id, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
            """,
            user_id,
            delta_cxp,
            balance_after,
            category,
            sub_type,
            counterparty_id,
            reference_id,
            meta_json,
        )

    async def update_user_cxp(
        self,
        user_id,
        delta_cxp,
        update_timestamp=False,
        category: str = "other",
        sub_type: str | None = None,
        counterparty_id: int | None = None,
        reference_id: int | None = None,
        metadata: dict | None = None,
    ):
        if not self.pool:
            return False
        utc_date = datetime.now(timezone.utc).date()
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                if update_timestamp:
                    new_cxp = await conn.fetchval(
                        """
                        INSERT INTO users (user_id, cxp, last_message_time)
                        VALUES ($1, $2, CURRENT_TIMESTAMP)
                        ON CONFLICT (user_id) DO UPDATE
                        SET cxp = users.cxp + $2, last_message_time = CURRENT_TIMESTAMP
                        RETURNING cxp
                        """,
                        user_id,
                        delta_cxp,
                    )
                else:
                    new_cxp = await conn.fetchval(
                        """
                        INSERT INTO users (user_id, cxp)
                        VALUES ($1, $2)
                        ON CONFLICT (user_id) DO UPDATE
                        SET cxp = users.cxp + $2
                        RETURNING cxp
                        """,
                        user_id,
                        delta_cxp,
                    )
                await conn.execute(
                    """
                    INSERT INTO user_daily_cxp (user_id, date, cxp_gained)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (user_id, date) DO UPDATE 
                    SET cxp_gained = user_daily_cxp.cxp_gained + $3
                    """,
                    user_id,
                    utc_date,
                    delta_cxp,
                )
                if delta_cxp != 0:
                    await self._record_cxp_history_in_tx(
                        conn,
                        user_id=user_id,
                        delta_cxp=delta_cxp,
                        balance_after=new_cxp or 0,
                        category=category,
                        sub_type=sub_type,
                        counterparty_id=counterparty_id,
                        reference_id=reference_id,
                        metadata=metadata,
                    )
            return True

    async def get_user_rank(self, cxp):
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM users WHERE cxp > $1 AND (is_blacklisted = FALSE OR is_blacklisted IS NULL)",
                cxp,
            )
            return count + 1

    async def get_leaderboard(self, limit=3, start_date=None, end_date=None):
        if not self.pool:
            return []
        async with self.pool.acquire() as conn:
            if start_date and end_date:
                return await conn.fetch(
                    """
                    SELECT u.user_id, u.is_admin, u.display_name, u.username, u.cxp AS total_cxp,
                           COALESCE(SUM(d.cxp_gained), 0) AS cxp
                    FROM users u
                    JOIN user_daily_cxp d ON u.user_id = d.user_id
                    LEFT JOIN channel_links cl ON u.user_id = cl.channel_id
                    WHERE d.date >= $1 AND d.date <= $2 AND cl.channel_id IS NULL AND (u.is_blacklisted = FALSE OR u.is_blacklisted IS NULL)
                    GROUP BY u.user_id, u.is_admin, u.display_name, u.username, u.cxp
                    HAVING COALESCE(SUM(d.cxp_gained), 0) > 0
                    ORDER BY cxp DESC
                    LIMIT $3
                    """,
                    start_date, end_date, limit
                )
            else:
                return await conn.fetch(
                    """
                    SELECT u.* FROM users u 
                    LEFT JOIN channel_links cl ON u.user_id = cl.channel_id 
                    WHERE cl.channel_id IS NULL AND (u.is_blacklisted = FALSE OR u.is_blacklisted IS NULL)
                    ORDER BY u.cxp DESC LIMIT $1
                    """, limit
                )

    async def record_message(self, chat_id, message_id, user_id):
        if not self.pool:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO messages (chat_id, message_id, user_id) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING",
                chat_id,
                message_id,
                user_id,
            )

    async def get_message_author(self, chat_id, message_id):
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT user_id FROM messages WHERE chat_id = $1 AND message_id = $2",
                chat_id,
                message_id,
            )

    async def save_original_translation_text(
        self, chat_id: int, message_id: int, text: str
    ):
        if not self.pool:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO translations (chat_id, message_id, original_text) 
                VALUES ($1, $2, $3) 
                ON CONFLICT (chat_id, message_id) DO UPDATE SET original_text = EXCLUDED.original_text
                """,
                chat_id,
                message_id,
                text,
            )

    async def get_translation(self, chat_id: int, message_id: int, lang_code: str):
        """Fetch the specific cached language text if it exists."""
        if not self.pool:
            return None

        # Valid language codes matching our strict columns
        valid_langs = {
            "en", "es", "fr", "pt", "id", "fa", "ru", "uk", "tr",
            "nl", "el", "sk", "de", "it", "hi", "da", "ja", "ar"
        }
        if lang_code not in valid_langs:
            return None

        # Dynamically build the column name (safe because it's strictly validated above)
        column_name = f"lang_{lang_code}"

        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                f"SELECT {column_name} FROM translations WHERE chat_id = $1 AND message_id = $2",
                chat_id,
                message_id,
            )

    async def get_translation_original_text(self, chat_id: int, message_id: int):
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT original_text FROM translations WHERE chat_id = $1 AND message_id = $2",
                chat_id,
                message_id,
            )

    async def save_translation(
        self, chat_id: int, message_id: int, lang_code: str, translated_text: str
    ):
        if not self.pool:
            return

        valid_langs = {
            "en", "es", "fr", "pt", "id", "fa", "ru", "uk", "tr",
            "nl", "el", "sk", "de", "it", "hi", "da", "ja", "ar"
        }
        if lang_code not in valid_langs:
            return

        column_name = f"lang_{lang_code}"
        async with self.pool.acquire() as conn:
            await conn.execute(
                f"UPDATE translations SET {column_name} = $3 WHERE chat_id = $1 AND message_id = $2",
                chat_id,
                message_id,
                translated_text,
            )

    async def link_translation(
        self,
        chat_id: int,
        message_id: int,
        original_message_id: int,
        author_id: int,
        author_name: str,
    ):
        if not self.pool:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO translated_messages (chat_id, message_id, original_message_id, author_id, author_name)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (chat_id, message_id) DO NOTHING
                """,
                chat_id,
                message_id,
                original_message_id,
                author_id,
                author_name,
            )

    async def get_translation_link(self, chat_id: int, message_id: int):
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(
                "SELECT original_message_id, author_id, author_name FROM translated_messages WHERE chat_id = $1 AND message_id = $2",
                chat_id,
                message_id,
            )

    async def link_channel(self, channel_id: int, user_id: int):
        if not self.pool:
            return
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO channel_links (channel_id, user_id) 
                    VALUES ($1, $2) 
                    ON CONFLICT (channel_id) DO UPDATE SET user_id = EXCLUDED.user_id
                    """,
                    channel_id,
                    user_id,
                )
                
                # Check if channel had existing CXP
                channel_data = await conn.fetchrow("SELECT cxp FROM users WHERE user_id = $1", channel_id)
                if channel_data and channel_data["cxp"] > 0:
                    channel_cxp = channel_data["cxp"]
                    utc_date = datetime.now(timezone.utc).date()
                    # Add to user (ensuring user row exists)
                    new_user_cxp = await conn.fetchval(
                        """
                        INSERT INTO users (user_id, cxp)
                        VALUES ($1, $2)
                        ON CONFLICT (user_id) DO UPDATE
                        SET cxp = users.cxp + $2
                        RETURNING cxp
                        """,
                        user_id,
                        channel_cxp,
                    )
                    await conn.execute(
                        """
                        INSERT INTO user_daily_cxp (user_id, date, cxp_gained)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (user_id, date) DO UPDATE
                        SET cxp_gained = user_daily_cxp.cxp_gained + $3
                        """,
                        user_id,
                        utc_date,
                        channel_cxp,
                    )
                    await self._record_cxp_history_in_tx(
                        conn,
                        user_id=user_id,
                        delta_cxp=channel_cxp,
                        balance_after=new_user_cxp or 0,
                        category="other",
                        sub_type="channel_link",
                        counterparty_id=channel_id,
                    )
                    # Reset channel cxp
                    await conn.execute("UPDATE users SET cxp = 0 WHERE user_id = $1", channel_id)
                    await self._record_cxp_history_in_tx(
                        conn,
                        user_id=channel_id,
                        delta_cxp=-channel_cxp,
                        balance_after=0,
                        category="other",
                        sub_type="channel_link_out",
                        counterparty_id=user_id,
                    )

    async def get_channel_owner(self, channel_id: int):
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT user_id FROM channel_links WHERE channel_id = $1", channel_id
            )

    async def get_user_channels(self, user_id: int):
        if not self.pool:
            return []
        async with self.pool.acquire() as conn:
            return await conn.fetch(
                "SELECT channel_id FROM channel_links WHERE user_id = $1", user_id
            )

    async def unlink_channel(self, channel_id: int):
        if not self.pool:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM channel_links WHERE channel_id = $1", channel_id
            )


    async def _adjust_cxp_in_tx(
        self,
        conn,
        user_id: int,
        delta_cxp: int,
        utc_date,
        category: str | None = None,
        sub_type: str | None = None,
        counterparty_id: int | None = None,
        reference_id: int | None = None,
        metadata: dict | None = None,
        record_zero: bool = False,
    ) -> tuple[int, int]:
        row = await conn.fetchrow("SELECT cxp FROM users WHERE user_id = $1", user_id)
        old_cxp = (row["cxp"] or 0) if row else 0
        if delta_cxp == 0:
            if category and record_zero:
                await self._record_cxp_history_in_tx(
                    conn,
                    user_id=user_id,
                    delta_cxp=0,
                    balance_after=old_cxp,
                    category=category,
                    sub_type=sub_type,
                    counterparty_id=counterparty_id,
                    reference_id=reference_id,
                    metadata=metadata,
                )
            return old_cxp, old_cxp

        new_cxp = await conn.fetchval(
            "UPDATE users SET cxp = cxp + $2 WHERE user_id = $1 RETURNING cxp",
            user_id,
            delta_cxp,
        )
        await conn.execute(
            """
            INSERT INTO user_daily_cxp (user_id, date, cxp_gained)
            VALUES ($1, $2, $3)
            ON CONFLICT (user_id, date) DO UPDATE
            SET cxp_gained = user_daily_cxp.cxp_gained + $3
            """,
            user_id,
            utc_date,
            delta_cxp,
        )
        if category:
            await self._record_cxp_history_in_tx(
                conn,
                user_id=user_id,
                delta_cxp=delta_cxp,
                balance_after=new_cxp,
                category=category,
                sub_type=sub_type,
                counterparty_id=counterparty_id,
                reference_id=reference_id,
                metadata=metadata,
            )
        return old_cxp, new_cxp

    async def get_gamble_cooldown_remaining(self, user_id: int) -> int:
        if not self.pool:
            return 0
        async with self.pool.acquire() as conn:
            elapsed = await conn.fetchval(
                """
                SELECT EXTRACT(EPOCH FROM (LOCALTIMESTAMP - last_gamble_time))
                FROM users
                WHERE user_id = $1
                """,
                user_id,
            )
            if elapsed is not None and elapsed < 3600:
                return max(1, int(3600 - elapsed))
            return 0

    async def place_bet_and_settle(
        self,
        user_id: int,
        bet: int,
        payout: int,
        spin_meta: dict | None = None,
    ) -> dict:
        if not self.pool:
            return {"ok": False, "reason": "db_unavailable"}

        utc_date = datetime.now(timezone.utc).date()
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO users (user_id) VALUES ($1) ON CONFLICT (user_id) DO NOTHING",
                    user_id,
                )
                row = await conn.fetchrow(
                    "SELECT cxp, last_gamble_time FROM users WHERE user_id = $1 FOR UPDATE",
                    user_id,
                )
                if row and row["last_gamble_time"] is not None:
                    elapsed = await conn.fetchval(
                        "SELECT EXTRACT(EPOCH FROM (LOCALTIMESTAMP - $1::timestamp))",
                        row["last_gamble_time"],
                    )
                    if elapsed is not None and elapsed < 3600:
                        return {
                            "ok": False,
                            "reason": "cooldown",
                            "remaining_seconds": max(1, int(3600 - elapsed)),
                        }

                old_cxp = (row["cxp"] or 0) if row else 0
                if old_cxp < bet:
                    return {
                        "ok": False,
                        "reason": "insufficient_cxp",
                        "current_cxp": old_cxp,
                    }

                net_delta = payout - bet
                meta = {"bet": bet, "payout": payout}
                if spin_meta:
                    meta.update(spin_meta)
                _, new_cxp = await self._adjust_cxp_in_tx(
                    conn,
                    user_id,
                    net_delta,
                    utc_date,
                    category="gamble",
                    sub_type="win" if payout > 0 else "loss",
                    metadata=meta,
                    record_zero=True,
                )
                await conn.execute(
                    "UPDATE users SET last_gamble_time = LOCALTIMESTAMP WHERE user_id = $1",
                    user_id,
                )
                return {
                    "ok": True,
                    "old_cxp": old_cxp,
                    "new_cxp": new_cxp,
                    "net_delta": net_delta,
                }

    async def create_question(
        self,
        asker_id: int,
        asker_name: str,
        question_text: str,
        answer_text: str,
        prize_cxp: int,
        guess_cost: int,
    ) -> dict:
        if not self.pool:
            return {"ok": False, "reason": "db_unavailable"}

        utc_date = datetime.now(timezone.utc).date()
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock(777001)")
                await conn.execute(
                    "INSERT INTO users (user_id) VALUES ($1) ON CONFLICT (user_id) DO NOTHING",
                    asker_id,
                )
                row = await conn.fetchrow(
                    "SELECT cxp FROM users WHERE user_id = $1 FOR UPDATE", asker_id
                )
                old_cxp = (row["cxp"] or 0) if row else 0
                if old_cxp < prize_cxp:
                    return {
                        "ok": False,
                        "reason": "insufficient_cxp",
                        "current_cxp": old_cxp,
                    }

                _, new_cxp = await self._adjust_cxp_in_tx(
                    conn, asker_id, -prize_cxp, utc_date
                )

                active_id = await conn.fetchval(
                    "SELECT id FROM questions WHERE status = 'active' LIMIT 1"
                )
                if active_id is None:
                    status = "active"
                    queue_position = 0
                else:
                    status = "queued"
                    queued_count = await conn.fetchval(
                        "SELECT COUNT(*) FROM questions WHERE status = 'queued'"
                    )
                    queue_position = (queued_count or 0) + 1

                q_row = await conn.fetchrow(
                    """
                    INSERT INTO questions (
                        asker_id, asker_name, question_text, answer_text, prize_cxp, guess_cost, status
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING *
                    """,
                    asker_id,
                    (asker_name or "")[:255],
                    question_text,
                    answer_text,
                    prize_cxp,
                    guess_cost,
                    status,
                )
                await self._record_cxp_history_in_tx(
                    conn,
                    user_id=asker_id,
                    delta_cxp=-prize_cxp,
                    balance_after=new_cxp,
                    category="trivia",
                    sub_type="escrow_prize",
                    reference_id=q_row["id"],
                    metadata={"prize_cxp": prize_cxp, "guess_cost": guess_cost},
                )
                return {
                    "ok": True,
                    "question": dict(q_row),
                    "became_active": status == "active",
                    "queue_position": queue_position,
                    "remaining_cxp": new_cxp,
                }

    async def set_question_message_id(self, question_id: int, group_message_id: int):
        if not self.pool:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE questions SET group_message_id = $2 WHERE id = $1",
                question_id,
                group_message_id,
            )

    async def get_active_question(self):
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM questions WHERE status = 'active' ORDER BY id ASC LIMIT 1"
            )
            return dict(row) if row else None

    async def get_queued_questions_count(self) -> int:
        if not self.pool:
            return 0
        async with self.pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM questions WHERE status = 'queued'"
            )
            return count or 0

    async def submit_question_answer(
        self, guesser_id: int, normalized_guess: str
    ) -> dict:
        if not self.pool:
            return {"ok": False, "reason": "db_unavailable"}

        utc_date = datetime.now(timezone.utc).date()
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock(777001)")
                q = await conn.fetchrow(
                    "SELECT * FROM questions WHERE status = 'active' ORDER BY id ASC LIMIT 1 FOR UPDATE"
                )
                if not q:
                    return {"ok": False, "reason": "no_active_question"}

                q_dict = dict(q)
                asker_id = q_dict["asker_id"]
                if asker_id == guesser_id:
                    return {
                        "ok": False,
                        "reason": "self_answer",
                        "question": q_dict,
                    }

                for uid in sorted({guesser_id, asker_id}):
                    await conn.execute(
                        "INSERT INTO users (user_id) VALUES ($1) ON CONFLICT (user_id) DO NOTHING",
                        uid,
                    )
                locked_balances = {}
                for uid in sorted({guesser_id, asker_id}):
                    urow = await conn.fetchrow(
                        "SELECT cxp FROM users WHERE user_id = $1 FOR UPDATE", uid
                    )
                    locked_balances[uid] = (urow["cxp"] or 0) if urow else 0

                guesser_old_cxp = locked_balances[guesser_id]
                guess_cost = q_dict["guess_cost"]
                if guess_cost > 0 and guesser_old_cxp < guess_cost:
                    return {
                        "ok": False,
                        "reason": "insufficient_cxp",
                        "current_cxp": guesser_old_cxp,
                        "guess_cost": guess_cost,
                        "question": q_dict,
                    }

                new_collected = (q_dict["collected_cxp"] or 0) + guess_cost
                new_guess_count = (q_dict["guess_count"] or 0) + 1
                accepted_answers = [
                    a.strip()
                    for a in (q_dict["answer_text"] or "").split(",")
                    if a.strip()
                ]
                is_correct = normalized_guess in accepted_answers

                if not is_correct:
                    _, guesser_new_cxp = await self._adjust_cxp_in_tx(
                        conn,
                        guesser_id,
                        -guess_cost,
                        utc_date,
                        category="trivia",
                        sub_type="wrong_guess",
                        counterparty_id=asker_id,
                        reference_id=q_dict["id"],
                        metadata={"guess_cost": guess_cost},
                        record_zero=True,
                    )
                    await conn.execute(
                        "UPDATE questions SET collected_cxp = $2, guess_count = $3 WHERE id = $1",
                        q_dict["id"],
                        new_collected,
                        new_guess_count,
                    )
                    return {
                        "ok": True,
                        "correct": False,
                        "guess_cost": guess_cost,
                        "guesser_old_cxp": guesser_old_cxp,
                        "guesser_new_cxp": guesser_new_cxp,
                        "collected_cxp": new_collected,
                        "guess_count": new_guess_count,
                        "question": q_dict,
                    }

                prize_cxp = q_dict["prize_cxp"]
                _, guesser_new_cxp = await self._adjust_cxp_in_tx(
                    conn,
                    guesser_id,
                    prize_cxp - guess_cost,
                    utc_date,
                    category="trivia",
                    sub_type="win",
                    counterparty_id=asker_id,
                    reference_id=q_dict["id"],
                    metadata={"prize_cxp": prize_cxp, "guess_cost": guess_cost},
                    record_zero=True,
                )
                asker_old_cxp, asker_new_cxp = await self._adjust_cxp_in_tx(
                    conn,
                    asker_id,
                    new_collected,
                    utc_date,
                    category="trivia" if new_collected > 0 else None,
                    sub_type="collect_fees",
                    counterparty_id=guesser_id,
                    reference_id=q_dict["id"],
                    metadata={
                        "collected_cxp": new_collected,
                        "guess_count": new_guess_count,
                    },
                )
                await conn.execute(
                    """
                    UPDATE questions
                    SET status = 'answered',
                        collected_cxp = $2,
                        guess_count = $3,
                        winner_id = $4,
                        closed_at = CURRENT_TIMESTAMP
                    WHERE id = $1
                    """,
                    q_dict["id"],
                    new_collected,
                    new_guess_count,
                    guesser_id,
                )

                next_row = await conn.fetchrow(
                    "SELECT * FROM questions WHERE status = 'queued' ORDER BY id ASC LIMIT 1 FOR UPDATE"
                )
                next_q = None
                if next_row:
                    updated_next = await conn.fetchrow(
                        "UPDATE questions SET status = 'active' WHERE id = $1 RETURNING *",
                        next_row["id"],
                    )
                    next_q = dict(updated_next)

                return {
                    "ok": True,
                    "correct": True,
                    "guess_cost": guess_cost,
                    "prize_cxp": prize_cxp,
                    "collected_cxp": new_collected,
                    "guess_count": new_guess_count,
                    "guesser_old_cxp": guesser_old_cxp,
                    "guesser_new_cxp": guesser_new_cxp,
                    "asker_old_cxp": asker_old_cxp,
                    "asker_new_cxp": asker_new_cxp,
                    "matched_answer": normalized_guess,
                    "accepted_answers": accepted_answers,
                    "question": q_dict,
                    "next_question": next_q,
                }

    async def cancel_active_question(self, actor_id: int, is_admin: bool) -> dict:
        if not self.pool:
            return {"ok": False, "reason": "db_unavailable"}

        utc_date = datetime.now(timezone.utc).date()
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock(777001)")
                q = await conn.fetchrow(
                    "SELECT * FROM questions WHERE status = 'active' ORDER BY id ASC LIMIT 1 FOR UPDATE"
                )
                if not q:
                    return {"ok": False, "reason": "no_active_question"}

                q_dict = dict(q)
                asker_id = q_dict["asker_id"]
                if not is_admin and asker_id != actor_id:
                    return {
                        "ok": False,
                        "reason": "forbidden",
                        "question": q_dict,
                    }

                await conn.execute(
                    "INSERT INTO users (user_id) VALUES ($1) ON CONFLICT (user_id) DO NOTHING",
                    asker_id,
                )
                await conn.fetchrow(
                    "SELECT cxp FROM users WHERE user_id = $1 FOR UPDATE", asker_id
                )

                prize_cxp = q_dict["prize_cxp"]
                collected_cxp = q_dict["collected_cxp"] or 0
                total_return = prize_cxp + collected_cxp
                asker_old_cxp, asker_new_cxp = await self._adjust_cxp_in_tx(
                    conn,
                    asker_id,
                    total_return,
                    utc_date,
                    category="trivia",
                    sub_type="cancel_refund",
                    reference_id=q_dict["id"],
                    metadata={
                        "prize_returned": prize_cxp,
                        "collected_paid": collected_cxp,
                        "guess_count": q_dict["guess_count"] or 0,
                    },
                )

                await conn.execute(
                    "UPDATE questions SET status = 'cancelled', closed_at = CURRENT_TIMESTAMP WHERE id = $1",
                    q_dict["id"],
                )

                next_row = await conn.fetchrow(
                    "SELECT * FROM questions WHERE status = 'queued' ORDER BY id ASC LIMIT 1 FOR UPDATE"
                )
                next_q = None
                if next_row:
                    updated_next = await conn.fetchrow(
                        "UPDATE questions SET status = 'active' WHERE id = $1 RETURNING *",
                        next_row["id"],
                    )
                    next_q = dict(updated_next)

                return {
                    "ok": True,
                    "cancelled_question": q_dict,
                    "prize_returned": prize_cxp,
                    "collected_paid": collected_cxp,
                    "guess_count": q_dict["guess_count"] or 0,
                    "asker_old_cxp": asker_old_cxp,
                    "asker_new_cxp": asker_new_cxp,
                    "next_question": next_q,
                }


    async def attempt_steal(
        self, thief_id: int, desired_amount: int, shielded: bool
    ) -> dict:
        if not self.pool:
            return {"ok": False, "reason": "db_unavailable"}

        utc_date = datetime.now(timezone.utc).date()
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO users (user_id) VALUES ($1) ON CONFLICT (user_id) DO NOTHING",
                    thief_id,
                )

                elapsed = await conn.fetchval(
                    """
                    SELECT EXTRACT(EPOCH FROM (LOCALTIMESTAMP - last_steal_time))
                    FROM users
                    WHERE user_id = $1
                    """,
                    thief_id,
                )
                if elapsed is not None and elapsed < 3600:
                    return {
                        "ok": False,
                        "reason": "cooldown",
                        "remaining_seconds": max(1, int(3600 - elapsed)),
                    }

                locked_target = None
                thief_old_cxp = 0
                for _ in range(3):
                    candidate_id = await conn.fetchval(
                        """
                        SELECT u.user_id
                        FROM users u
                        LEFT JOIN channel_links cl ON u.user_id = cl.channel_id
                        WHERE u.user_id != $1
                          AND u.user_id > 0
                          AND u.user_id NOT IN (777000, 136817688, 1087968824)
                          AND cl.channel_id IS NULL
                          AND u.cxp >= 1
                        ORDER BY RANDOM()
                        LIMIT 1
                        """,
                        thief_id,
                    )
                    if candidate_id is None:
                        return {"ok": False, "reason": "no_valid_target"}

                    rows_by_id = {}
                    for uid in sorted({thief_id, candidate_id}):
                        rows_by_id[uid] = await conn.fetchrow(
                            "SELECT * FROM users WHERE user_id = $1 FOR UPDATE", uid
                        )

                    thief_row = rows_by_id[thief_id]
                    if thief_row and thief_row["last_steal_time"] is not None:
                        elapsed_locked = await conn.fetchval(
                            "SELECT EXTRACT(EPOCH FROM (LOCALTIMESTAMP - $1::timestamp))",
                            thief_row["last_steal_time"],
                        )
                        if elapsed_locked is not None and elapsed_locked < 3600:
                            return {
                                "ok": False,
                                "reason": "cooldown",
                                "remaining_seconds": max(1, int(3600 - elapsed_locked)),
                            }

                    cand_row = rows_by_id[candidate_id]
                    if cand_row and (cand_row["cxp"] or 0) >= 1:
                        locked_target = dict(cand_row)
                        thief_old_cxp = (thief_row["cxp"] or 0) if thief_row else 0
                        break

                if not locked_target:
                    return {"ok": False, "reason": "no_valid_target"}

                if shielded:
                    await conn.execute(
                        "UPDATE users SET last_steal_time = LOCALTIMESTAMP WHERE user_id = $1",
                        thief_id,
                    )
                    await self._record_cxp_history_in_tx(
                        conn,
                        user_id=thief_id,
                        delta_cxp=0,
                        balance_after=thief_old_cxp,
                        category="steal",
                        sub_type="blocked",
                        counterparty_id=locked_target["user_id"],
                    )
                    await self._record_cxp_history_in_tx(
                        conn,
                        user_id=locked_target["user_id"],
                        delta_cxp=0,
                        balance_after=locked_target["cxp"] or 0,
                        category="steal",
                        sub_type="defended",
                        counterparty_id=thief_id,
                    )
                    return {
                        "ok": True,
                        "shielded": True,
                        "target": locked_target,
                    }

                target_id = locked_target["user_id"]
                target_available = locked_target["cxp"] or 0
                actual_stolen = min(desired_amount, target_available)

                target_old_cxp, target_new_cxp = await self._adjust_cxp_in_tx(
                    conn,
                    target_id,
                    -actual_stolen,
                    utc_date,
                    category="steal",
                    sub_type="robbed",
                    counterparty_id=thief_id,
                    metadata={"capped": actual_stolen < desired_amount},
                )
                _, thief_new_cxp = await self._adjust_cxp_in_tx(
                    conn,
                    thief_id,
                    actual_stolen,
                    utc_date,
                    category="steal",
                    sub_type="stole",
                    counterparty_id=target_id,
                    metadata={
                        "desired_amount": desired_amount,
                        "capped": actual_stolen < desired_amount,
                    },
                )
                await conn.execute(
                    "UPDATE users SET last_steal_time = LOCALTIMESTAMP WHERE user_id = $1",
                    thief_id,
                )

                return {
                    "ok": True,
                    "shielded": False,
                    "stolen_amount": actual_stolen,
                    "desired_amount": desired_amount,
                    "capped": actual_stolen < desired_amount,
                    "target": locked_target,
                    "target_old_cxp": target_old_cxp,
                    "target_new_cxp": target_new_cxp,
                    "thief_old_cxp": thief_old_cxp,
                    "thief_new_cxp": thief_new_cxp,
                }

    async def transfer_cxp(
        self,
        sender_id: int,
        recipient_id: int,
        amount: int | None = None,
        *,
        transfer_all: bool = False,
        reference_id: int | None = None,
    ) -> dict:
        if not self.pool:
            return {"ok": False, "reason": "db_unavailable"}
        if sender_id == recipient_id:
            return {"ok": False, "reason": "self_transfer"}
        if not transfer_all and (amount is None or amount < 1):
            return {"ok": False, "reason": "invalid_amount"}

        utc_date = datetime.now(timezone.utc).date()
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for uid in sorted({sender_id, recipient_id}):
                    await conn.execute(
                        "INSERT INTO users (user_id) VALUES ($1) ON CONFLICT (user_id) DO NOTHING",
                        uid,
                    )

                rows_by_id = {}
                for uid in sorted({sender_id, recipient_id}):
                    rows_by_id[uid] = await conn.fetchrow(
                        "SELECT * FROM users WHERE user_id = $1 FOR UPDATE", uid
                    )

                sender_row = rows_by_id.get(sender_id)
                sender_old_cxp = (sender_row["cxp"] or 0) if sender_row else 0
                actual_amount = sender_old_cxp if transfer_all else int(amount)

                if actual_amount < 1 or sender_old_cxp < actual_amount:
                    return {
                        "ok": False,
                        "reason": "insufficient_cxp",
                        "current_cxp": sender_old_cxp,
                    }

                _, sender_new_cxp = await self._adjust_cxp_in_tx(
                    conn,
                    sender_id,
                    -actual_amount,
                    utc_date,
                    category="give",
                    sub_type="sent",
                    counterparty_id=recipient_id,
                    reference_id=reference_id,
                )
                recipient_old_cxp, recipient_new_cxp = await self._adjust_cxp_in_tx(
                    conn,
                    recipient_id,
                    actual_amount,
                    utc_date,
                    category="give",
                    sub_type="received",
                    counterparty_id=sender_id,
                    reference_id=reference_id,
                )

                return {
                    "ok": True,
                    "amount": actual_amount,
                    "sender_old_cxp": sender_old_cxp,
                    "sender_new_cxp": sender_new_cxp,
                    "recipient_old_cxp": recipient_old_cxp,
                    "recipient_new_cxp": recipient_new_cxp,
                }

    async def get_user_cxp_history_data(self, user_id: int) -> dict | None:
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            user_row = await conn.fetchrow(
                "SELECT * FROM users WHERE user_id = $1", user_id
            )
            daily_rows = await conn.fetch(
                """
                SELECT date, cxp_gained
                FROM user_daily_cxp
                WHERE user_id = $1
                ORDER BY date ASC
                """,
                user_id,
            )
            ledger_rows = await conn.fetch(
                """
                SELECT id, delta_cxp, balance_after, category, sub_type,
                       counterparty_id, reference_id, metadata, created_at
                FROM cxp_history
                WHERE user_id = $1
                ORDER BY created_at ASC, id ASC
                """,
                user_id,
            )
            question_rows = await conn.fetch(
                """
                SELECT id, asker_id, prize_cxp, guess_cost, collected_cxp,
                       guess_count, status, winner_id, created_at, closed_at
                FROM questions
                WHERE asker_id = $1 OR winner_id = $1
                ORDER BY id ASC
                """,
                user_id,
            )
            return {
                "user": dict(user_row) if user_row else None,
                "daily": [dict(r) for r in daily_rows],
                "ledger": [dict(r) for r in ledger_rows],
                "questions": [dict(r) for r in question_rows],
            }

    # =====================================================================
    # GROUP CHAT ARCHIVE & LONG-TERM CONTINUOUS MEMORY ENGINE METHODS
    # =====================================================================

    @staticmethod
    def _format_vector_literal(embedding: list[float] | None) -> str | None:
        if not embedding:
            return None
        return "[" + ",".join(f"{float(v):.7f}" for v in embedding) + "]"

    async def record_group_chat_message(
        self,
        chat_id: int,
        message_id: int,
        user_id: int | None,
        sender_name: str | None,
        sender_username: str | None,
        content: str,
        *,
        is_bot: bool = False,
        reply_to_message_id: int | None = None,
        thread_id: int | None = None,
        media_type: str | None = None,
        media_file_id: str | None = None,
        created_at: datetime | None = None,
        memory_digested: bool = False,
    ):
        if not self.pool or not content:
            return
        if created_at is None:
            created_at = datetime.now(timezone.utc)
        elif created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        clean_name = (sender_name or "")[:255] or None
        clean_uname = (sender_username or "").lstrip("@")[:255] or None
        clean_media = (media_type or "")[:64] or None
        clean_fid = (media_file_id or "").strip() or None

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO group_chat_messages (
                    chat_id, message_id, thread_id, user_id, sender_name, sender_username,
                    is_bot, reply_to_message_id, content, media_type, media_file_id, created_at, memory_digested
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ON CONFLICT (chat_id, message_id) DO UPDATE
                SET content = EXCLUDED.content,
                    sender_name = COALESCE(EXCLUDED.sender_name, group_chat_messages.sender_name),
                    sender_username = COALESCE(EXCLUDED.sender_username, group_chat_messages.sender_username),
                    media_type = COALESCE(EXCLUDED.media_type, group_chat_messages.media_type),
                    media_file_id = COALESCE(EXCLUDED.media_file_id, group_chat_messages.media_file_id)
                """,
                chat_id,
                message_id,
                thread_id,
                user_id,
                clean_name,
                clean_uname,
                is_bot,
                reply_to_message_id,
                content,
                clean_media,
                clean_fid,
                created_at,
                memory_digested,
            )

    async def get_message_media_file_ids(
        self, chat_id: int, message_ids: list[int]
    ) -> dict[int, str]:
        """Look up stored Telegram `media_file_id`s for specific `message_id`s in `chat_id`."""
        if not self.pool or not message_ids:
            return {}
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT message_id, media_file_id
                FROM group_chat_messages
                WHERE chat_id = $1
                  AND message_id = ANY($2::bigint[])
                  AND media_file_id IS NOT NULL
                """,
                chat_id,
                [int(m) for m in message_ids],
            )
            return {int(r["message_id"]): str(r["media_file_id"]) for r in rows}

    async def get_latest_group_image_file_id(
        self, chat_id: int
    ) -> tuple[int, str] | None:
        """Return `(message_id, media_file_id)` for the most recent image message in `chat_id`."""
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT message_id, media_file_id
                FROM group_chat_messages
                WHERE chat_id = $1
                  AND media_file_id IS NOT NULL
                ORDER BY created_at DESC, message_id DESC
                LIMIT 1
                """,
                chat_id,
            )
            if row and row["media_file_id"]:
                return int(row["message_id"]), str(row["media_file_id"])
            return None

    async def get_recent_group_messages(
        self,
        chat_id: int,
        limit: int = 250,
        exclude_bot_noise: bool = True,
    ) -> list[dict]:
        """
        Fetch the floating context window of the N most recent messages in `chat_id`,
        returned in chronological order (oldest -> newest).
        Filters out repetitive automated game/cooldown/stats bot output so the 250-message
        window stays dense with real conversation and AI dialogue.
        """
        if not self.pool:
            return []

        async with self.pool.acquire() as conn:
            if exclude_bot_noise:
                rows = await conn.fetch(
                    """
                    SELECT message_id, thread_id, user_id, sender_name, sender_username,
                           is_bot, reply_to_message_id, content, media_type, media_file_id, created_at
                    FROM group_chat_messages
                    WHERE chat_id = $1
                      AND NOT (
                          is_bot = TRUE AND (
                              content LIKE '🎰%'
                              OR content LIKE '🚨 JACKPOT%'
                              OR content LIKE '⏳%'
                              OR content LIKE '📊 Statistics for%'
                              OR content LIKE '📊 CXP History%'
                              OR content LIKE '🏆 %Leaderboard%'
                              OR content LIKE '🎯 %stole%'
                              OR content LIKE '🛡️ Shield detected%'
                              OR content LIKE '🎉 %has reached Level%'
                              OR content LIKE '⚠️ Unlinked Channel%'
                              OR content LIKE '🇺🇸 [EN]%'
                              OR content LIKE '🇲🇽 [ES]%'
                              OR content LIKE '🇷🇺 [RU]%'
                              OR content LIKE '🇮🇷 [FA]%'
                          )
                      )
                      AND content !~ '^/(gamble|g|bet|slots|slot|spin|steal|s|rob|yoink|heist|mug|level|lvl|xp|cxp|stats|rank|bal|balance|me|leaderboard|lb|top|ranks|leaders|history|h|hist|chart|charts|graph|ledger|pnl|ping|status|latency|pong|checkid|uid|whois)([[:space:]]|$|@)'
                    ORDER BY created_at DESC, message_id DESC
                    LIMIT $2
                    """,
                    chat_id,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT message_id, thread_id, user_id, sender_name, sender_username,
                           is_bot, reply_to_message_id, content, media_type, media_file_id, created_at
                    FROM group_chat_messages
                    WHERE chat_id = $1
                    ORDER BY created_at DESC, message_id DESC
                    LIMIT $2
                    """,
                    chat_id,
                    limit,
                )
        result = [dict(r) for r in rows]
        result.reverse()
        return result

    async def search_group_chat_messages(
        self,
        chat_id: int,
        *,
        query_str: str | None = None,
        user_filter: str | None = None,
        order: str = "newest",
        time_range: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Search across all historical & live messages in `group_chat_messages`,
        expanding top matches with ±1 neighboring message for conversational context.
        """
        if not self.pool:
            return []

        limit = max(1, min(35, int(limit or 20)))
        sort_dir = "ASC" if (order or "").lower() == "oldest" else "DESC"

        where_clauses = ["chat_id = $1"]
        params: list = [chat_id]

        # Optional user filter (by @username, display name, or numeric ID)
        if user_filter and user_filter.strip():
            uf = user_filter.strip().lstrip("@")
            if uf.lstrip("-").isdigit():
                params.append(int(uf))
                where_clauses.append(f"user_id = ${len(params)}")
            else:
                params.append(f"%{uf}%")
                p_idx = len(params)
                where_clauses.append(
                    f"(sender_name ILIKE ${p_idx} OR sender_username ILIKE ${p_idx})"
                )

        # Optional time_range filter
        now_utc = datetime.now(timezone.utc)
        if time_range and time_range.strip():
            tr = time_range.strip().lower()
            if tr == "today":
                start_t = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
                params.append(start_t)
                where_clauses.append(f"created_at >= ${len(params)}")
            elif tr == "yesterday":
                start_today = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
                start_yest = start_today - timedelta(days=1)
                params.extend([start_yest, start_today])
                where_clauses.append(
                    f"(created_at >= ${len(params) - 1} AND created_at < ${len(params)})"
                )
            elif tr in ("week", "this_week", "past_week", "7d"):
                params.append(now_utc - timedelta(days=7))
                where_clauses.append(f"created_at >= ${len(params)}")
            elif tr in ("month", "this_month", "past_month", "30d"):
                params.append(now_utc - timedelta(days=30))
                where_clauses.append(f"created_at >= ${len(params)}")
            elif tr not in ("all", "all_time"):
                for fmt in ("%Y-%m-%d", "%Y-%m"):
                    try:
                        parsed_dt = datetime.strptime(tr, fmt).replace(tzinfo=timezone.utc)
                        if fmt == "%Y-%m-%d":
                            end_dt = parsed_dt + timedelta(days=1)
                        else:
                            end_dt = parsed_dt + timedelta(days=32)
                            end_dt = end_dt.replace(day=1)
                        params.extend([parsed_dt, end_dt])
                        where_clauses.append(
                            f"(created_at >= ${len(params) - 1} AND created_at < ${len(params)})"
                        )
                        break
                    except ValueError:
                        continue

        # Keyword / phrase search
        terms: list[str] = []
        if query_str and query_str.strip():
            raw_q = query_str.strip()
            stop = {
                "the", "and", "for", "with", "about", "what", "when", "where",
                "who", "why", "how", "did", "does", "was", "were", "are", "have",
                "has", "had", "that", "this", "from", "they", "you", "your", "said",
                "say", "talk", "talked", "talking", "mention", "mentioned", "chat",
            }
            words = [w for w in raw_q.split() if len(w) >= 2 and w.lower() not in stop]
            terms = words[:6] if words else [raw_q]

            term_conds = []
            if len(terms) > 1:
                params.append(f"%{' '.join(terms)}%")
                term_conds.append(f"content ILIKE ${len(params)}")
            for t in terms:
                params.append(f"%{t}%")
                term_conds.append(f"content ILIKE ${len(params)}")
            where_clauses.append("(" + " OR ".join(term_conds) + ")")

        params.append(limit)
        sql = f"""
            SELECT message_id, user_id, sender_name, sender_username, is_bot,
                   reply_to_message_id, content, created_at
            FROM group_chat_messages
            WHERE {' AND '.join(where_clauses)}
            ORDER BY created_at {sort_dir}, message_id {sort_dir}
            LIMIT ${len(params)}
        """

        async with self.pool.acquire() as conn:
            matched_rows = await conn.fetch(sql, *params)
            if not matched_rows:
                return []

            # Expand with ±1 message_id neighborhood when searching specific keywords for context
            if terms and len(matched_rows) <= 15:
                neighbor_ids: set[int] = set()
                for r in matched_rows:
                    mid = int(r["message_id"])
                    neighbor_ids.update({mid - 1, mid, mid + 1})
                expanded = await conn.fetch(
                    """
                    SELECT message_id, user_id, sender_name, sender_username, is_bot,
                           reply_to_message_id, content, created_at
                    FROM group_chat_messages
                    WHERE chat_id = $1 AND message_id = ANY($2::bigint[])
                    ORDER BY created_at ASC, message_id ASC
                    """,
                    chat_id,
                    sorted(neighbor_ids),
                )
                return [dict(r) for r in expanded]

            out = [dict(r) for r in matched_rows]
            out.sort(key=lambda r: (r["created_at"], r["message_id"]))
            return out

    async def upsert_user_memory_profile(
        self,
        user_id: int,
        canonical_name: str,
        *,
        username: str | None = None,
        known_aliases: list[str] | None = None,
        bio_summary: str | None = None,
        personality_and_style: str | None = None,
        interests: list[str] | None = None,
        notable_facts: list[str] | None = None,
        message_count: int | None = None,
        first_seen_at: datetime | None = None,
        last_seen_at: datetime | None = None,
    ):
        if not self.pool:
            return
        async with self.pool.acquire() as conn:
            existing = await conn.fetchrow(
                "SELECT * FROM user_memory_profiles WHERE user_id = $1", user_id
            )
            if existing:
                def _merge_list(old_raw, new_items, cap=25):
                    old_list = json.loads(old_raw) if isinstance(old_raw, str) else (list(old_raw) if old_raw else [])
                    merged = list(old_list)
                    seen_low = {str(x).lower() for x in merged}
                    for item in (new_items or []):
                        s = str(item).strip()
                        if s and s.lower() not in seen_low:
                            seen_low.add(s.lower())
                            merged.append(s)
                    return merged[:cap]

                merged_aliases = _merge_list(existing["known_aliases"], known_aliases, cap=15)
                merged_interests = _merge_list(existing["interests"], interests, cap=25)
                merged_facts = _merge_list(existing["notable_facts"], notable_facts, cap=30)

                new_bio = bio_summary if bio_summary else existing["bio_summary"]
                new_style = (
                    personality_and_style
                    if personality_and_style
                    else existing["personality_and_style"]
                )
                new_uname = (username or "").lstrip("@") or existing["username"]
                new_cname = canonical_name or existing["canonical_name"]
                new_mcnt = (
                    message_count
                    if message_count is not None
                    else (existing["message_count"] or 0)
                )
                f_seen = first_seen_at or existing["first_seen_at"]
                if existing["first_seen_at"] and first_seen_at:
                    f_seen = min(existing["first_seen_at"], first_seen_at)
                l_seen = last_seen_at or existing["last_seen_at"]
                if existing["last_seen_at"] and last_seen_at:
                    l_seen = max(existing["last_seen_at"], last_seen_at)

                await conn.execute(
                    """
                    UPDATE user_memory_profiles
                    SET canonical_name = $2,
                        username = $3,
                        known_aliases = $4::jsonb,
                        bio_summary = $5,
                        personality_and_style = $6,
                        interests = $7::jsonb,
                        notable_facts = $8::jsonb,
                        message_count = $9,
                        first_seen_at = $10,
                        last_seen_at = $11,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = $1
                    """,
                    user_id,
                    new_cname[:255],
                    new_uname[:255] if new_uname else None,
                    json.dumps(merged_aliases, ensure_ascii=False),
                    new_bio or "",
                    new_style or "",
                    json.dumps(merged_interests, ensure_ascii=False),
                    json.dumps(merged_facts, ensure_ascii=False),
                    new_mcnt,
                    f_seen,
                    l_seen,
                )
            else:
                await conn.execute(
                    """
                    INSERT INTO user_memory_profiles (
                        user_id, canonical_name, username, known_aliases,
                        bio_summary, personality_and_style, interests, notable_facts,
                        message_count, first_seen_at, last_seen_at
                    )
                    VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7::jsonb, $8::jsonb, $9, $10, $11)
                    """,
                    user_id,
                    (canonical_name or f"User {user_id}")[:255],
                    (username or "").lstrip("@")[:255] or None,
                    json.dumps(known_aliases or [], ensure_ascii=False),
                    bio_summary or "",
                    personality_and_style or "",
                    json.dumps(interests or [], ensure_ascii=False),
                    json.dumps(notable_facts or [], ensure_ascii=False),
                    message_count or 0,
                    first_seen_at,
                    last_seen_at,
                )

    async def get_user_memory_profiles_by_ids(self, user_ids: list[int]) -> dict[int, dict]:
        if not self.pool or not user_ids:
            return {}
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM user_memory_profiles WHERE user_id = ANY($1::bigint[])",
                user_ids,
            )
            return {int(r["user_id"]): dict(r) for r in rows}

    async def get_all_user_memory_profiles(self, limit: int = 50) -> list[dict]:
        if not self.pool:
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM user_memory_profiles
                ORDER BY message_count DESC, updated_at DESC
                LIMIT $1
                """,
                limit,
            )
            return [dict(r) for r in rows]

    async def find_member_profiles(self, search_str: str) -> list[dict]:
        if not self.pool or not search_str:
            return []
        clean = search_str.strip().lstrip("@")
        if not clean:
            return []
        async with self.pool.acquire() as conn:
            if clean.lstrip("-").isdigit():
                rows = await conn.fetch(
                    "SELECT * FROM user_memory_profiles WHERE user_id = $1",
                    int(clean),
                )
                if rows:
                    return [dict(r) for r in rows]

            rows = await conn.fetch(
                """
                SELECT * FROM user_memory_profiles
                WHERE username ILIKE $1
                   OR canonical_name ILIKE $1
                   OR canonical_name ILIKE ('%' || $1 || '%')
                   OR known_aliases::text ILIKE ('%' || $1 || '%')
                ORDER BY (username ILIKE $1) DESC, (canonical_name ILIKE $1) DESC, message_count DESC
                LIMIT 5
                """,
                clean,
            )
            return [dict(r) for r in rows]

    async def upsert_memory_document(
        self,
        *,
        scope: str,
        category: str,
        title: str,
        summary: str,
        content: str,
        subject_user_id: int | None = None,
        subject_name: str | None = None,
        attributes: dict | None = None,
        importance_score: float = 5.0,
        confidence_score: float = 0.9,
        source_start_msg_id: int | None = None,
        source_end_msg_id: int | None = None,
        embedding: list[float] | None = None,
        keywords: list[tuple[str, float]] | None = None,
    ) -> int | None:
        """
        Insert or update a `memory_documents` row (deduplicating on `(subject_user_id, LOWER(title))`)
        and re-index its weighted keywords in `memory_keyword_index`.
        """
        if not self.pool or not title or not summary:
            return None

        clean_scope = "COMMUNITY" if subject_user_id is None else (scope or "USER").upper()
        clean_title = title.strip()[:255]
        clean_subj_name = (subject_name or "")[:255] or None
        attr_json = json.dumps(attributes or {}, ensure_ascii=False)
        imp = max(1.0, min(10.0, float(importance_score or 5.0)))
        conf = max(0.0, min(1.0, float(confidence_score or 0.9)))
        vec_lit = self._format_vector_literal(embedding)

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                existing_id = await conn.fetchval(
                    """
                    SELECT id FROM memory_documents
                    WHERE COALESCE(subject_user_id, 0) = COALESCE($1::bigint, 0)
                      AND LOWER(title) = LOWER($2)
                    LIMIT 1
                    """,
                    subject_user_id,
                    clean_title,
                )

                if existing_id:
                    if vec_lit and self.has_pgvector:
                        doc_id = await conn.fetchval(
                            """
                            UPDATE memory_documents
                            SET scope = $2,
                                subject_name = COALESCE($3, subject_name),
                                category = $4,
                                summary = $5,
                                content = $6,
                                attributes = memory_documents.attributes || $7::jsonb,
                                importance_score = GREATEST(memory_documents.importance_score, $8),
                                confidence_score = $9,
                                source_end_msg_id = COALESCE($10, source_end_msg_id),
                                embedding = $11::vector,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = $1
                            RETURNING id
                            """,
                            existing_id,
                            clean_scope,
                            clean_subj_name,
                            category,
                            summary,
                            content,
                            attr_json,
                            imp,
                            conf,
                            source_end_msg_id,
                            vec_lit,
                        )
                    else:
                        doc_id = await conn.fetchval(
                            """
                            UPDATE memory_documents
                            SET scope = $2,
                                subject_name = COALESCE($3, subject_name),
                                category = $4,
                                summary = $5,
                                content = $6,
                                attributes = memory_documents.attributes || $7::jsonb,
                                importance_score = GREATEST(memory_documents.importance_score, $8),
                                confidence_score = $9,
                                source_end_msg_id = COALESCE($10, source_end_msg_id),
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = $1
                            RETURNING id
                            """,
                            existing_id,
                            clean_scope,
                            clean_subj_name,
                            category,
                            summary,
                            content,
                            attr_json,
                            imp,
                            conf,
                            source_end_msg_id,
                        )
                else:
                    if vec_lit and self.has_pgvector:
                        doc_id = await conn.fetchval(
                            """
                            INSERT INTO memory_documents (
                                scope, subject_user_id, subject_name, category, title,
                                summary, content, attributes, importance_score, confidence_score,
                                source_start_msg_id, source_end_msg_id, embedding
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10, $11, $12, $13::vector)
                            RETURNING id
                            """,
                            clean_scope,
                            subject_user_id,
                            clean_subj_name,
                            category,
                            clean_title,
                            summary,
                            content,
                            attr_json,
                            imp,
                            conf,
                            source_start_msg_id,
                            source_end_msg_id,
                            vec_lit,
                        )
                    else:
                        doc_id = await conn.fetchval(
                            """
                            INSERT INTO memory_documents (
                                scope, subject_user_id, subject_name, category, title,
                                summary, content, attributes, importance_score, confidence_score,
                                source_start_msg_id, source_end_msg_id
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10, $11, $12)
                            RETURNING id
                            """,
                            clean_scope,
                            subject_user_id,
                            clean_subj_name,
                            category,
                            clean_title,
                            summary,
                            content,
                            attr_json,
                            imp,
                            conf,
                            source_start_msg_id,
                            source_end_msg_id,
                        )

                if doc_id and keywords:
                    await conn.execute(
                        "DELETE FROM memory_keyword_index WHERE document_id = $1",
                        doc_id,
                    )
                    dedup_kw: dict[str, float] = {}
                    for kw, w in keywords:
                        norm = (kw or "").strip().lower()[:128]
                        if norm:
                            dedup_kw[norm] = max(dedup_kw.get(norm, 0.0), float(w))
                    if dedup_kw:
                        await conn.executemany(
                            """
                            INSERT INTO memory_keyword_index (
                                document_id, subject_user_id, keyword_or_phrase, weight
                            )
                            VALUES ($1, $2, $3, $4)
                            """,
                            [
                                (doc_id, subject_user_id, kw, w)
                                for kw, w in dedup_kw.items()
                            ],
                        )
                return int(doc_id) if doc_id else None

    async def get_memories_for_user(self, user_id: int, limit: int = 15) -> list[dict]:
        if not self.pool:
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, scope, subject_user_id, subject_name, category, title,
                       summary, content, attributes, importance_score, updated_at
                FROM memory_documents
                WHERE subject_user_id = $1
                ORDER BY importance_score DESC, updated_at DESC
                LIMIT $2
                """,
                user_id,
                limit,
            )
            return [dict(r) for r in rows]

    async def retrieve_hybrid_memories(
        self,
        *,
        query_keywords: list[str],
        query_embedding: list[float] | None = None,
        focus_user_ids: list[int] | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Hybrid long-term memory retrieval combining:
          1. Fast weighted keyword inverted index (`memory_keyword_index`)
          2. `pgvector` 768-dim cosine similarity (`1 - (embedding <=> $1::vector)`)
          3. Subject boost for the asking user & any mentioned community members
          4. Always-on high-importance `DIRECTIVE`, `COMMUNITY_LORE`, and `PROJECT_KNOWLEDGE`
        """
        if not self.pool:
            return []

        scores_by_id: dict[int, float] = {}
        focus_set = set(focus_user_ids or [])

        async with self.pool.acquire() as conn:
            # 1. Lexical inverted index lookup (<5ms)
            if query_keywords:
                kw_rows = await conn.fetch(
                    """
                    SELECT document_id, SUM(weight) AS kw_score
                    FROM memory_keyword_index
                    WHERE keyword_or_phrase = ANY($1::text[])
                    GROUP BY document_id
                    ORDER BY kw_score DESC
                    LIMIT 30
                    """,
                    query_keywords[:60],
                )
                for r in kw_rows:
                    did = int(r["document_id"])
                    scores_by_id[did] = scores_by_id.get(did, 0.0) + float(r["kw_score"]) * 1.5

            # 2. Vector similarity lookup via pgvector
            vec_lit = self._format_vector_literal(query_embedding)
            if vec_lit and self.has_pgvector:
                vec_rows = await conn.fetch(
                    """
                    SELECT id, (1.0 - (embedding <=> $1::vector)) AS sim
                    FROM memory_documents
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> $1::vector ASC
                    LIMIT 25
                    """,
                    vec_lit,
                )
                for r in vec_rows:
                    did = int(r["id"])
                    sim = max(0.0, float(r["sim"] or 0.0))
                    if sim >= 0.35:
                        scores_by_id[did] = scores_by_id.get(did, 0.0) + sim * 12.0

            # 3. Persistent community directives & top lore + focused user top memories
            baseline_rows = await conn.fetch(
                """
                SELECT id, importance_score, category, subject_user_id
                FROM memory_documents
                WHERE category = 'DIRECTIVE'
                   OR (scope = 'COMMUNITY' AND importance_score >= 7.5)
                   OR (subject_user_id = ANY($1::bigint[]) AND importance_score >= 6.5)
                ORDER BY importance_score DESC, updated_at DESC
                LIMIT 20
                """,
                list(focus_set) if focus_set else [0],
            )
            for r in baseline_rows:
                did = int(r["id"])
                bonus = 6.0 if r["category"] == "DIRECTIVE" else 3.5
                if r["subject_user_id"] in focus_set:
                    bonus += 2.5
                scores_by_id[did] = scores_by_id.get(did, 0.0) + bonus

            if not scores_by_id:
                return []

            candidate_ids = sorted(
                scores_by_id.keys(), key=lambda k: scores_by_id[k], reverse=True
            )[: max(limit * 2, 30)]

            docs = await conn.fetch(
                """
                SELECT id, scope, subject_user_id, subject_name, category, title,
                       summary, content, attributes, importance_score, updated_at
                FROM memory_documents
                WHERE id = ANY($1::bigint[])
                """,
                candidate_ids,
            )

            ranked: list[tuple[float, dict]] = []
            for d in docs:
                did = int(d["id"])
                base = scores_by_id.get(did, 0.0) + float(d["importance_score"] or 5.0) * 0.45
                if d["subject_user_id"] in focus_set:
                    base += 2.0
                ranked.append((base, dict(d)))

            ranked.sort(key=lambda pair: pair[0], reverse=True)
            selected = [doc for _, doc in ranked[:limit]]

            if selected:
                sel_ids = [int(d["id"]) for d in selected]
                await conn.execute(
                    "UPDATE memory_documents SET last_accessed_at = CURRENT_TIMESTAMP WHERE id = ANY($1::bigint[])",
                    sel_ids,
                )
            return selected

    async def count_undigested_messages(self, chat_id: int) -> dict:
        if not self.pool:
            return {"count": 0, "oldest_ts": None}
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT COUNT(*) AS cnt, MIN(created_at) AS oldest_ts
                FROM group_chat_messages
                WHERE chat_id = $1 AND memory_digested = FALSE
                """,
                chat_id,
            )
            return {
                "count": int(row["cnt"] or 0) if row else 0,
                "oldest_ts": row["oldest_ts"] if row else None,
            }

    async def get_undigested_message_batch(
        self, chat_id: int, batch_size: int = 80
    ) -> list[dict]:
        if not self.pool:
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT message_id, user_id, sender_name, sender_username, is_bot,
                       reply_to_message_id, content, created_at
                FROM group_chat_messages
                WHERE chat_id = $1 AND memory_digested = FALSE
                ORDER BY created_at ASC, message_id ASC
                LIMIT $2
                """,
                chat_id,
                batch_size,
            )
            return [dict(r) for r in rows]

    async def mark_messages_digested(self, chat_id: int, message_ids: list[int]):
        if not self.pool or not message_ids:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE group_chat_messages
                SET memory_digested = TRUE
                WHERE chat_id = $1 AND message_id = ANY($2::bigint[])
                """,
                chat_id,
                message_ids,
            )

    async def get_unembedded_memory_documents(self, limit: int = 32) -> list[dict]:
        if not self.pool or not self.has_pgvector:
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, scope, subject_name, category, title, summary, content
                FROM memory_documents
                WHERE embedding IS NULL
                ORDER BY importance_score DESC, id ASC
                LIMIT $1
                """,
                limit,
            )
            return [dict(r) for r in rows]

    async def update_memory_document_embedding(
        self, doc_id: int, embedding: list[float]
    ):
        if not self.pool or not self.has_pgvector or not embedding:
            return
        vec_lit = self._format_vector_literal(embedding)
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE memory_documents SET embedding = $2::vector WHERE id = $1",
                doc_id,
                vec_lit,
            )


db = Database()
