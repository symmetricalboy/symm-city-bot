import asyncpg
import os
import logging
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class Database:
    def __init__(self):
        self.pool = None
        self._config_cache = None
        self._config_cache_time = 0

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
                    answer_text VARCHAR(255) NOT NULL,
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

                CREATE INDEX IF NOT EXISTS idx_questions_status_id ON questions (status, id);

                INSERT INTO bot_config (id, welcome_message, rules_message) 
                VALUES (1, 'Welcome {mention} to symm.city! 🌆\n\n📜 <b>symm.city Community Rules</b>:\n\n1. Be kind.\n2. Assume good intent.\n3. Be symmetrical.\n4. Be patient.\n5. Be <i>very</i> symmetrical.\n\nWe''re glad to have you here! 💚', '📜 <b>symm.city Community Rules</b>:\n\n1. Be kind.\n2. Assume good intent.\n3. Be symmetrical.\n4. Be patient.\n5. Be <i>very</i> symmetrical.') 
                ON CONFLICT (id) DO NOTHING;
            """
            )

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

    async def update_user_cxp(self, user_id, delta_cxp, update_timestamp=False):
        if not self.pool:
            return False
        utc_date = datetime.now(timezone.utc).date()
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                if update_timestamp:
                    await conn.execute(
                        """
                        INSERT INTO users (user_id, cxp, last_message_time)
                        VALUES ($1, $2, CURRENT_TIMESTAMP)
                        ON CONFLICT (user_id) DO UPDATE
                        SET cxp = users.cxp + $2, last_message_time = CURRENT_TIMESTAMP
                        """,
                        user_id,
                        delta_cxp,
                    )
                else:
                    await conn.execute(
                        """
                        INSERT INTO users (user_id, cxp)
                        VALUES ($1, $2)
                        ON CONFLICT (user_id) DO UPDATE
                        SET cxp = users.cxp + $2
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
                    # Add to user
                    await conn.execute("UPDATE users SET cxp = cxp + $2 WHERE user_id = $1", user_id, channel_cxp)
                    # Reset channel cxp
                    await conn.execute("UPDATE users SET cxp = 0 WHERE user_id = $1", channel_id)

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


    async def _adjust_cxp_in_tx(self, conn, user_id: int, delta_cxp: int, utc_date) -> tuple[int, int]:
        row = await conn.fetchrow("SELECT cxp FROM users WHERE user_id = $1", user_id)
        old_cxp = (row["cxp"] or 0) if row else 0
        if delta_cxp == 0:
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
        return old_cxp, new_cxp

    async def place_bet_and_settle(self, user_id: int, bet: int, payout: int) -> dict:
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
                    "SELECT cxp FROM users WHERE user_id = $1 FOR UPDATE", user_id
                )
                old_cxp = (row["cxp"] or 0) if row else 0
                if old_cxp < bet:
                    return {
                        "ok": False,
                        "reason": "insufficient_cxp",
                        "current_cxp": old_cxp,
                    }

                net_delta = payout - bet
                _, new_cxp = await self._adjust_cxp_in_tx(
                    conn, user_id, net_delta, utc_date
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
                    answer_text[:255],
                    prize_cxp,
                    guess_cost,
                    status,
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
                is_correct = normalized_guess == q_dict["answer_text"]

                if not is_correct:
                    _, guesser_new_cxp = await self._adjust_cxp_in_tx(
                        conn, guesser_id, -guess_cost, utc_date
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
                    conn, guesser_id, prize_cxp - guess_cost, utc_date
                )
                asker_old_cxp, asker_new_cxp = await self._adjust_cxp_in_tx(
                    conn, asker_id, new_collected, utc_date
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
                    conn, asker_id, total_return, utc_date
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
                    return {
                        "ok": True,
                        "shielded": True,
                        "target": locked_target,
                    }

                target_id = locked_target["user_id"]
                target_available = locked_target["cxp"] or 0
                actual_stolen = min(desired_amount, target_available)

                target_old_cxp, target_new_cxp = await self._adjust_cxp_in_tx(
                    conn, target_id, -actual_stolen, utc_date
                )
                _, thief_new_cxp = await self._adjust_cxp_in_tx(
                    conn, thief_id, actual_stolen, utc_date
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


db = Database()
