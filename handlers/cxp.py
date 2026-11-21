import html
import logging
import os
import math
import random
import asyncio
import calendar
from datetime import datetime, timedelta, timezone
from telegram import Update, ChatPermissions, error
from telegram.constants import ChatAction, MessageLimit, ParseMode
from telegram.ext import CallbackContext
import httpx
from telegram.helpers import escape_markdown

from database import db
from handlers.common import resolve_username, is_user_admin
from handlers.podium import fetch_user_avatar_bytes, render_podium_png

logger = logging.getLogger(__name__)

# Emoji constants
POSITIVE_EMOJIS = {
    "👍",
    "👍🏻",
    "👍🏼",
    "👍🏽",
    "👍🏾",
    "👍🏿",
    "❤️",
    "🧡",
    "💛",
    "💚",
    "💙",
    "💜",
    "🖤",
    "🤍",
    "🤎",
    "❤️‍🔥",
    "💖",
    "💝",
    "💕",
    "💞",
    "💓",
    "💗",
    "💘",
    "💟",
    "🫶",
    "🫶🏻",
    "🫶🏼",
    "🫶🏽",
    "🫶🏾",
    "🫶🏿",
    "🔥",
    "💯",
    "👏",
    "👏🏻",
    "👏🏼",
    "👏🏽",
    "👏🏾",
    "👏🏿",
    "🎉",
    "🎊",
    "🤩",
    "😍",
    "🥰",
    "😘",
    "🤯",
    "🥳",
    "😎",
    "😇",
    "🥹",
    "🤤",
    "👑",
    "⭐",
    "🌟",
    "✨",
    "💫",
    "🌠",
    "🏆",
    "🥇",
    "🏅",
    "💎",
    "🚀",
    "📈",
    "🧠",
    "🥂",
    "🍻",
    "🤝",
    "🤝🏻",
    "🤝🏼",
    "🤝🏽",
    "🤝🏾",
    "🤝🏿",
    "🙌",
    "🙌🏻",
    "🙌🏼",
    "🙌🏽",
    "🙌🏾",
    "🙌🏿",
    "🪄",
    "🎯",
    "👌",
    "👌🏻",
    "👌🏼",
    "👌🏽",
    "👌🏾",
    "👌🏿",
    "✌️",
    "✌🏻",
    "✌🏼",
    "✌🏽",
    "✌🏾",
    "✌🏿",
    "🤌",
    "🤌🏻",
    "🤌🏼",
    "🤌🏽",
    "🤌🏾",
    "🤌🏿",
    "💪",
    "💪🏻",
    "💪🏼",
    "💪🏽",
    "💪🏾",
    "💪🏿",
    "💸",
    "💰",
    "💹",
}
NEGATIVE_EMOJIS = {
    "👎",
    "👎🏻",
    "👎🏼",
    "👎🏽",
    "👎🏾",
    "👎🏿",
    "😡",
    "🤬",
    "😠",
    "👿",
    "😾",
    "😤",
    "💢",
    "🤡",
    "🤮",
    "🤢",
    "💩",
    "🗑️",
    "🚮",
    "🚽",
    "📉",
    "🛑",
    "🚫",
    "❌",
    "✖️",
    "🤦",
    "🤦‍♂️",
    "🤦‍♀️",
    "🙄",
    "😒",
    "😑",
    "😐",
    "🤨",
    "🥱",
    "😴",
    "🖕",
    "🖕🏻",
    "🖕🏼",
    "🖕🏽",
    "🖕🏾",
    "🖕🏿",
    "🙅",
    "🙅‍♂️",
    "🙅‍♀️",
}

NORMALIZED_POSITIVE = {e.replace("\ufe0f", "") for e in POSITIVE_EMOJIS}
NORMALIZED_NEGATIVE = {e.replace("\ufe0f", "") for e in NEGATIVE_EMOJIS}

from collections import OrderedDict


class MemoryCache:
    """In-memory cache to skip DB reads for rate limiting and message history."""

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


MESSAGE_AUTHOR_CACHE = MemoryCache(5000)
RATE_LIMIT_CACHE = MemoryCache(5000)
UNLINKED_WARNING_CACHE = MemoryCache(5000)
RECENT_VIDEO_CACHE = MemoryCache(300)
LATEST_CHAT_VIDEO: dict[int, tuple[int, str]] = {}


def record_recent_video_file_id(chat_id: int, message_id: int, video_file_id: str | None):
    """Cache downloadable Telegram video `file_id` by `(chat_id, message_id)` and latest per chat."""
    if not video_file_id:
        return
    RECENT_VIDEO_CACHE.put((int(chat_id), int(message_id)), str(video_file_id))
    LATEST_CHAT_VIDEO[int(chat_id)] = (int(message_id), str(video_file_id))

MESSAGE_CXP = 50
POSITIVE_REACTION_BASE = 50
NEGATIVE_REACTION_BASE = -50
RATE_LIMIT_SECONDS = 60


def calculate_level(cxp: int) -> int:
    """Calculate user level from total CXP based on inverse quadratic formula."""
    safe_cxp = max(0, cxp)
    return math.floor((1 + math.sqrt(1 + 4 * safe_cxp / 250)) / 2)


async def apply_level_permissions(bot, chat_id: int, user_id: int, level: int = 1):
    """Apply unrestricted Telegram chat permissions to a user."""
    permissions = ChatPermissions(
        can_send_messages=True,
        can_send_audios=True,
        can_send_documents=True,
        can_send_photos=True,
        can_send_videos=True,
        can_send_video_notes=True,
        can_send_voice_notes=True,
        can_send_polls=True,
        can_send_other_messages=True,
        can_add_web_page_previews=True,
        can_change_info=False,
        can_invite_users=True,
        can_pin_messages=False,
        can_manage_topics=False,
    )
    try:
        await bot.restrict_chat_member(chat_id, user_id, permissions, use_independent_chat_permissions=True)
    except error.RetryAfter:
        raise
    except Exception as e:
        logger.error("Failed to apply permissions to user %s: %s", user_id, e)


def _get_member_tag_string(level: int) -> str:
    if level >= 60:
        return "Pure Symmetry"
    elif level >= 50:
        return "Very Symmetrical"
    elif level >= 40:
        return "Supersymmetric"
    elif level >= 30:
        return "Exquisitely Symmetrical"
    elif level >= 20:
        return "Crystalline Order"
    elif level >= 10:
        return "Rotational Harmony"
    elif level >= 5:
        return "Bilateral Balance"
    elif level >= 1:
        return "Budding Symmetry"
    return ""


async def _update_member_tag(bot, user_id: int, new_level: int):
    """Disabled: Member tags are managed customly in the community."""
    return


async def _announce_level_up(context: CallbackContext, user, new_level: int):
    """Announce level up in the group."""
    config = await db.get_config()
    if not config:
        return

    cxp_topic_id = config.get("cxp_topic_id")
    main_group_id = config.get("main_group_id")

    if main_group_id:
        try:
            full_name = user.first_name + (
                f" {user.last_name}" if getattr(user, "last_name", None) else ""
            )
            mention = f'<a href="tg://user?id={user.id}">{full_name}</a>'
            msg = f"🎉 {mention} has reached <b>Level {new_level}</b>!"
            await context.bot.send_message(
                chat_id=main_group_id,
                message_thread_id=cxp_topic_id if cxp_topic_id else None,
                text=msg,
                parse_mode="HTML",
            )
        except Exception as e:
            logger.error("Failed to send level up announcement: %s", e)


def _extract_message_content_and_media(
    message, is_forum: bool = False
) -> tuple[str, str | None, int | None]:
    """Extract readable message content, media_type, and true reply_to_message_id for memory archiving."""
    base_text = (getattr(message, "text", None) or getattr(message, "caption", None) or "").strip()
    media_type = None
    media_tag = ""

    if getattr(message, "photo", None):
        media_type = "photo"
        media_tag = "[Photo]"
    elif getattr(message, "sticker", None):
        media_type = "sticker"
        emoji = getattr(message.sticker, "emoji", "") or ""
        media_tag = f"[Sticker {emoji}]".strip() if emoji else "[Sticker]"
    elif getattr(message, "voice", None):
        media_type = "voice"
        media_tag = "[Voice Message]"
    elif getattr(message, "video_note", None):
        media_type = "video_note"
        media_tag = "[Video Note]"
    elif getattr(message, "animation", None):
        media_type = "animation"
        media_tag = "[GIF]"
    elif getattr(message, "video", None):
        media_type = "video"
        media_tag = "[Video]"
    elif getattr(message, "audio", None):
        media_type = "audio"
        title = getattr(message.audio, "title", None) or getattr(message.audio, "file_name", None)
        media_tag = f"[Audio: {title}]" if title else "[Audio]"
    elif getattr(message, "document", None):
        media_type = "document"
        fname = getattr(message.document, "file_name", None)
        media_tag = f"[File: {fname}]" if fname else "[File]"
    elif getattr(message, "poll", None):
        media_type = "poll"
        q = getattr(message.poll, "question", "")
        media_tag = f"[Poll: {q}]" if q else "[Poll]"

    if media_tag and base_text:
        content = f"{media_tag} {base_text}"
    elif media_tag:
        content = media_tag
    else:
        content = base_text

    reply_to_id = None
    reply_msg = getattr(message, "reply_to_message", None)
    if reply_msg:
        r_id = reply_msg.message_id
        thread_id = getattr(message, "message_thread_id", None)
        is_topic_root = is_forum and (
            (thread_id is None and r_id == 1)
            or (thread_id is not None and r_id == thread_id)
        )
        if not is_topic_root:
            reply_to_id = r_id

    return content, media_type, reply_to_id


def extract_image_file_id(message) -> str | None:
    """Extract a downloadable Telegram image/thumbnail `file_id` from photo, GIF, video, image doc, or sticker."""
    if not message:
        return None
    photos = getattr(message, "photo", None)
    if photos:
        return getattr(photos[-1], "file_id", None)
    anim = getattr(message, "animation", None)
    if anim:
        if (getattr(anim, "mime_type", None) or "").lower() == "image/gif":
            return getattr(anim, "file_id", None)
        thumb = getattr(anim, "thumbnail", None)
        if thumb and getattr(thumb, "file_id", None):
            return thumb.file_id
    vid = getattr(message, "video", None)
    if vid:
        thumb = getattr(vid, "thumbnail", None)
        if thumb and getattr(thumb, "file_id", None):
            return thumb.file_id
    doc = getattr(message, "document", None)
    if doc and (getattr(doc, "mime_type", None) or "").lower().startswith("image/"):
        return getattr(doc, "file_id", None)
    sticker = getattr(message, "sticker", None)
    if sticker:
        if not getattr(sticker, "is_animated", False) and not getattr(sticker, "is_video", False):
            return getattr(sticker, "file_id", None)
        thumb = getattr(sticker, "thumbnail", None)
        if thumb and getattr(thumb, "file_id", None):
            return thumb.file_id
    return None


def extract_video_file_id(message) -> str | None:
    """Extract a downloadable Telegram video `file_id` (<=20MB Bot API limit) from `video`, `animation`, or `video/*` document."""
    if not message:
        return None
    max_dl_bytes = 20 * 1024 * 1024
    vid = getattr(message, "video", None)
    if vid and getattr(vid, "file_id", None):
        fsize = getattr(vid, "file_size", None)
        if fsize is None or int(fsize) <= max_dl_bytes:
            return vid.file_id
    vnote = getattr(message, "video_note", None)
    if vnote and getattr(vnote, "file_id", None):
        fsize = getattr(vnote, "file_size", None)
        if fsize is None or int(fsize) <= max_dl_bytes:
            return vnote.file_id
    anim = getattr(message, "animation", None)
    if anim and getattr(anim, "file_id", None):
        mime = (getattr(anim, "mime_type", None) or "").lower()
        fsize = getattr(anim, "file_size", None)
        if mime.startswith("video/") and (fsize is None or int(fsize) <= max_dl_bytes):
            return anim.file_id
    doc = getattr(message, "document", None)
    if doc and getattr(doc, "file_id", None):
        mime = (getattr(doc, "mime_type", None) or "").lower()
        fsize = getattr(doc, "file_size", None)
        if mime.startswith("video/") and (fsize is None or int(fsize) <= max_dl_bytes):
            return doc.file_id
    return None


async def _archive_group_message_only(
    chat_id: int,
    msg_id: int,
    user_id: int,
    sender_name: str | None,
    sender_username: str | None,
    content: str,
    *,
    reply_to_message_id: int | None = None,
    thread_id: int | None = None,
    media_type: str | None = None,
    media_file_id: str | None = None,
    created_at: datetime | None = None,
):
    """Record a rate-limited message in `messages` and `group_chat_messages` without awarding CXP."""
    await db.record_message(chat_id, msg_id, user_id)
    if content:
        await db.record_group_chat_message(
            chat_id=chat_id,
            message_id=msg_id,
            user_id=user_id,
            sender_name=sender_name,
            sender_username=sender_username,
            content=content,
            is_bot=False,
            reply_to_message_id=reply_to_message_id,
            thread_id=thread_id,
            media_type=media_type,
            media_file_id=media_file_id,
            created_at=created_at,
        )
        try:
            from memory_engine import memory_engine

            memory_engine.notify_new_message()
        except Exception as e:
            logger.debug("Memory engine notify skipped: %s", e)


async def _process_db_message_and_cxp(
    context: CallbackContext,
    chat_id: int,
    msg_id: int,
    user,
    *,
    content: str = "",
    reply_to_message_id: int | None = None,
    thread_id: int | None = None,
    media_type: str | None = None,
    media_file_id: str | None = None,
    created_at: datetime | None = None,
):
    """Background task to save the message, archive for AI memory, and award CXP without blocking the chat."""
    await db.record_message(chat_id, msg_id, user.id)
    first_name = getattr(user, "first_name", None)
    if first_name:
        last_name = getattr(user, "last_name", None)
        display_name = first_name + (f" {last_name}" if last_name else "")
    else:
        display_name = getattr(
            user, "title", getattr(user, "username", f"User {user.id}")
        )
    username = getattr(user, "username", None)
    await db.update_user_display_name(user.id, display_name, username)

    if content:
        await db.record_group_chat_message(
            chat_id=chat_id,
            message_id=msg_id,
            user_id=user.id,
            sender_name=display_name,
            sender_username=username,
            content=content,
            is_bot=False,
            reply_to_message_id=reply_to_message_id,
            thread_id=thread_id,
            media_type=media_type,
            media_file_id=media_file_id,
            created_at=created_at,
        )
        try:
            from memory_engine import memory_engine

            memory_engine.notify_new_message()
        except Exception as e:
            logger.debug("Memory engine notify skipped: %s", e)

    user_data = await db.get_user(user.id)
    current_cxp = user_data.get("cxp", 0) if user_data else 0
    old_level = calculate_level(current_cxp)
    success = await db.update_user_cxp(
        user.id,
        MESSAGE_CXP,
        update_timestamp=True,
        category="message",
        sub_type="chat",
        reference_id=msg_id,
        metadata={"chat_id": chat_id},
    )
    if success:
        new_level = calculate_level(current_cxp + MESSAGE_CXP)
        if new_level > old_level:
            await _announce_level_up(context, user, new_level)


async def track_message_activity(update: Update, context: CallbackContext):
    """Track messaging activity using an in-memory cache to prevent DB bottlenecking."""
    if not update.effective_user or not update.message:
        return

    # If the user is a bot, it could be an anonymous admin or channel
    if getattr(update.effective_user, "is_bot", True):
        # We allow messages sent on behalf of a channel or anonymous admin group
        if not update.message.sender_chat:
            return

        chat_source = update.message.sender_chat
        channel_id = chat_source.id

        if getattr(chat_source, "type", "") == "channel":
            owner_id = await db.get_channel_owner(channel_id)
            if owner_id:
                user_id = owner_id
                db_user = await db.get_user(owner_id)
                
                class MockUser:
                    def __init__(self):
                        self.id = owner_id
                        self.username = db_user.get("username") if db_user else None
                        self.first_name = db_user.get("display_name") if db_user else f"Owner {owner_id}"
                        self.last_name = None
                        self.is_bot = False
                        
                user_obj = MockUser()
            else:
                user_id = channel_id
                class MockUser:
                    def __init__(self, chat):
                        self.id = chat.id
                        self.username = chat.username
                        self.first_name = chat.title or chat.username or f"Channel {chat.id}"
                        self.last_name = None
                        self.is_bot = False
                user_obj = MockUser(chat_source)
                
                # Setup warning for unlinked channels
                now = datetime.now()
                last_warn = UNLINKED_WARNING_CACHE.get(channel_id)
                if not last_warn or (now - last_warn).total_seconds() > 3600:
                    UNLINKED_WARNING_CACHE.put(channel_id, now)
                    try:
                        from telegram import InlineKeyboardMarkup, InlineKeyboardButton
                        bot_username = context.bot.username
                        keyboard = InlineKeyboardMarkup([
                            [InlineKeyboardButton("🔗 Link Channel", url=f"https://t.me/{bot_username}?start=setchannel")]
                        ])
                        msg = await update.message.reply_text(
                            f"⚠️ *Unlinked Channel Warning*\n\n"
                            f"You are posting as a channel that is not linked to your user account. This means you will not earn CXP.\n\n"
                            f"Please click the button below to link it.",
                            reply_markup=keyboard,
                        )
                    except Exception as e:
                        logger.error(f"Failed to send unlinked channel warning: {e}")
        else:
            # Anonymous admin group
            user_id = channel_id
            class MockUser:
                def __init__(self, chat):
                    self.id = chat.id
                    self.username = chat.username
                    self.first_name = chat.title or chat.username or f"Channel {chat.id}"
                    self.last_name = None
                    self.is_bot = False
            user_obj = MockUser(chat_source)
    else:
        user_id = update.effective_user.id
        user_obj = update.effective_user

    chat_id = update.effective_chat.id

    config = await db.get_config()
    main_group_id = config.get("main_group_id") if config else None

    # Only track activity in the main group
    if chat_id != main_group_id:
        return

    # Fast in-memory cache update for reactions
    MESSAGE_AUTHOR_CACHE.put((chat_id, update.message.message_id), user_id)

    is_forum = bool(getattr(update.effective_chat, "is_forum", False))
    content, media_type, reply_to_id = _extract_message_content_and_media(
        update.message, is_forum=is_forum
    )
    media_file_id = extract_image_file_id(update.message)
    vid_file_id = extract_video_file_id(update.message)
    if vid_file_id:
        record_recent_video_file_id(chat_id, update.message.message_id, vid_file_id)
    msg_thread_id = getattr(update.message, "message_thread_id", None)
    msg_date = getattr(update.message, "date", None)

    first_name = getattr(user_obj, "first_name", None)
    if first_name:
        last_name = getattr(user_obj, "last_name", None)
        sender_name = first_name + (f" {last_name}" if last_name else "")
    else:
        sender_name = getattr(
            user_obj, "title", getattr(user_obj, "username", f"User {user_id}")
        )
    sender_username = getattr(user_obj, "username", None)

    # Fast in-memory rate limiting check
    now = datetime.now()
    last_msg_time = RATE_LIMIT_CACHE.get(user_id)

    if last_msg_time and (now - last_msg_time).total_seconds() < RATE_LIMIT_SECONDS:
        # Rate limited for CXP, so archive message + feed AI floating context/memory without CXP.
        context.application.create_task(
            _archive_group_message_only(
                chat_id,
                update.message.message_id,
                user_id,
                sender_name,
                sender_username,
                content,
                reply_to_message_id=reply_to_id,
                thread_id=msg_thread_id,
                media_type=media_type,
                media_file_id=media_file_id,
                created_at=msg_date,
            )
        )
        return

    # User is eligible for CXP! Update memory cache immediately.
    RATE_LIMIT_CACHE.put(user_id, now)

    # Dispatch to background task to keep the handler lightning fast
    context.application.create_task(
        _process_db_message_and_cxp(
            context,
            chat_id,
            update.message.message_id,
            user_obj,
            content=content,
            reply_to_message_id=reply_to_id,
            thread_id=msg_thread_id,
            media_type=media_type,
            media_file_id=media_file_id,
            created_at=msg_date,
        )
    )


async def _resolve_reaction_emoji(reaction, context: CallbackContext) -> str | None:
    if reaction.type == "emoji":
        return reaction.emoji
    elif reaction.type == "custom_emoji":
        try:
            stickers = await context.bot.get_custom_emoji_stickers(
                [reaction.custom_emoji_id]
            )
            if stickers and stickers[0].emoji:
                return stickers[0].emoji
        except Exception as e:
            logger.warning(
                "Failed to resolve custom emoji %s: %s", reaction.custom_emoji_id, e
            )
    return None


async def evaluate_reaction(update: Update, context: CallbackContext):
    """Parse reactions and apply Karma scaled by reactor level."""
    if not update.message_reaction:
        return

    reaction_update = update.message_reaction

    # If the user is missing or is a bot, try getting the actor_chat (for channels)
    reactor_id = None
    if reaction_update.user and not reaction_update.user.is_bot:
        reactor_id = reaction_update.user.id
    elif reaction_update.actor_chat:
        reactor_id = reaction_update.actor_chat.id
        
        # Check if the reactor chat is a linked channel
        if getattr(reaction_update.actor_chat, "type", "") == "channel":
            owner_id = await db.get_channel_owner(reactor_id)
            if owner_id:
                reactor_id = owner_id

    if not reactor_id:
        return

    # Message author caching
    author_id = MESSAGE_AUTHOR_CACHE.get(
        (reaction_update.chat.id, reaction_update.message_id)
    )
    if not author_id:
        author_id = await db.get_message_author(
            reaction_update.chat.id, reaction_update.message_id
        )
        if not author_id:
            return
        # Reseed cache
        MESSAGE_AUTHOR_CACHE.put(
            (reaction_update.chat.id, reaction_update.message_id), author_id
        )

    # Don't let users farm karma on their own messages
    if author_id == reactor_id:
        return

    reactor_data = await db.get_user(reactor_id)
    if not reactor_data:
        return

    reactor_level = calculate_level(reactor_data.get("cxp", 0))
    multiplier = 1.0 + ((reactor_level - 1) * 0.05)

    # Calculate difference in emojis
    old_emojis = []
    for r in reaction_update.old_reaction:
        resolved = await _resolve_reaction_emoji(r, context)
        if resolved:
            old_emojis.append(resolved.replace("\ufe0f", ""))

    new_emojis = []
    for r in reaction_update.new_reaction:
        resolved = await _resolve_reaction_emoji(r, context)
        if resolved:
            new_emojis.append(resolved.replace("\ufe0f", ""))

    added = [e for e in new_emojis if e not in old_emojis]
    removed = [e for e in old_emojis if e not in new_emojis]

    total_delta = 0
    for e in added:
        if e in NORMALIZED_POSITIVE:
            total_delta += POSITIVE_REACTION_BASE * multiplier
        elif e in NORMALIZED_NEGATIVE:
            total_delta += NEGATIVE_REACTION_BASE * multiplier

    for e in removed:
        if e in NORMALIZED_POSITIVE:
            total_delta -= POSITIVE_REACTION_BASE * multiplier
        elif e in NORMALIZED_NEGATIVE:
            total_delta -= NEGATIVE_REACTION_BASE * multiplier

    if total_delta != 0:
        total_delta = int(round(total_delta))
        author_data = await db.get_user(author_id)
        if not author_data:
            return

        old_level = calculate_level(author_data.get("cxp", 0))
        await db.update_user_cxp(
            author_id,
            total_delta,
            category="reaction",
            sub_type="positive" if total_delta > 0 else "negative",
            counterparty_id=reactor_id,
            reference_id=reaction_update.message_id,
            metadata={
                "multiplier": round(multiplier, 2),
                "added": added,
                "removed": removed,
            },
        )
        new_data = await db.get_user(author_id)
        new_cxp = new_data.get("cxp", 0)
        new_level = calculate_level(new_cxp)

        if new_level > old_level:
            try:
                author_user = await context.bot.get_chat(author_id)
                await _announce_level_up(context, author_user, new_level)
            except Exception as e:
                logger.warning(
                    "Could not fetch author %s for level up announcement: %s",
                    author_id,
                    e,
                )


async def _delete_message_job(context: CallbackContext):
    """Job to automatically delete a message."""
    chat_id = context.job.data.get("chat_id")
    message_id = context.job.data.get("message_id")
    try:
        await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
    except Exception:
        pass


async def enforce_cxp_topic(
    update: Update, context: CallbackContext, main_group_id, cxp_topic_id
) -> bool:
    """Topic enforcement disabled - commands are allowed in all topics and general chat."""
    return True


async def user_stats_cmd(update: Update, context: CallbackContext):
    """Handler for /level."""
    if not update.effective_user or not update.message:
        return

    if (
        getattr(update.effective_user, "is_bot", True)
        and not update.message.sender_chat
    ):
        return


    if getattr(update.effective_user, "is_bot", True) and update.message.sender_chat:
        target_id = update.message.sender_chat.id
        target_name = (
            update.message.sender_chat.title
            or update.message.sender_chat.username
            or f"Channel {target_id}"
        )
        if getattr(update.message.sender_chat, "type", "") == "channel":
            owner_id = await db.get_channel_owner(target_id)
            if owner_id:
                target_id = owner_id
                db_user = await db.get_user(owner_id)
                owner_name = db_user.get("display_name") if db_user else f"Owner {owner_id}"
                target_name = f"{owner_name} (via {update.message.sender_chat.title})"
    else:
        target_id = update.effective_user.id
        target_name = update.effective_user.first_name + (
            f" {update.effective_user.last_name}"
            if update.effective_user.last_name
            else ""
        )

    # Check for reply targeting another user.
    target_chat = None
    target_user = None

    if update.message.reply_to_message and not getattr(
        update.message, "is_automatic_forward", False
    ):
        if (
            update.message.is_topic_message
            and update.message.reply_to_message.message_id
            == update.message.message_thread_id
        ):
            pass  # Ignore implicit topic reply
        else:
            target_chat = update.message.reply_to_message.sender_chat
            target_user = update.message.reply_to_message.from_user

    if target_chat:
        target_id = target_chat.id
        target_name = (
            target_chat.title or target_chat.username or f"Channel {target_id}"
        )
        if getattr(target_chat, "type", "") == "channel":
            owner_id = await db.get_channel_owner(target_id)
            if owner_id:
                target_id = owner_id
                db_user = await db.get_user(owner_id)
                owner_name = db_user.get("display_name") if db_user else f"Owner {owner_id}"
                target_name = f"{owner_name} (via {target_chat.title})"
    elif target_user and not getattr(target_user, "is_bot", False):
        target_id = target_user.id
        target_name = target_user.first_name

    # Check for arguments targeting another user
    if context.args:
        arg_str = " ".join(context.args)
        resolved_id, resolved_name = await resolve_username(arg_str, update, context)

        if resolved_id:
            target_id = resolved_id
            target_name = resolved_name
        else:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                message_thread_id=update.message.message_thread_id,
                text="Could not resolve a target user from the arguments provided via Telegram. Please ensure the @username is correct.",
            )
            return

    user_data = await db.get_user(target_id)
    if not user_data:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            message_thread_id=update.message.message_thread_id,
            text="No stats found for that user.",
        )
        return

    cxp = user_data.get("cxp", 0)
    level = calculate_level(cxp)
    is_blacklisted = user_data.get("is_blacklisted", False)

    if is_blacklisted:
        rank_display = "Unranked"
    else:
        rank = await db.get_user_rank(cxp)
        rank_display = f"#{rank}"

    next_level_cxp = 250 * (level + 1) * level

    # Try our best to get a name if target_name is missing, which happens for channels
    if not target_name:
        try:
            chat = await context.bot.get_chat(target_id)
            target_name = chat.title or chat.first_name or f"Channel {target_id}"
        except Exception:
            target_name = f"User/Channel {target_id}"

    level_title = _get_member_tag_string(level)
    level_display = f"{level} ({level_title})" if level_title else str(level)

    safe_target_name = escape_markdown(target_name, version=1)

    msg = (
        f"📊 **Statistics for {safe_target_name}**\n\n"
        f"🏆 **Rank:** {rank_display}\n"
        f"🔰 **Level:** {level_display}\n"
        f"✨ **CXP:** {cxp:,} / {next_level_cxp:,}"
    )

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        message_thread_id=update.message.message_thread_id,
        text=msg,
        parse_mode="Markdown",
    )


def parse_leaderboard_args(args, config):
    """Returns (start_date, end_date, title_string) or (None, None, 'Global Leaderboard (All-Time)') for all-time"""
    if not args:
        return None, None, "Global Leaderboard (All-Time)"
        
    arg_str = " ".join(args).lower().strip()
    now_utc = datetime.now(timezone.utc).date()
    
    if arg_str == "contest":
        if config and config.get("contest_start") and config.get("contest_end"):
            return config["contest_start"], config["contest_end"], "Contest Leaderboard"
        return None, None, "Contest Leaderboard (No Dates Set)"
        
    if arg_str in ["today", "day", "daily", "this day"]:
        return now_utc, now_utc, "Daily Leaderboard"
        
    if arg_str == "yesterday":
        yesterday = now_utc - timedelta(days=1)
        return yesterday, yesterday, "Yesterday's Leaderboard"
        
    if arg_str in ["week", "weekly", "this week"]:
        start_of_week = now_utc - timedelta(days=now_utc.weekday())
        end_of_week = start_of_week + timedelta(days=6)
        return start_of_week, end_of_week, "Weekly Leaderboard"
        
    if arg_str == "last week":
        start_of_last_week = now_utc - timedelta(days=now_utc.weekday() + 7)
        end_of_last_week = start_of_last_week + timedelta(days=6)
        return start_of_last_week, end_of_last_week, "Last Week's Leaderboard"
        
    if arg_str in ["month", "monthly", "this month"]:
        start_of_month = now_utc.replace(day=1)
        last_day = calendar.monthrange(now_utc.year, now_utc.month)[1]
        end_of_month = now_utc.replace(day=last_day)
        return start_of_month, end_of_month, "Monthly Leaderboard"
        
    if arg_str == "last month":
        first_of_this_month = now_utc.replace(day=1)
        end_of_last_month = first_of_this_month - timedelta(days=1)
        start_of_last_month = end_of_last_month.replace(day=1)
        return start_of_last_month, end_of_last_month, "Last Month's Leaderboard"
        
    # Handle specific months
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }
    
    if arg_str in months:
        target_month = months[arg_str]
        target_year = now_utc.year
        if target_month > now_utc.month:
            target_year -= 1 # Assume they mean last year if the month hasn't happened yet this year
            
        start_date = datetime(target_year, target_month, 1).date()
        last_day = calendar.monthrange(target_year, target_month)[1]
        end_date = datetime(target_year, target_month, last_day).date()
        month_name = calendar.month_name[target_month]
        return start_date, end_date, f"{month_name} Leaderboard"
        
    return None, None, "Global Leaderboard (All-Time)"


async def leaderboard_cmd(update: Update, context: CallbackContext):
    """Handler for /leaderboard (/lb). Renders a Top 3 Podium image + Top 10 caption."""
    if not update.effective_chat or not update.message:
        return

    chat_id = update.effective_chat.id
    thread_id = update.message.message_thread_id
    config = await db.get_config()

    start_date, end_date, title = parse_leaderboard_args(context.args, config)
    is_timeframe = bool(start_date and end_date)

    actual_top_10 = await db.get_leaderboard(
        limit=10, start_date=start_date, end_date=end_date
    )
    if not actual_top_10:
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=thread_id,
            text=f"The {title.lower()} is currently empty!",
        )
        return

    try:
        await context.bot.send_chat_action(
            chat_id=chat_id,
            action=ChatAction.UPLOAD_PHOTO,
            message_thread_id=thread_id if getattr(update.effective_chat, "is_forum", False) else None,
        )
    except Exception:
        pass

    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
    resolved_rows: list[dict] = []

    for i, row in enumerate(actual_top_10):
        u_id = row.get("user_id")
        cxp = int(row.get("cxp") or 0)
        total_cxp = int(row.get("total_cxp", cxp) or 0)
        level = calculate_level(total_cxp)
        level_title = _get_member_tag_string(level)

        name = row.get("display_name")
        username = row.get("username")
        if not name:
            if username:
                name = f"@{username}"
            else:
                try:
                    chat = await context.bot.get_chat(u_id)
                    name = chat.title or chat.first_name or f"Channel {u_id}"
                except Exception:
                    name = f"User {u_id}"

        resolved_rows.append(
            {
                "rank": i + 1,
                "user_id": u_id,
                "name": name,
                "username": username,
                "cxp": cxp,
                "total_cxp": total_cxp,
                "level": level,
                "level_title": level_title,
                "medal": medals[i] if i < len(medals) else f"#{i + 1}",
            }
        )

    # Fetch Top 3 avatars concurrently (with 15-min in-memory cache)
    top3 = resolved_rows[:3]
    avatars = await asyncio.gather(
        *(fetch_user_avatar_bytes(context.bot, r["user_id"]) for r in top3),
        return_exceptions=True,
    )
    podium_entries = []
    for r, av in zip(top3, avatars):
        entry = dict(r)
        entry["avatar_bytes"] = av if isinstance(av, (bytes, bytearray)) else None
        podium_entries.append(entry)

    subtitle_detail = None
    if is_timeframe and start_date and end_date:
        if start_date == end_date:
            subtitle_detail = start_date.strftime("%b %d, %Y")
        else:
            subtitle_detail = (
                f"{start_date.strftime('%b %d')} – {end_date.strftime('%b %d, %Y')}"
            )

    podium_buf = await asyncio.to_thread(
        render_podium_png,
        title,
        podium_entries,
        is_timeframe=is_timeframe,
        subtitle_detail=subtitle_detail,
    )

    lines = [f"🏆 <b>{html.escape(title)} (Top 10)</b> 🏆\n"]
    for r in resolved_rows:
        safe_name = html.escape(r["name"])
        sign_prefix = "+" if (is_timeframe and r["cxp"] > 0) else ""
        lines.append(
            f"{r['medal']} <b>{safe_name}</b> — Level {r['level']} "
            f"(<code>{sign_prefix}{r['cxp']:,} CXP</code>)"
        )
    caption_html = "\n".join(lines)

    if len(caption_html) <= MessageLimit.CAPTION_LENGTH:
        await context.bot.send_photo(
            chat_id=chat_id,
            message_thread_id=thread_id,
            photo=podium_buf,
            filename="leaderboard_podium.png",
            caption=caption_html,
            parse_mode=ParseMode.HTML,
        )
    else:
        await context.bot.send_photo(
            chat_id=chat_id,
            message_thread_id=thread_id,
            photo=podium_buf,
            filename="leaderboard_podium.png",
            caption=f"🏆 <b>{html.escape(title)}</b> 🏆",
            parse_mode=ParseMode.HTML,
        )
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=thread_id,
            text=caption_html,
            parse_mode=ParseMode.HTML,
        )


async def get_id_cmd(update: Update, context: CallbackContext):
    """Test command to explicitly check API/DB username resolution and reply parsing. (was /id, now /checkid)"""
    if not update.effective_user or not update.message:
        return

    if (
        getattr(update.effective_user, "is_bot", True)
        and not update.message.sender_chat
    ):
        return

    config = await db.get_config()
    main_group_id = config.get("main_group_id") if config else None

    # Determine actor identity
    actor_id = update.effective_user.id
    if getattr(update.effective_user, "is_bot", True) and update.message.sender_chat:
        actor_id = update.message.sender_chat.id

    if not await is_user_admin(actor_id, main_group_id, context):
        return

    # 1. Check for replied-to message
    if (
        update.message.reply_to_message
        and not getattr(update.message, "is_automatic_forward", False)
        and getattr(update.message.reply_to_message.from_user, "id", None)
    ):
        # Ignore implicit replies to the topic starter message
        if not (
            update.message.is_topic_message
            and update.message.reply_to_message.message_id
            == update.message.message_thread_id
        ):
            target_user = update.message.reply_to_message.from_user
            full_name = target_user.first_name + (
                f" {target_user.last_name}" if target_user.last_name else ""
            )
            safe_full_name = escape_markdown(full_name, version=1)
            await update.message.reply_text(
                f"Reply Resolution Success!\nName: {safe_full_name}\nID: `{target_user.id}`",
                parse_mode="Markdown",
            )
            return

    # 2. Check for @username arguments
    args = context.args
    if not args:
        await update.message.reply_text(
            "Please provide a @username or reply to their message."
        )
        return

    arg_str = args[0]
    resolved_id, resolved_name = await resolve_username(arg_str, update, context)

    if resolved_id:
        safe_resolved_name = escape_markdown(resolved_name, version=1)
        await update.message.reply_text(
            f"Resolution Success!\nName: {safe_resolved_name}\nID: `{resolved_id}`",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text(
            "Failed to resolve that handle via Telegram API or DB Tracker."
        )


async def give_cxp_cmd(update: Update, context: CallbackContext):
    """Admin only: /give <val> [@username/reply]. Grant or remove CXP."""
    if not update.effective_user or not update.message:
        return

    if (
        getattr(update.effective_user, "is_bot", True)
        and not update.message.sender_chat
    ):
        return

    config = await db.get_config()
    main_group_id = config.get("main_group_id") if config else None

    # Must be in main group
    if not main_group_id or update.effective_chat.id != main_group_id:
        return

    # Determine actor identity (User or Channel)
    actor_id = update.effective_user.id
    if getattr(update.effective_user, "is_bot", True) and update.message.sender_chat:
        actor_id = update.message.sender_chat.id

    if not await is_user_admin(actor_id, main_group_id, context):
        return

    # Parse args (could be /give 100 @user or /give @user 100)
    args = context.args
    if not args and not update.message.reply_to_message:
        await update.message.reply_text(
            "Usage: `/give <amount> [@username]` or reply to a message with `/give <amount>`",
            parse_mode="Markdown",
        )
        return

    target_id = None
    target_name = None
    delta_cxp = None

    # Try reply target
    # Channels automatically "reply" to the root channel post when posting in the discussion group.
    # We must explicitly exclude `is_automatic_forward` logic and topic roots.
    target_chat = None
    target_user = None

    if update.message.reply_to_message and not getattr(
        update.message, "is_automatic_forward", False
    ):
        # If it's a forum topic, ignore the implicit reply to the topic starter message
        if (
            update.message.is_topic_message
            and update.message.reply_to_message.message_id
            == update.message.message_thread_id
        ):
            pass
        else:
            target_chat = update.message.reply_to_message.sender_chat
            target_user = update.message.reply_to_message.from_user

    if target_chat:
        target_id = target_chat.id
        target_name = (
            target_chat.title or target_chat.username or f"Channel {target_id}"
        )
    elif target_user and not getattr(target_user, "is_bot", False):
        target_id = target_user.id
        target_name = target_user.first_name

    # Parse args to support varying formats: `/give 1000 @usr`, `/give John Doe 1000`
    arg_str = None
    target_args = []

    for arg in args:
        if delta_cxp is None:
            try:
                delta_cxp = int(arg)
                continue
            except ValueError:
                pass
        target_args.append(arg)

    if target_args:
        arg_str = " ".join(target_args)

    # Attempt resolution to allow explicit usernames to override a reply target
    # If no explicit @ string was found, still pass it through for text_mentions
    resolved_id, resolved_name = await resolve_username(arg_str, update, context)
    if resolved_id:
        target_id = resolved_id
        target_name = resolved_name

    if target_id is None:
        await update.message.reply_text(
            "Could not resolve the target user via Telegram. Please ensure the @username is correct or reply to their message."
        )
        return

    if delta_cxp is None:
        await update.message.reply_text("Invalid amount provided.")
        return

    # Safely fetch target and update
    target_data = await db.get_user(target_id)
    if not target_data:
        await update.message.reply_text("Could not fetch target user data.")
        return

    if target_name and (not target_data.get("display_name") or not target_data.get("username")):
        clean_user = target_name.lstrip("@") if target_name.startswith("@") else None
        await db.update_user_display_name(target_id, target_name, clean_user)

    old_level = calculate_level(target_data.get("cxp", 0))
    await db.update_user_cxp(
        target_id,
        delta_cxp,
        category="admin_give",
        sub_type="grant" if delta_cxp > 0 else "deduct",
        counterparty_id=actor_id,
    )
    new_data = await db.get_user(target_id)
    new_cxp = new_data.get("cxp", 0)
    new_level = calculate_level(new_cxp)

    action = "granted" if delta_cxp > 0 else "removed"
    msg = f"✅ Successfully {action} {abs(delta_cxp):,} CXP to {target_name}. Their new total is {new_cxp:,} CXP (Level {new_level})."
    await update.message.reply_text(msg)


async def set_admin_cmd(update: Update, context: CallbackContext):
    """Bot Owner only: /setadmin <true/false> [@username/reply]. Grant or remove DB admin status."""
    from bot import BOT_OWNER_ID  # Ensure only the ultimate absolute owner can run this

    if not BOT_OWNER_ID or update.effective_user.id != BOT_OWNER_ID:
        return

    if not update.effective_user or not update.message:
        return

    # Parse args (can be `/setadmin true @user` or `/setadmin @user true`)
    args = context.args
    if not args and not update.message.reply_to_message:
        await update.message.reply_text(
            "Usage: `/setadmin [true/false] [@username]` or reply to a message with `/setadmin [true/false]`",
            parse_mode="Markdown",
        )
        return

    target_id = None
    target_name = None
    target_chat = None
    target_user = None
    is_admin_flag = None

    # Try reply target
    if update.message.reply_to_message and not getattr(
        update.message, "is_automatic_forward", False
    ):
        if (
            update.message.is_topic_message
            and update.message.reply_to_message.message_id
            == update.message.message_thread_id
        ):
            pass
        else:
            target_chat = update.message.reply_to_message.sender_chat
            target_user = update.message.reply_to_message.from_user

    if target_chat:
        target_id = target_chat.id
        target_name = (
            target_chat.title or target_chat.username or f"Channel {target_id}"
        )
    elif target_user and not getattr(target_user, "is_bot", False):
        target_id = target_user.id
        target_name = target_user.first_name

    arg_str = None
    target_args = []

    for arg in args:
        lower_arg = arg.lower()
        if is_admin_flag is None and lower_arg in ["true", "1", "yes", "t"]:
            is_admin_flag = True
            continue
        elif is_admin_flag is None and lower_arg in ["false", "0", "no", "f", "remove"]:
            is_admin_flag = False
            continue

        target_args.append(arg)

    if target_args:
        arg_str = " ".join(target_args)

    resolved_id, resolved_name = await resolve_username(arg_str, update, context)
    if resolved_id:
        target_id = resolved_id
        target_name = resolved_name

    if target_id is None:
        await update.message.reply_text(
            "Could not resolve the target user via Telegram. Please ensure the @username is correct or reply to their message."
        )
        return

    if is_admin_flag is None:
        await update.message.reply_text("Please provide true or false.")
        return

    # Safely fetch target and update
    target_data = await db.get_user(target_id)
    if not target_data:
        # User doesn't exist, we should seed them
        await db.record_message(
            update.effective_chat.id, update.message.message_id, target_id
        )

    await db.update_user_admin_status(target_id, is_admin_flag)

    action = (
        "granted DB Admin powers to" if is_admin_flag else "revoked DB Admin powers for"
    )
    await update.message.reply_text(f"✅ Successfully {action} {target_name}.")


async def steal_cxp_cmd(update: Update, context: CallbackContext):
    """Member command: /steal. Steal up to 25-100 CXP from a random user with >= 1 CXP (1-hour cooldown)."""
    if not update.effective_user or not update.message:
        return

    # Check if we are in the main group
    config = await db.get_config()
    main_group_id = config.get("main_group_id") if config else None

    if not main_group_id or update.effective_chat.id != main_group_id:
        return

    if getattr(update.effective_user, "is_bot", True):
        if not update.message.sender_chat:
            return
        sender_chat = update.message.sender_chat
        if getattr(sender_chat, "type", "") != "channel":
            return
        owner_id = await db.get_channel_owner(sender_chat.id)
        if not owner_id:
            from telegram import InlineKeyboardMarkup, InlineKeyboardButton

            bot_username = context.bot.username
            keyboard = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "🔗 Link Channel",
                            url=f"https://t.me/{bot_username}?start=setchannel",
                        )
                    ]
                ]
            )
            await update.message.reply_text(
                "⚠️ Please link your channel to your user account first before using `/steal`.",
                parse_mode="Markdown",
                reply_markup=keyboard,
            )
            return
        user_id = owner_id
        db_user = await db.get_user(owner_id)
        user_name = (
            (db_user.get("display_name") if db_user else None)
            or sender_chat.title
            or f"User {owner_id}"
        )

        class _MockStealUser:
            def __init__(self, uid: int, name: str):
                self.id = uid
                self.first_name = name
                self.last_name = None
                self.is_bot = False

        user_obj = _MockStealUser(user_id, user_name)
    else:
        user_id = update.effective_user.id
        user_name = update.effective_user.first_name + (
            f" {update.effective_user.last_name}"
            if update.effective_user.last_name
            else ""
        )
        user_obj = update.effective_user

    desired_amount = random.randint(25, 100)
    shielded = random.random() < 0.10

    res = await db.attempt_steal(
        thief_id=user_id,
        desired_amount=desired_amount,
        shielded=shielded,
    )

    if not res.get("ok"):
        reason = res.get("reason")
        if reason == "cooldown":
            remaining_seconds = res.get("remaining_seconds", 0)
            minutes = int(remaining_seconds // 60)
            seconds = int(remaining_seconds % 60)
            safe_user_name = escape_markdown(user_name, version=1)
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                message_thread_id=update.message.message_thread_id,
                text=f"⏳ {safe_user_name}, you are on cooldown! Please wait `{minutes}m {seconds}s` before trying again.",
                parse_mode="Markdown",
            )
        elif reason == "no_valid_target":
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                message_thread_id=update.message.message_thread_id,
                text="❌ Nobody in the group currently has at least 1 CXP available to steal from!",
            )
        else:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                message_thread_id=update.message.message_thread_id,
                text="❌ Could not perform steal right now. Please try again.",
            )
        return

    target_data = res["target"]
    target_id = target_data.get("user_id")
    target_name = target_data.get("display_name")

    if not target_name:
        username = target_data.get("username")
        if username:
            target_name = f"@{username}"
        else:
            try:
                chat = await context.bot.get_chat(target_id)
                target_name = (
                    chat.first_name
                    + (f" {chat.last_name}" if getattr(chat, "last_name", None) else "")
                    if chat.first_name
                    else chat.title or f"User {target_id}"
                )
            except Exception:
                target_name = f"User {target_id}"

    if res.get("shielded"):
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            message_thread_id=update.message.message_thread_id,
            text=f"🛡️ Shield detected! {user_name} failed to steal!",
        )
        return

    stolen_amount = res["stolen_amount"]
    safe_user_name = escape_markdown(user_name, version=1)
    safe_target_name = escape_markdown(target_name, version=1)
    capped_note = " (all the CXP they had left)" if res.get("capped") else ""

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        message_thread_id=update.message.message_thread_id,
        text=f"🎯 **{safe_user_name}** stole `{stolen_amount}` CXP from **{safe_target_name}**{capped_note}!",
        parse_mode="Markdown",
    )

    old_level = calculate_level(res["thief_old_cxp"])
    new_level = calculate_level(res["thief_new_cxp"])
    if new_level > old_level:
        await _announce_level_up(context, user_obj, new_level)


async def contest_cmd(update: Update, context: CallbackContext):
    """Alias for /leaderboard contest."""
    context.args = ["contest"]
    await leaderboard_cmd(update, context)


async def setcontest_cmd(update: Update, context: CallbackContext):
    """Admin only: /setcontest <mm/dd/yy> <mm/dd/yy> OR <mm/dd/yy> <days> OR <days>. Set contest timeframe."""
    if not update.effective_user or not update.message:
        return

    if (
        getattr(update.effective_user, "is_bot", True)
        and not update.message.sender_chat
    ):
        return

    config = await db.get_config()
    main_group_id = config.get("main_group_id") if config else None

    # Determine actor identity
    actor_id = update.effective_user.id
    if getattr(update.effective_user, "is_bot", True) and update.message.sender_chat:
        actor_id = update.message.sender_chat.id

    if not await is_user_admin(actor_id, main_group_id, context):
        return

    args = context.args
    if not args:
        await update.message.reply_text(
            "Usage:\n`/setcontest <days>` (starts today, inclusive length)\n`/setcontest <mm/dd/yy> <days>`\n`/setcontest <mm/dd/yy> <mm/dd/yy>`",
            parse_mode="Markdown",
        )
        return

    now_utc = datetime.now(timezone.utc).date()
    start_date = None
    end_date = None

    def parse_date(date_str):
        try:
            return datetime.strptime(date_str, "%m/%d/%y").date()
        except ValueError:
            return None

    if len(args) == 1:
        try:
            days = int(args[0])
            if days <= 0:
                await update.message.reply_text("Days must be > 0.")
                return
            start_date = now_utc
            end_date = start_date + timedelta(days=days-1)
        except ValueError:
            await update.message.reply_text("Invalid length in days.")
            return
    elif len(args) >= 2:
        start_date = parse_date(args[0])
        if not start_date:
            await update.message.reply_text("Invalid start date format. Use mm/dd/yy.")
            return
            
        try:
            days = int(args[1])
            if days <= 0:
                await update.message.reply_text("Days must be > 0.")
                return
            end_date = start_date + timedelta(days=days-1)
        except ValueError:
            end_date = parse_date(args[1])
            if not end_date:
                await update.message.reply_text("Invalid end date or days format. Use mm/dd/yy or an integer.")
                return

    if start_date and end_date:
        if end_date < start_date:
            await update.message.reply_text("End date must be after or equal to the start date.")
            return

        success = await db.update_config(contest_start=start_date, contest_end=end_date)
        if success:
            await update.message.reply_text(f"✅ Contest timeframe set:\nStart: {start_date.strftime('%Y-%m-%d')}\nEnd: {end_date.strftime('%Y-%m-%d')}")
        else:
            await update.message.reply_text("❌ Failed to update the database.")
    else:
        await update.message.reply_text("Invalid arguments.")


async def syncperms_cmd(update: Update, context: CallbackContext):
    """Admin command to backfill permissions for all users in the DB to ensure full chat access."""
    if not update.effective_user or not update.message:
        return

    user_id = update.effective_user.id
    from bot import BOT_OWNER_ID

    if user_id != BOT_OWNER_ID:
        user_data = await db.get_user(user_id)
        if not user_data or not user_data.get("is_admin"):
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                message_thread_id=update.message.message_thread_id,
                text="❌ You do not have permission to use this command.",
            )
            return

    config = await db.get_config()
    main_group_id = config.get("main_group_id") if config else None
    if not main_group_id:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            message_thread_id=update.message.message_thread_id,
            text="❌ Main group ID not configured. Cannot sync permissions.",
        )
        return

    await update.message.reply_text(
        "🔄 Starting permission backfill to grant full chat access to all members... Check console logs for progress.",
        reply_to_message_id=update.message.message_id,
    )

    # Run the backfill in the background to avoid blocking the bot handler
    context.application.create_task(
        _run_syncperms_background(context.bot, main_group_id)
    )


async def _run_syncperms_background(bot, main_group_id):
    logger.info("Starting background permission sync...")
    if not db.pool:
        logger.warning("DB pool not initialized, skipping sync.")
        return

    # Fetch all users
    try:
        async with db.pool.acquire() as conn:
            users = await conn.fetch("SELECT user_id FROM users")
    except Exception as e:
        logger.error("Database error while fetching users for sync: %s", e)
        return

    success_count = 0
    fail_count = 0

    for row in users:
        user_id = row["user_id"]

        try:
            await apply_level_permissions(bot, main_group_id, user_id)
            success_count += 1
        except error.RetryAfter as e:
            logger.warning(
                "Rate limit hit during permission sync for user %s. Sleeping for %s seconds",
                user_id,
                e.retry_after,
            )
            await asyncio.sleep(e.retry_after + 1.0)
            # Try once more
            try:
                await apply_level_permissions(bot, main_group_id, user_id)
                success_count += 1
            except Exception as e2:
                logger.error(
                    "Failed to sync permissions for user %s after retry: %s",
                    user_id,
                    e2,
                )
                fail_count += 1
        except error.BadRequest as e:
            # Common errors: User is admin, user not found, user left the chat
            logger.debug("Skipping permission sync for user %s: %s", user_id, e)
            fail_count += 1
        except Exception as e:
            logger.error("Error setting permissions for user %s: %s", user_id, e)
            fail_count += 1

        # Standard safety delay
        await asyncio.sleep(0.1)

    logger.info(
        "Finished background permission sync. Success: %s, Failed/Skipped: %s",
        success_count,
        fail_count,
    )
