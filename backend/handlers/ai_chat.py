import html
import logging
import re
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import CallbackContext

from database import db
from gemini_service import get_gemini_client
from handlers.cxp import _extract_message_content_and_media
from memory_engine import memory_engine

logger = logging.getLogger(__name__)


async def _typing_indicator_job(context: CallbackContext):
    """Job to continually send typing indicator while Gemini 4 Neon thinks & searches."""
    chat_id = context.job.data.get("chat_id")
    message_thread_id = context.job.data.get("message_thread_id")

    try:
        await context.bot.send_chat_action(
            chat_id=chat_id,
            action=ChatAction.TYPING,
            message_thread_id=message_thread_id,
        )
    except Exception as e:
        logger.warning("Failed to send typing action: %s", e)


async def _delete_messages_job(context: CallbackContext):
    """Job to delete messages after a delay."""
    chat_id = context.job.data.get("chat_id")
    message_ids = context.job.data.get("message_ids", [])
    for msg_id in message_ids:
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=msg_id)
        except Exception as e:
            logger.warning("Failed to auto-delete message %s: %s", msg_id, e)


def _is_implicit_topic_root(update: Update, reply) -> bool:
    """Return True if `reply` is merely Telegram's implicit forum topic root rather than an explicit user reply."""
    if getattr(reply, "forum_topic_created", None) is not None:
        return True
    is_forum = bool(getattr(update.effective_chat, "is_forum", False))
    if not is_forum or not update.message:
        return False
    thread_id = update.message.message_thread_id
    return (thread_id is None and reply.message_id == 1) or (
        thread_id is not None and reply.message_id == thread_id
    )


def _extract_reply_context(update: Update, bot_id: int | None = None) -> dict | None:
    """Extract metadata & text of a replied-to message (ignoring implicit forum topic roots)."""
    msg = update.message
    if not msg or not msg.reply_to_message:
        return None

    reply = msg.reply_to_message
    if _is_implicit_topic_root(update, reply):
        return None

    is_forum = bool(getattr(update.effective_chat, "is_forum", False))
    reply_text, _, _ = _extract_message_content_and_media(reply, is_forum=is_forum)

    author_id = None
    author_name = "Unknown"

    if reply.sender_chat:
        author_id = reply.sender_chat.id
        author_name = (
            reply.sender_chat.title
            or reply.sender_chat.username
            or f"Channel {author_id}"
        )
    elif reply.from_user:
        author_id = reply.from_user.id
        first = reply.from_user.first_name or ""
        last = getattr(reply.from_user, "last_name", None) or ""
        author_name = f"{first} {last}".strip() or reply.from_user.username or str(author_id)

    is_self_bot = bool(bot_id is not None and author_id == bot_id)

    return {
        "message_id": reply.message_id,
        "author_id": author_id,
        "author_name": author_name,
        "text": reply_text,
        "is_self_bot": is_self_bot,
    }


async def _resolve_asker_identity(update: Update) -> tuple[int, str, str | None]:
    """Resolve the asking member's canonical user_id, display name, and username (handling linked channels)."""
    msg = update.message
    if msg and msg.sender_chat and getattr(msg.sender_chat, "type", "") == "channel":
        channel_id = msg.sender_chat.id
        owner_id = await db.get_channel_owner(channel_id)
        if owner_id:
            db_user = await db.get_user(owner_id)
            if db_user:
                return (
                    int(owner_id),
                    db_user.get("display_name") or db_user.get("username") or f"User {owner_id}",
                    db_user.get("username"),
                )
        return (
            int(channel_id),
            msg.sender_chat.title or msg.sender_chat.username or f"Channel {channel_id}",
            msg.sender_chat.username,
        )

    u = update.effective_user
    first = getattr(u, "first_name", "") or ""
    last = getattr(u, "last_name", "") or ""
    full_name = f"{first} {last}".strip() or getattr(u, "username", None) or f"User {u.id}"
    return int(u.id), full_name, getattr(u, "username", None)


async def _run_ask_flow(
    update: Update,
    context: CallbackContext,
    *,
    question: str,
    raw_archive_text: str,
    reply_ctx: dict | None,
    media_type: str | None = None,
):
    """Shared execution flow for `/ask` and direct replies to any message sent by the bot."""
    if not get_gemini_client():
        await update.message.reply_text(
            "The AI agent is currently offline. (API key not configured).",
        )
        return

    thread_id = update.message.message_thread_id
    chat_id = update.effective_chat.id

    action_thread_id = thread_id
    if update.effective_chat.is_forum and thread_id is None:
        action_thread_id = 1

    try:
        await context.bot.send_chat_action(
            chat_id=chat_id,
            action=ChatAction.TYPING,
            message_thread_id=action_thread_id,
        )
    except Exception:
        pass

    typing_job = context.job_queue.run_repeating(
        _typing_indicator_job,
        interval=4,
        first=4,
        data={
            "chat_id": chat_id,
            "message_thread_id": action_thread_id,
        },
    )

    try:
        asker_id, asker_name, asker_username = await _resolve_asker_identity(update)

        answer_text = await memory_engine.answer_ask_query(
            chat_id=chat_id,
            asker_id=asker_id,
            asker_name=asker_name,
            asker_username=asker_username,
            question=question,
            reply_context=reply_ctx,
        )

        if len(answer_text) > 3900:
            answer_text = answer_text[:3897] + "..."

        try:
            sent_msg = await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=thread_id,
                text=answer_text,
                parse_mode=ParseMode.HTML,
                reply_to_message_id=update.message.message_id,
            )
        except BadRequest as html_err:
            logger.warning("HTML parse fallback in /ask (%s); retrying plain text.", html_err)
            plain_fallback = html.unescape(re.sub(r"<[^>]+>", "", answer_text))
            sent_msg = await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=thread_id,
                text=plain_fallback[:3900],
                reply_to_message_id=update.message.message_id,
            )

        # Archive both the user's message and the bot's response into the floating
        # context window & continuous memory queue so follow-up conversations flow naturally.
        try:
            config = await db.get_config()
            main_group_id = config.get("main_group_id") if config else None
            if main_group_id and chat_id == main_group_id:
                await db.record_group_chat_message(
                    chat_id=chat_id,
                    message_id=update.message.message_id,
                    user_id=asker_id,
                    sender_name=asker_name,
                    sender_username=asker_username,
                    content=raw_archive_text,
                    is_bot=False,
                    reply_to_message_id=reply_ctx["message_id"] if reply_ctx else None,
                    thread_id=thread_id,
                    media_type=media_type,
                    created_at=update.message.date,
                )
                plain_bot_reply = html.unescape(re.sub(r"<[^>]+>", "", answer_text)).strip()
                bot_me = context.bot
                await db.record_group_chat_message(
                    chat_id=chat_id,
                    message_id=sent_msg.message_id,
                    user_id=bot_me.id,
                    sender_name=getattr(bot_me, "first_name", None) or "symm.city bot",
                    sender_username=getattr(bot_me, "username", None) or "symmcitybot",
                    content=plain_bot_reply,
                    is_bot=True,
                    reply_to_message_id=update.message.message_id,
                    thread_id=thread_id,
                    created_at=sent_msg.date,
                )
                memory_engine.notify_new_message()
        except Exception as log_err:
            logger.warning("Failed to archive /ask exchange: %s", log_err)

    except Exception as e:
        logger.error("Error handling AI chat response: %s", e, exc_info=True)
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=thread_id,
            text="My connection is experiencing interference right now. Please try again in a moment.",
            reply_to_message_id=update.message.message_id,
        )
    finally:
        if typing_job:
            typing_job.schedule_removal()


async def ask_cmd(update: Update, context: CallbackContext):
    """Handle the explicit `/ask` command."""
    if not update.effective_user or not update.message or not update.effective_chat:
        return

    raw_text = (update.message.text or update.message.caption or "").strip()
    parts = raw_text.split(maxsplit=1)
    question = parts[1].strip() if len(parts) > 1 else ""
    if not question and context.args:
        question = " ".join(context.args).strip()

    bot_id = getattr(context.bot, "id", None)
    reply_ctx = _extract_reply_context(update, bot_id=bot_id)

    # If the user replied to a message with bare `/ask`, treat that message as the prompt
    if not question and reply_ctx and reply_ctx.get("text"):
        question = reply_ctx["text"]

    if not question:
        await update.message.reply_text(
            "Please provide a question after `/ask` (or reply to any message from me directly!), e.g.:\n"
            "`/ask What's everyone been talking about today?`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    await _run_ask_flow(
        update,
        context,
        question=question,
        raw_archive_text=raw_text or f"/ask {question}",
        reply_ctx=reply_ctx,
    )


async def reply_to_bot_cmd(update: Update, context: CallbackContext):
    """
    Respond exactly as if `/ask` was used whenever a member replies directly
    to anything sent by the bot in the group chat.
    """
    msg = update.message
    if not update.effective_user or not msg or not update.effective_chat:
        return

    # Ignore actual bots so we never loop; allow channel-linked senders (`msg.sender_chat`)
    if getattr(update.effective_user, "is_bot", False) and not msg.sender_chat:
        return

    reply = msg.reply_to_message
    if not reply or _is_implicit_topic_root(update, reply):
        return

    reply_from = getattr(reply, "from_user", None)
    if not reply_from:
        return

    bot_id = getattr(context.bot, "id", None)
    bot_username = (getattr(context.bot, "username", None) or "").lower()
    reply_username = (getattr(reply_from, "username", None) or "").lower()

    is_target_this_bot = (bot_id is not None and reply_from.id == bot_id) or (
        bool(bot_username) and reply_username == bot_username
    )
    if not is_target_this_bot:
        return

    is_forum = bool(getattr(update.effective_chat, "is_forum", False))
    content, media_type, _ = _extract_message_content_and_media(msg, is_forum=is_forum)
    if not content:
        return

    reply_ctx = _extract_reply_context(update, bot_id=bot_id)
    await _run_ask_flow(
        update,
        context,
        question=content,
        raw_archive_text=content,
        reply_ctx=reply_ctx,
        media_type=media_type,
    )
