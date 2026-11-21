import asyncio
import html
import io
import logging
import re
from PIL import Image
from telegram import Update
from telegram.constants import ChatAction, MessageLimit, ParseMode
from telegram.error import BadRequest
from telegram.ext import CallbackContext

from database import db
from gemini_service import get_gemini_client
from handlers.cxp import _extract_message_content_and_media, extract_image_file_id
from memory_engine import memory_engine

logger = logging.getLogger(__name__)


async def _delete_messages_job(context: CallbackContext):
    """Job to delete messages after a delay."""
    chat_id = context.job.data.get("chat_id")
    message_ids = context.job.data.get("message_ids", [])
    for msg_id in message_ids:
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=msg_id)
        except Exception as e:
            logger.warning("Failed to auto-delete message %s: %s", msg_id, e)


def _normalize_image_sync(raw: bytes) -> tuple[bytes, str]:
    img = Image.open(io.BytesIO(raw))
    fmt = (img.format or "PNG").upper()
    max_dim = max(img.size)
    if fmt in ("JPEG", "PNG", "WEBP") and max_dim <= 2048 and len(raw) <= 4_000_000:
        mime_map = {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
        }
        return raw, mime_map[fmt]

    if max_dim > 2048:
        scale = 2048 / max_dim
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, resample=Image.Resampling.LANCZOS)

    out = io.BytesIO()
    if img.mode in ("RGBA", "LA", "P") and len(raw) <= 2_500_000:
        img.convert("RGBA").save(out, format="PNG")
        return out.getvalue(), "image/png"
    img.convert("RGB").save(out, format="JPEG", quality=92)
    return out.getvalue(), "image/jpeg"


def _prepare_telegram_photo_sync(raw: bytes, mime: str) -> tuple[bytes, str]:
    """Ensure generated image fits comfortably within Telegram's 10MB / dimension limits and uploads fast."""
    try:
        img = Image.open(io.BytesIO(raw))
        max_dim = max(img.size)
        if max_dim <= 2048 and len(raw) <= 3_500_000:
            ext = "jpg" if "jpeg" in (mime or "").lower() else "png"
            return raw, f"symm_image.{ext}"

        if max_dim > 2048:
            scale = 2048 / max_dim
            img = img.resize(
                (int(img.size[0] * scale), int(img.size[1] * scale)),
                resample=Image.Resampling.LANCZOS,
            )
        out = io.BytesIO()
        img.convert("RGB").save(out, format="JPEG", quality=93, optimize=True)
        return out.getvalue(), "symm_image.jpg"
    except Exception:
        ext = "jpg" if "jpeg" in (mime or "").lower() else "png"
        return raw, f"symm_image.{ext}"


async def _download_telegram_image(bot, file_id: str | None) -> tuple[bytes, str] | None:
    """Download a Telegram image `file_id` with bounded timeout and normalize off the event loop."""
    if not bot or not file_id:
        return None
    try:
        tg_file = await asyncio.wait_for(bot.get_file(file_id), timeout=12.0)
        raw = bytes(
            await asyncio.wait_for(tg_file.download_as_bytearray(), timeout=20.0)
        )
        if not raw:
            return None
        return await asyncio.to_thread(_normalize_image_sync, raw)
    except Exception as e:
        logger.warning("Failed downloading Telegram image %s: %s", file_id, e)
        return None


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
        "image_file_id": extract_image_file_id(reply),
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
    """Shared execution flow for `/ask`, `/?`, `@bot` mentions, and direct replies to the bot."""
    if not get_gemini_client():
        await update.message.reply_text(
            "The AI agent is currently offline. (API key not configured).",
        )
        return

    is_forum = bool(getattr(update.effective_chat, "is_forum", False))
    thread_id = update.message.message_thread_id
    send_thread_id = thread_id if is_forum else None
    action_thread_id = (thread_id or 1) if is_forum else None
    chat_id = update.effective_chat.id

    # Maintain a live Telegram chat action indicator ("typing..." -> "sending photo..." during image gen/edit).
    # IMPORTANT: In non-forum supergroups, replying to a message sets `message.message_thread_id` to the reply-chain
    # root ID; passing that to `send_chat_action` hides the indicator from the main group view, so `action_thread_id`
    # MUST be `None` whenever `is_forum` is False!
    current_action = [ChatAction.TYPING]
    action_switch_event = asyncio.Event()
    stop_action = False

    async def _action_pulse_loop():
        while not stop_action:
            action_switch_event.clear()
            sent_action = current_action[0]
            try:
                await context.bot.send_chat_action(
                    chat_id=chat_id,
                    action=sent_action,
                    message_thread_id=action_thread_id,
                )
            except asyncio.CancelledError:
                break
            except Exception as act_err:
                logger.debug("send_chat_action skipped: %s", act_err)
            if stop_action or current_action[0] != sent_action:
                continue
            try:
                await asyncio.wait_for(action_switch_event.wait(), timeout=4.0)
            except asyncio.CancelledError:
                break
            except asyncio.TimeoutError:
                pass

    action_task = asyncio.create_task(_action_pulse_loop())
    await asyncio.sleep(0)

    async def _switch_to_upload_photo():
        current_action[0] = ChatAction.UPLOAD_PHOTO
        action_switch_event.set()

    async def _switch_to_upload_animation():
        current_action[0] = ChatAction.UPLOAD_VIDEO
        action_switch_event.set()

    def _stop_action_pulse():
        nonlocal stop_action
        stop_action = True
        action_switch_event.set()
        if action_task and not action_task.done():
            action_task.cancel()

    try:
        asker_id, asker_name, asker_username = await _resolve_asker_identity(update)

        # Gather any directly attached image on the message and/or replied-to message
        attached_images: list[tuple[bytes, str]] = []
        seen_fids: set[str] = set()

        msg_fid = extract_image_file_id(update.message)
        if msg_fid:
            seen_fids.add(msg_fid)
            dl = await _download_telegram_image(context.bot, msg_fid)
            if dl:
                attached_images.append(dl)

        reply_fid = reply_ctx.get("image_file_id") if reply_ctx else None
        if reply_fid and reply_fid not in seen_fids:
            seen_fids.add(reply_fid)
            dl = await _download_telegram_image(context.bot, reply_fid)
            if dl:
                attached_images.append(dl)

        async def _fetch_chat_msg_image(target_msg_id: int) -> tuple[bytes, str] | None:
            fid_map = await db.get_message_media_file_ids(chat_id, [target_msg_id])
            fid = fid_map.get(int(target_msg_id))
            if not fid:
                return None
            return await _download_telegram_image(context.bot, fid)

        ai_res = await asyncio.wait_for(
            memory_engine.answer_ask_query(
                chat_id=chat_id,
                asker_id=asker_id,
                asker_name=asker_name,
                asker_username=asker_username,
                question=question,
                reply_context=reply_ctx,
                attached_images=attached_images if attached_images else None,
                fetch_message_image=_fetch_chat_msg_image,
                on_image_action=_switch_to_upload_photo,
                on_gif_action=_switch_to_upload_animation,
            ),
            timeout=165.0,
        )

        _stop_action_pulse()

        answer_text = (ai_res.get("text") or "").strip()
        gen_img_bytes: bytes | None = ai_res.get("image_bytes")
        gen_img_mime: str = ai_res.get("image_mime") or "image/png"
        media_kind: str = ai_res.get("media_kind") or (
            "animation" if gen_img_mime == "image/gif" else "photo"
        )
        is_gif_output = bool(gen_img_bytes and media_kind == "animation")

        if is_gif_output:
            gif_w = ai_res.get("width")
            gif_h = ai_res.get("height")
            try:
                await context.bot.send_chat_action(
                    chat_id=chat_id,
                    action=ChatAction.UPLOAD_VIDEO,
                    message_thread_id=action_thread_id,
                )
            except Exception:
                pass

            if len(answer_text) <= MessageLimit.CAPTION_LENGTH:
                try:
                    sent_msg = await context.bot.send_animation(
                        chat_id=chat_id,
                        message_thread_id=send_thread_id,
                        animation=io.BytesIO(gen_img_bytes),
                        filename="symm_animation.gif",
                        width=gif_w,
                        height=gif_h,
                        caption=answer_text or None,
                        parse_mode=ParseMode.HTML if answer_text else None,
                        reply_to_message_id=update.message.message_id,
                    )
                except BadRequest as html_err:
                    logger.warning("HTML caption fallback in /ask animation (%s)", html_err)
                    plain_cap = html.unescape(re.sub(r"<[^>]+>", "", answer_text)).strip()
                    sent_msg = await context.bot.send_animation(
                        chat_id=chat_id,
                        message_thread_id=send_thread_id,
                        animation=io.BytesIO(gen_img_bytes),
                        filename="symm_animation.gif",
                        width=gif_w,
                        height=gif_h,
                        caption=plain_cap[: MessageLimit.CAPTION_LENGTH] or None,
                        reply_to_message_id=update.message.message_id,
                    )
            else:
                sent_msg = await context.bot.send_animation(
                    chat_id=chat_id,
                    message_thread_id=send_thread_id,
                    animation=io.BytesIO(gen_img_bytes),
                    filename="symm_animation.gif",
                    width=gif_w,
                    height=gif_h,
                    reply_to_message_id=update.message.message_id,
                )
                try:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        message_thread_id=send_thread_id,
                        text=answer_text[:3900],
                        parse_mode=ParseMode.HTML,
                        reply_to_message_id=sent_msg.message_id,
                    )
                except BadRequest:
                    plain_fallback = html.unescape(re.sub(r"<[^>]+>", "", answer_text))
                    await context.bot.send_message(
                        chat_id=chat_id,
                        message_thread_id=send_thread_id,
                        text=plain_fallback[:3900],
                        reply_to_message_id=sent_msg.message_id,
                    )
        elif gen_img_bytes:
            upload_bytes, fname = await asyncio.to_thread(
                _prepare_telegram_photo_sync, gen_img_bytes, gen_img_mime
            )
            try:
                await context.bot.send_chat_action(
                    chat_id=chat_id,
                    action=ChatAction.UPLOAD_PHOTO,
                    message_thread_id=action_thread_id,
                )
            except Exception:
                pass

            if len(answer_text) <= MessageLimit.CAPTION_LENGTH:
                try:
                    sent_msg = await context.bot.send_photo(
                        chat_id=chat_id,
                        message_thread_id=send_thread_id,
                        photo=io.BytesIO(upload_bytes),
                        filename=fname,
                        caption=answer_text or None,
                        parse_mode=ParseMode.HTML if answer_text else None,
                        reply_to_message_id=update.message.message_id,
                    )
                except BadRequest as html_err:
                    logger.warning("HTML caption fallback in /ask photo (%s)", html_err)
                    plain_cap = html.unescape(re.sub(r"<[^>]+>", "", answer_text)).strip()
                    sent_msg = await context.bot.send_photo(
                        chat_id=chat_id,
                        message_thread_id=send_thread_id,
                        photo=io.BytesIO(upload_bytes),
                        filename=fname,
                        caption=plain_cap[: MessageLimit.CAPTION_LENGTH] or None,
                        reply_to_message_id=update.message.message_id,
                    )
            else:
                sent_msg = await context.bot.send_photo(
                    chat_id=chat_id,
                    message_thread_id=send_thread_id,
                    photo=io.BytesIO(upload_bytes),
                    filename=fname,
                    reply_to_message_id=update.message.message_id,
                )
                try:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        message_thread_id=send_thread_id,
                        text=answer_text[:3900],
                        parse_mode=ParseMode.HTML,
                        reply_to_message_id=sent_msg.message_id,
                    )
                except BadRequest:
                    plain_fallback = html.unescape(re.sub(r"<[^>]+>", "", answer_text))
                    await context.bot.send_message(
                        chat_id=chat_id,
                        message_thread_id=send_thread_id,
                        text=plain_fallback[:3900],
                        reply_to_message_id=sent_msg.message_id,
                    )
        else:
            if len(answer_text) > 3900:
                answer_text = answer_text[:3897] + "..."

            try:
                sent_msg = await context.bot.send_message(
                    chat_id=chat_id,
                    message_thread_id=send_thread_id,
                    text=answer_text,
                    parse_mode=ParseMode.HTML,
                    reply_to_message_id=update.message.message_id,
                )
            except BadRequest as html_err:
                logger.warning("HTML parse fallback in /ask (%s); retrying plain text.", html_err)
                plain_fallback = html.unescape(re.sub(r"<[^>]+>", "", answer_text))
                sent_msg = await context.bot.send_message(
                    chat_id=chat_id,
                    message_thread_id=send_thread_id,
                    text=plain_fallback[:3900],
                    reply_to_message_id=update.message.message_id,
                )

        # Archive both the user's message and the bot's response (including any Telegram image file_id)
        # into `group_chat_messages` so multi-turn follow-ups & image edits flow seamlessly!
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
                    media_type=media_type or ("photo" if msg_fid else None),
                    media_file_id=msg_fid,
                    created_at=update.message.date,
                )
                plain_bot_reply = html.unescape(re.sub(r"<[^>]+>", "", answer_text)).strip()
                if is_gif_output:
                    bot_archive_text = f"[Generated GIF] {plain_bot_reply}".strip()
                    bot_media_type = "animation"
                elif gen_img_bytes:
                    bot_archive_text = f"[Generated Image] {plain_bot_reply}".strip()
                    bot_media_type = "photo"
                else:
                    bot_archive_text = plain_bot_reply
                    bot_media_type = None
                bot_sent_fid = extract_image_file_id(sent_msg)
                bot_me = context.bot
                await db.record_group_chat_message(
                    chat_id=chat_id,
                    message_id=sent_msg.message_id,
                    user_id=bot_me.id,
                    sender_name=getattr(bot_me, "first_name", None) or "symm.city bot",
                    sender_username=getattr(bot_me, "username", None) or "symmcitybot",
                    content=bot_archive_text or ("[Generated GIF]" if is_gif_output else "[Generated Image]"),
                    is_bot=True,
                    reply_to_message_id=update.message.message_id,
                    thread_id=thread_id,
                    media_type=bot_media_type,
                    media_file_id=bot_sent_fid,
                    created_at=sent_msg.date,
                )
                memory_engine.notify_new_message()
        except Exception as log_err:
            logger.warning("Failed to archive /ask exchange: %s", log_err)

    except asyncio.TimeoutError:
        _stop_action_pulse()
        logger.warning("Timed out handling AI chat request for chat %s", chat_id)
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=send_thread_id,
            text="⏳ That request timed out waiting on the AI model — please try again!",
            reply_to_message_id=update.message.message_id,
        )
    except Exception as e:
        _stop_action_pulse()
        logger.error("Error handling AI chat response: %s", e, exc_info=True)
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=send_thread_id,
            text="My connection is experiencing interference right now. Please try again in a moment.",
            reply_to_message_id=update.message.message_id,
        )
    finally:
        _stop_action_pulse()


async def ask_cmd(update: Update, context: CallbackContext):
    """Handle `/ask`, `/?`, `/ai`, `/chat`, `/symm`, and `/bot` (text, media captions, and replies)."""
    if not update.effective_user or not update.message or not update.effective_chat:
        return

    raw_text = (update.message.text or update.message.caption or "").strip()
    parts = raw_text.split(maxsplit=1)
    question = parts[1].strip() if len(parts) > 1 else ""
    if not question and context.args:
        question = " ".join(context.args).strip()

    bot_id = getattr(context.bot, "id", None)
    reply_ctx = _extract_reply_context(update, bot_id=bot_id)
    has_direct_image = bool(
        extract_image_file_id(update.message)
        or (reply_ctx and reply_ctx.get("image_file_id"))
    )

    # If the user replied to a message with bare `/ask` or `/?`, treat that message as the prompt
    if not question and reply_ctx and reply_ctx.get("text"):
        question = reply_ctx["text"]
    elif not question and has_direct_image:
        question = "Take a look at this image and tell me what you notice!"

    if not question:
        await update.message.reply_text(
            "Please provide a prompt after `/ask` or `/?` (or attach/reply to an image, or reply to me directly!), e.g.:\n"
            "`/? Draw a neon cyberpunk skyline of symm.city`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    is_forum = bool(getattr(update.effective_chat, "is_forum", False))
    _, media_type, _ = _extract_message_content_and_media(update.message, is_forum=is_forum)

    await _run_ask_flow(
        update,
        context,
        question=question,
        raw_archive_text=raw_text or f"/ask {question}",
        reply_ctx=reply_ctx,
        media_type=media_type,
    )


async def reply_to_bot_cmd(update: Update, context: CallbackContext):
    """
    Respond contextually (including text, vision, image generation, & image editing) whenever
    a member replies directly to anything sent by the bot OR mentions `@<bot_username>` in the group.
    """
    msg = update.message
    if not update.effective_user or not msg or not update.effective_chat:
        return

    if getattr(update.effective_user, "is_bot", False) and not msg.sender_chat:
        return

    bot_id = getattr(context.bot, "id", None)
    bot_username = (getattr(context.bot, "username", None) or "").lower()

    is_target_this_bot = False
    reply = msg.reply_to_message
    if reply and not _is_implicit_topic_root(update, reply):
        reply_from = getattr(reply, "from_user", None)
        if reply_from:
            reply_username = (getattr(reply_from, "username", None) or "").lower()
            if (bot_id is not None and reply_from.id == bot_id) or (
                bool(bot_username) and reply_username == bot_username
            ):
                is_target_this_bot = True

    raw_body = (msg.text or msg.caption or "").strip()
    is_bot_mentioned = bool(
        bot_username and re.search(rf"@{re.escape(bot_username)}\b", raw_body, flags=re.IGNORECASE)
    )

    if not is_target_this_bot and not is_bot_mentioned:
        return

    is_forum = bool(getattr(update.effective_chat, "is_forum", False))
    content, media_type, _ = _extract_message_content_and_media(msg, is_forum=is_forum)
    if not content:
        return

    # Strip leading @bot_username if the user addressed the bot via @mention
    cleaned_question = content
    if is_bot_mentioned and bot_username:
        cleaned_question = re.sub(
            rf"@{re.escape(bot_username)}\b\s*[,:-]?\s*",
            "",
            cleaned_question,
            count=1,
            flags=re.IGNORECASE,
        ).strip() or content

    reply_ctx = _extract_reply_context(update, bot_id=bot_id)
    await _run_ask_flow(
        update,
        context,
        question=cleaned_question,
        raw_archive_text=content,
        reply_ctx=reply_ctx,
        media_type=media_type,
    )
