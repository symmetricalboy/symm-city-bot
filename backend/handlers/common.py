import logging
from telegram import Update, ChatMember
from telegram.ext import CallbackContext
from database import db

logger = logging.getLogger(__name__)


async def resolve_username(
    input_str: str, update: Update, context: CallbackContext
) -> tuple[int | None, str | None]:
    """
    Centralized function to resolve a username, user ID, or text_mention to a User ID and Name.
    Relies on Telegram message entities, Telegram API, and DB Tracker fallback.
    Returns (user_id, user_name).
    """
    # 1. Check for explicit text_mentions (links without @ usernames) in the message
    if update.message:
        entities = update.message.parse_entities(None)
        for ent, text in entities.items():
            if ent.type == "text_mention" and ent.user:
                full_name = ent.user.first_name + (
                    f" {ent.user.last_name}" if ent.user.last_name else ""
                )
                return ent.user.id, full_name

    if not input_str:
        return None, None

    cleaned_input = input_str.strip()
    if not cleaned_input:
        return None, None

    # 2. Check if input is a numeric Telegram User ID or Channel ID
    try:
        potential_id = int(cleaned_input)
        if db.pool:
            async with db.pool.acquire() as conn:
                existing = await conn.fetchrow("SELECT * FROM users WHERE user_id = $1", potential_id)
            if existing and (existing.get("display_name") or existing.get("username")):
                name = existing.get("display_name") or (f"@{existing.get('username')}" if existing.get("username") else f"User {potential_id}")
                return potential_id, name
        # Verify via Telegram API
        try:
            chat = await context.bot.get_chat(potential_id)
            name = (
                chat.title
                or (
                    chat.first_name + (f" {chat.last_name}" if chat.last_name else "")
                    if chat.first_name
                    else None
                )
                or f"User {potential_id}"
            )
            return potential_id, name
        except Exception:
            pass
    except ValueError:
        pass

    # 3. Extract pure mention entity if message contains one
    username_str = cleaned_input
    if update.message and update.message.text:
        entities = update.message.parse_entities(["mention"])
        for ent, text in entities.items():
            username_str = text
            break

    # Format to ensure @ for Telegram API
    handle = username_str if username_str.startswith("@") else f"@{username_str}"

    # 4. Hit the Telegram API directly
    try:
        chat = await context.bot.get_chat(handle)
        name = (
            chat.title
            or (
                chat.first_name + (f" {chat.last_name}" if chat.last_name else "")
                if chat.first_name
                else None
            )
            or handle
        )
        return chat.id, name
    except Exception as e:
        logger.debug("Telegram get_chat failed for %s: %s", handle, e)

    # 5. Fallback to DB Tracker
    clean_name = username_str.lstrip("@").strip()
    user_row = await db.get_user_by_username(clean_name)
    if user_row:
        display_name = user_row.get("display_name") or (
            f"@{user_row.get('username')}" if user_row.get("username") else handle
        )
        return user_row.get("user_id"), display_name

    logger.warning(
        "Failed to resolve username %s via API and DB", username_str
    )
    return None, None


async def is_user_admin(user_id: int, chat_id: int, context: CallbackContext) -> bool:
    """
    Robustly check if a user or channel is an admin.
    Checks:
    1. Database 'is_admin' flag.
    2. Telegram API for 'administrator' or 'creator' status.
    """
    # 1. Check Database
    user_data = await db.get_user(user_id)
    if user_data and user_data.get("is_admin", False):
        return True

    # 2. Check Telegram API
    try:
        member = await context.bot.get_chat_member(chat_id, user_id)
        if member.status in (ChatMember.ADMINISTRATOR, ChatMember.OWNER):
            return True
    except Exception as e:
        logger.warning(f"Failed to check admin status for {user_id} in {chat_id}: {e}")

    return False


async def get_target(
    update: Update, context: CallbackContext
) -> tuple[int | None, str | None]:
    """Helper to resolve target from reply or mention."""
    target_id = None
    target_name = None

    # Try reply
    if update.message.reply_to_message and not getattr(
        update.message, "is_automatic_forward", False
    ):
        if not (
            update.message.is_topic_message
            and update.message.reply_to_message.message_id
            == update.message.message_thread_id
        ):
            if update.message.reply_to_message.sender_chat:
                target_id = update.message.reply_to_message.sender_chat.id
                target_name = (
                    update.message.reply_to_message.sender_chat.title or "Channel"
                )
            elif update.message.reply_to_message.from_user:
                user = update.message.reply_to_message.from_user
                target_id = user.id
                target_name = user.first_name + (
                    f" {user.last_name}" if getattr(user, "last_name", None) else ""
                )

    # Try mention
    if not target_id and context.args:
        arg_str = context.args[0]
        res_id, res_name = await resolve_username(arg_str, update, context)
        if res_id:
            target_id = res_id
            target_name = res_name

    return target_id, target_name
