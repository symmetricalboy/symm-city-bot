import html
import json
import logging
from datetime import datetime
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import CallbackContext

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

from database import db
from gemini_service import generate_json, get_gemini_client

logger = logging.getLogger(__name__)

LOCATION_SCHEMA = {
    "type": "object",
    "properties": {
        "valid": {"type": "boolean"},
        "timezone": {
            "type": "string",
            "description": "Valid IANA timezone string, e.g. 'America/Chicago', 'Europe/Athens', 'Asia/Tehran'.",
        },
        "location_name": {
            "type": "string",
            "description": "Clean formatted display name, e.g. 'Minneapolis, MN', 'Athens, Greece'.",
        },
    },
    "required": ["valid", "timezone", "location_name"],
}


def get_ordinal_suffix(day: int) -> str:
    """Return the ordinal suffix for a given day (1-31)."""
    if 11 <= (day % 100) <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def format_time_string(location_name: str, target_name: str, tz_info: ZoneInfo) -> str:
    """Format the exact output string utilizing native python time calculations."""
    now = datetime.now(tz_info)
    day = now.day
    suffix = get_ordinal_suffix(day)

    date_str = now.strftime(f"%A, %B {day}{suffix} - %I:%M%p")
    if " - 0" in date_str:
        date_str = date_str.replace(" - 0", " - ")

    if target_name:
        return (
            f"{html.escape(target_name)}\n"
            f"<b>{html.escape(location_name)}</b>\n"
            f"{date_str}"
        )
    return f"<b>{html.escape(location_name)}</b>\n{date_str}"


async def resolve_location(location_str: str) -> dict | None:
    """
    Given a location string, ask Gemini (`gemini-4-neon`) to parse it into an IANA timezone string
    and a clean display name.
    Returns {"timezone": "...", "location_name": "..."} or None if invalid.
    """
    if not get_gemini_client() or not location_str:
        return None

    prompt = (
        f"Resolve the following location or timezone string into a valid IANA timezone name "
        f"(e.g. 'America/Chicago', 'Europe/London', 'Asia/Tokyo') and a clean display name "
        f"(e.g. 'Minneapolis, MN', 'London, UK', 'Tokyo, Japan'). "
        f"If the input cannot be resolved to a real timezone, set valid=false.\n\n"
        f"Location input: {location_str}"
    )

    try:
        parsed = await generate_json(
            prompt,
            LOCATION_SCHEMA,
            thinking_level="low",
        )
        if (
            parsed
            and parsed.get("valid")
            and parsed.get("timezone")
            and parsed.get("location_name")
        ):
            # Validate ZoneInfo resolves cleanly
            ZoneInfo(parsed["timezone"])
            return {
                "timezone": parsed["timezone"],
                "location_name": parsed["location_name"],
            }
        return None
    except Exception as e:
        logger.error("Failed to resolve location via Gemini 4 Neon: %s", e)
        return None


async def settime_cmd(update: Update, context: CallbackContext):
    """Handle the /settime <location> command."""
    if not update.effective_user or not update.message:
        return

    location_input = " ".join(context.args) if context.args else ""
    if not location_input:
        await update.message.reply_text(
            "Please provide a location. Usage: `/settime <location>`\nExample: `/settime Tokyo`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    chat_id = update.effective_chat.id
    thread_id = update.message.message_thread_id

    if not get_gemini_client():
        await update.message.reply_text(
            "The AI agent is currently offline. (API key not configured)."
        )
        return

    try:
        await context.bot.send_chat_action(
            chat_id=chat_id,
            action=ChatAction.TYPING,
            message_thread_id=thread_id if update.effective_chat.is_forum else None,
        )
    except Exception:
        pass

    resolved = await resolve_location(location_input)

    if not resolved:
        await update.message.reply_text(
            "I couldn't figure out the standard timezone for that location. Please try being more specific, "
            "like a major city near you, or a timezone name (e.g. 'Minneapolis, MN' or 'America/Chicago')."
        )
        return

    user_id = update.effective_user.id

    try:
        location_json = json.dumps(resolved)
        await db.update_user_location(user_id, location_json)
        await update.message.reply_text(
            f"✅ Success! Your location has been securely set to: <b>{html.escape(resolved['location_name'])}</b> ({html.escape(resolved['timezone'])})",
            parse_mode=ParseMode.HTML,
            reply_to_message_id=update.message.message_id,
        )
    except Exception as e:
        logger.error("Error saving location for user %s: %s", user_id, e)
        await update.message.reply_text(
            "An error occurred while trying to save your location. Please try again later.",
        )


async def time_cmd(update: Update, context: CallbackContext):
    """Handle the /time <location | @username> command."""
    if not update.effective_user or not update.message:
        return

    chat_id = update.effective_chat.id
    thread_id = update.message.message_thread_id

    target = " ".join(context.args).strip().lower() if context.args else ""

    city_aliases = [
        "",
        "symm",
        "symm.city",
        "symm city",
        "city",
        "utc",
    ]

    target_name = None
    resolved_location = None

    if target in city_aliases:
        resolved_location = {"timezone": "UTC", "location_name": "symm.city (UTC)"}
    elif target.startswith("@"):
        username = target.lstrip("@")
        user = await db.get_user_by_username(username)

        if user and user.get("location"):
            try:
                resolved_location = json.loads(user["location"])
                target_name = (
                    user.get("display_name") or f"@{user.get('username') or username}"
                )
            except json.JSONDecodeError:
                raw_location = user["location"]
                target_name = (
                    user.get("display_name") or f"@{user.get('username') or username}"
                )
                resolved_location = await resolve_location(raw_location)
        else:
            await update.message.reply_text(
                f"User @{username} has not set a location or doesn't exist.\n"
                "You can set your own location using `/settime <location>`.\n"
                "Example: `/settime Tokyo`",
                parse_mode=ParseMode.MARKDOWN,
            )
            return
    else:
        resolved_location = await resolve_location(target)

    if not resolved_location:
        await update.message.reply_text(
            "I couldn't calculate the standard timezone for that location. Please try being more specific.",
        )
        return

    try:
        tz_info = ZoneInfo(resolved_location["timezone"])
        answer_text = format_time_string(
            resolved_location["location_name"], target_name, tz_info
        )

        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=thread_id,
            text=answer_text,
            parse_mode=ParseMode.HTML,
        )

    except Exception as e:
        logger.error(
            "Error handling /time command (timezone %s): %s",
            resolved_location.get("timezone"),
            e,
        )
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=thread_id,
            text="I ran into an internal error resolving the math for that specific time zone.",
        )
