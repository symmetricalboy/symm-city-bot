from telegram import Update
from telegram.ext import CallbackContext
import asyncio
from database import db


async def help_cmd(update: Update, context: CallbackContext):
    """Handler for /help to show general bot info."""
    msg = (
        "🏙️ **About symm.city Bot** 🤖\n\n"
        "Welcome to **symm.city**! I manage community engagement, translation, time zones, and the AI assistant.\n\n"
        "**Features:**\n"
        "• **CXP System**: Earn Community Experience Points (CXP) by chatting and reacting to messages. Level up to gain reaction influence and titles!\n"
        "• **Leaderboards & Contests**: Compete for top spots globally or during special contests (`/leaderboard`, `/contest`).\n"
        "• **Translations**: Quick commands (`/en`, `/es`, `/fr`, `/pt`, `/id`, `/ru`, `/tr`, `/fa`, `/uk`) and interactive menu (`/translate`).\n"
        "• **Time Zones**: Look up times worldwide or check member times (`/time`, `/settime`).\n"
        "• **AI Assistant**: Grounded in symm's project history & Bluesky archive (`/ask <question>`).\n\n"
        "**Earning CXP:**\n"
        "• **Messages**: Earn `50 CXP` for chatting (limit 1 per minute).\n"
        "• **Reactions**: Earn or lose CXP when others react to your messages (`+50 CXP` for positive, `-50 CXP` for negative).\n"
        "• **Influence**: Higher level members multiply the CXP impact of their reactions!\n\n"
        "**Level Titles:**\n"
        "• **Lvl 1-4:** Script Kiddie\n"
        "• **Lvl 5-9:** Hash Cracker\n"
        "• **Lvl 10-19:** True Operator\n"
        "• **Lvl 20-29:** Dirty Phreak\n"
        "• **Lvl 30-39:** Ledger Forger\n"
        "• **Lvl 40-49:** Clean Splicer\n"
        "• **Lvl 50-59:** Whale Hunter\n"
        "• **Lvl 60+:** Zero-Day Broker\n\n"
        "To view all commands, type `/commands`. Visit **symm.city**!"
    )
    try:
        await update.message.delete()
    except Exception:
        pass

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        message_thread_id=update.message.message_thread_id,
        text=msg,
        parse_mode="Markdown",
    )


async def commands_cmd(update: Update, context: CallbackContext):
    """Handler for /commands to show CXP info."""
    msg = (
        "🌟 **symm.city Bot Commands** 🌟\n\n"
        "**Community & CXP:**\n"
        "• `/level` — View your own or another user's stats and rank (`/level` or `/level @username`).\n"
        "• `/leaderboard` — View the top CXP leaders (`/leaderboard` or `/leaderboard <day|week|month|contest>`).\n"
        "• `/contest` — View the current contest leaderboard.\n"
        "• `/steal` — Steal 25-100 CXP from a random member (1-hour cooldown).\n\n"
        "**AI & Utilities:**\n"
        "• `/ask <question>` — Ask the AI assistant about symm, projects (Symmetris, music), or general topics.\n"
        "• `/time <location | @username>` — Get the current time for a user or location (defaults to symm.city time).\n"
        "• `/settime <location>` — Set your personal local time zone/location.\n"
        "• `/rules` — View the community rules.\n"
        "• `/ping` — Check bot response latency and connection method.\n\n"
        "**Translation Commands:**\n"
        "Start or reply to a message with a language code:\n"
        "`/en` (English), `/es` (Spanish), `/fr` (French), `/de` (German),\n"
        "`/it` (Italian), `/nl` (Dutch), `/pt` (Portuguese), `/ru` (Russian),\n"
        "`/uk` (Ukrainian), `/el` (Greek), `/sk` (Slovak), `/tr` (Turkish),\n"
        "`/fa` (Persian), `/id` (Indonesian), `/hi` (Hindi).\n"
        "Or reply with `/translate` for an interactive menu."
    )
    try:
        await update.message.delete()
    except Exception:
        pass

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        message_thread_id=update.message.message_thread_id,
        text=msg,
        parse_mode="Markdown",
    )


async def rules_cmd(update: Update, context: CallbackContext):
    """Handler for /rules to show the rules message. Auto-deletes after 5 minutes."""
    config = await db.get_config()
    msg_text = config.get("rules_message") if config else None
    if not msg_text:
        msg_text = (
            "📜 <b>symm.city Community Rules</b>:\n\n"
            "1. Be kind.\n"
            "2. Assume good intent.\n"
            "3. Be symmetrical.\n"
            "4. Be patient.\n"
            "5. Be <i>very</i> symmetrical."
        )

    try:
        await update.message.delete()
    except Exception:
        pass

    reply_msg = await context.bot.send_message(
        chat_id=update.effective_chat.id,
        message_thread_id=update.message.message_thread_id,
        text=msg_text,
        parse_mode="HTML",
    )

    # Schedule deletion of the bot's reply after 5 minutes
    async def delete_later():
        await asyncio.sleep(300)
        try:
            await reply_msg.delete()
        except Exception:
            pass

    asyncio.create_task(delete_later())
