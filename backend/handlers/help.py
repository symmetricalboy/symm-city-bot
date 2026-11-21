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
        "• **CXP History & Charts**: Track CXP gains, losses, high/low points, and visual charts across every source (`/history`, `/history today`, `/history @user month`).\n"
        "• **Games & Trivia**: Steal CXP (`/steal`), spin the CXP slot machine with better odds on larger bets (`/gamble`), or DM the bot `/question` to host CXP trivia questions in the group (`/answer`)!\n"
        "• **Leaderboards & Contests**: Compete for top spots globally or during special contests (`/leaderboard`, `/contest`).\n"
        "• **Translations**: Quick commands (`/en`, `/es`, `/fr`, `/de`, `/it`, `/nl`, `/pt`, `/ru`, `/uk`, `/el`, `/sk`, `/tr`, `/fa`, `/id`, `/hi`, `/da`, `/ja`, `/ar`) and interactive menu (`/translate`).\n"
        "• **Time Zones**: Look up times worldwide or check member times (`/time`, `/settime`).\n"
        "• **AI Assistant**: Powered by Gemini 4 Neon with continual chat awareness, long-term community memory, & Bluesky history (`/ask <question>`).\n\n"
        "**Earning CXP:**\n"
        "• **Messages**: Earn `50 CXP` for chatting (limit 1 per minute).\n"
        "• **Reactions**: Earn or lose CXP when others react to your messages (`+50 CXP` for positive, `-50 CXP` for negative).\n"
        "• **Influence**: Higher level members multiply the CXP impact of their reactions!\n\n"
        "**Level Titles:**\n"
        "• **Lvl 1-4:** Budding Symmetry\n"
        "• **Lvl 5-9:** Bilateral Balance\n"
        "• **Lvl 10-19:** Rotational Harmony\n"
        "• **Lvl 20-29:** Crystalline Order\n"
        "• **Lvl 30-39:** Exquisitely Symmetrical\n"
        "• **Lvl 40-49:** Supersymmetric\n"
        "• **Lvl 50-59:** Very Symmetrical\n"
        "• **Lvl 60+:** Pure Symmetry\n\n"
        "To view all commands, type `/commands`. Visit **symm.city**!"
    )

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
        "• `/history` — View CXP history charts, high/low points, and source breakdowns (`/history`, `/history today`, `/history @username month` in any order).\n"
        "• `/leaderboard` — View the top CXP leaders (`/leaderboard` or `/leaderboard <day|week|month|contest>`).\n"
        "• `/contest` — View the current contest leaderboard.\n\n"
        "**Games & Trivia:**\n"
        "• `/steal` — Steal up to 25-100 CXP from a random member with available CXP (1-hour cooldown).\n"
        "• `/gamble <amount>` — Bet CXP on the 5-line slot machine (1-hour cooldown, e.g. `/gamble 50`). All 3 rows and both diagonals pay out (stacking multiple wins), with better odds the more you bet!\n"
        "• `/question` (DM only) — DM the bot to create a group trivia question with one or more comma-separated correct answers, a CXP prize, and a guess fee (queued if another question is live).\n"
        "• `/answer <guess>` — Pay the guess fee to answer the active trivia question (or `/answer` alone to view it).\n"
        "• `/cancel` — Cancel the active trivia question, refunding the prize to the asker while keeping paid guess fees (admins or question author only).\n\n"
        "**AI & Utilities:**\n"
        "• `/ask <question>` — Chat with the Gemini 4 Neon AI assistant (or simply reply directly to anything the bot says!).\n"
        "• `/time <location | @username>` — Get the current time for a user or location (defaults to symm.city time).\n"
        "• `/settime <location>` — Set your personal local time zone/location.\n"
        "• `/rules` — View the community rules.\n"
        "• `/ping` — Check bot response latency and connection method.\n\n"
        "**Translation Commands:**\n"
        "Start or reply to a message with a language code:\n"
        "`/en` (English), `/es` (Spanish), `/fr` (French), `/de` (German),\n"
        "`/it` (Italian), `/nl` (Dutch), `/pt` (Portuguese), `/ru` (Russian),\n"
        "`/uk` (Ukrainian), `/el` (Greek), `/sk` (Slovak), `/tr` (Turkish),\n"
        "`/fa` (Persian), `/id` (Indonesian), `/hi` (Hindi),\n"
        "`/da` (Dari), `/ja` (Japanese), `/ar` (Arabic).\n"
        "Or reply with `/translate` for an interactive menu."
    )

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        message_thread_id=update.message.message_thread_id,
        text=msg,
        parse_mode="Markdown",
    )


async def rules_cmd(update: Update, context: CallbackContext):
    """Handler for /rules to show the rules message."""
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


    reply_msg = await context.bot.send_message(
        chat_id=update.effective_chat.id,
        message_thread_id=update.message.message_thread_id,
        text=msg_text,
        parse_mode="HTML",
    )


