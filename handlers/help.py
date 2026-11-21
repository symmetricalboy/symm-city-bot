from telegram import Update
from telegram.ext import CallbackContext
from database import db


async def help_cmd(update: Update, context: CallbackContext):
    """Handler for /help (/info, /about) to show general bot info."""
    msg = (
        "🏙️ **About symm.city Bot** 🤖\n\n"
        "Welcome to **symm.city**! I manage community engagement, CXP games & charts, translation, time zones, and the AI assistant.\n\n"
        "**Features & Quick Commands:**\n"
        "• **CXP & Stats**: Earn CXP by chatting and reacting. Check stats with `/level` (`/lvl`, `/xp`, `/rank`, `/bal`).\n"
        "• **Leaderboards & Podiums**: View the Top 3 Podium graphic & Top 10 rankings globally or by timeframe with `/leaderboard` (`/lb`, `/top`, `/ranks`) and `/contest` (`/event`).\n"
        "• **CXP History & Charts**: View balance graphs, highs/lows, and source breakdowns with `/history` (`/h`, `/hist`, `/chart`, `/pnl`).\n"
        "• **Games & Trivia**: Raid CXP with `/steal` (`/s`, `/rob`, `/yoink`), spin the slot machine with `/gamble` (`/g`, `/bet`, `/slots`, `/spin`), or DM `/question` (`/trivia`, `/quiz`, `/q`) to host group trivia (`/answer` / `/ans` / `/guess` / `/a`)!\n"
        "• **Translations**: Quick language codes (`/en`, `/es`, `/ru`, `/fa`, `/el` or `/gr`, `/ja` or `/jp`, `/uk` or `/ua`, etc.) and interactive menu (`/translate` or `/tl`).\n"
        "• **Time Zones**: Check worldwide or member times with `/time` (`/tz`, `/now`) and save yours with `/settime` (`/settz`).\n"
        "• **AI Assistant, Image, GIF & Video Studio**: Powered by Gemini 4 Neon, Nano Banana 2.1 & Gemini Omni 1.2 Flash — chat, analyze media, generate/edit artwork, create fluid animated GIFs, or generate/edit/extend HD videos with sound contextually (`/ask`, `/?`, `/ai`, `/chat`, or just reply to anything I say!).\n\n"
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
        "Type `/commands` (`/cmds` or `/aliases`) for the full list of commands and shortcuts. Visit **symm.city**!"
    )

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        message_thread_id=update.message.message_thread_id,
        text=msg,
        parse_mode="Markdown",
    )


async def commands_cmd(update: Update, context: CallbackContext):
    """Handler for /commands (/cmds, /cmd, /aliases) to show all commands and shorthand aliases."""
    msg = (
        "🌟 **symm.city Bot Commands & Shortcuts** 🌟\n\n"
        "**Community & CXP:**\n"
        "• `/level` (aliases: `/lvl`, `/xp`, `/cxp`, `/stats`, `/rank`, `/bal`, `/me`) — View your own or another user's CXP, level, and rank.\n"
        "• `/leaderboard` (aliases: `/lb`, `/top`, `/ranks`, `/leaders`) — View the Top 3 Podium image & Top 10 CXP leaders (`/lb`, `/lb today`, `/lb week`, `/lb month`, `/lb contest`).\n"
        "• `/history` (aliases: `/h`, `/hist`, `/chart`, `/charts`, `/graph`, `/ledger`, `/pnl`) — View CXP history charts, highs/lows, and source breakdowns (`/h today`, `/chart @user month` in any order).\n"
        "• `/contest` (aliases: `/event`, `/tourney`) — View the current contest leaderboard.\n"
        "• `/setchannel` (alias: `/linkchannel`) — Link your Telegram channel (DM first) so channel posts earn CXP on your account.\n\n"
        "**Games & Trivia:**\n"
        "• `/steal` (aliases: `/s`, `/rob`, `/yoink`, `/heist`, `/mug`) — Steal 25-100 CXP from a random member with available CXP (1-hour cooldown).\n"
        "• `/gamble <amount|all>` (aliases: `/g`, `/bet`, `/slots`, `/slot`, `/spin`) — Spin the 5-line CXP slot machine (1-hour cooldown, better odds on larger bets).\n"
        "• `/question` (aliases: `/trivia`, `/quiz`, `/q`) — DM the bot to create a CXP trivia question (or run in group to view/get a DM button).\n"
        "• `/answer <guess>` (aliases: `/ans`, `/guess`, `/solve`, `/a`) — Answer the active trivia question (or run with no guess to view it).\n"
        "• `/cancel` (alias: `/abort`) — Cancel the active trivia question (question author or admins).\n\n"
        "**AI & Utilities:**\n"
        "• `/ask <prompt>` (aliases: `/?`, `/ai`, `/chat`, `/symm`, `/bot`) — Chat with the AI, generate/edit images, create fluid animated GIFs, or generate/edit/extend HD videos with sound contextually (you can also just reply directly to anything the bot says!).\n"
        "• `/time [location | @user]` (aliases: `/tz`, `/clock`, `/now`) — Check current time anywhere or for a member (defaults to symm.city UTC).\n"
        "• `/settime <location>` (aliases: `/settz`, `/timezone`, `/setloc`) — Save your local time zone/location.\n"
        "• `/help` (aliases: `/info`, `/about`) — Show bot overview and CXP guide.\n"
        "• `/commands` (aliases: `/cmds`, `/cmd`, `/aliases`) — Show this command & alias reference.\n"
        "• `/rules` (aliases: `/rule`, `/guidelines`) — View the community rules.\n"
        "• `/ping` (aliases: `/status`, `/latency`, `/pong`) — Check bot latency and mode.\n\n"
        "**Translation Commands:**\n"
        "Start or reply to a message with a language code:\n"
        "`/en` (English), `/es` (Spanish), `/fr` (French), `/de` (German),\n"
        "`/it` (Italian), `/nl` (Dutch), `/pt` (Portuguese), `/ru` (Russian),\n"
        "`/uk` or `/ua` (Ukrainian), `/el` or `/gr` (Greek), `/sk` (Slovak), `/tr` (Turkish),\n"
        "`/fa` or `/farsi` (Persian), `/id` (Indonesian), `/hi` (Hindi),\n"
        "`/da` or `/dr` (Dari), `/ja` or `/jp` (Japanese), `/ar` (Arabic).\n"
        "Or reply with `/translate` (alias: `/tl`) for an interactive menu."
    )

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        message_thread_id=update.message.message_thread_id,
        text=msg,
        parse_mode="Markdown",
    )


async def rules_cmd(update: Update, context: CallbackContext):
    """Handler for /rules (/rule, /guidelines) to show the rules message."""
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

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        message_thread_id=update.message.message_thread_id,
        text=msg_text,
        parse_mode="HTML",
    )
