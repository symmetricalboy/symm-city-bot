import html
import logging
import math
import os
import random
import re
import warnings
from telegram import (
    ChatMember,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ParseMode
from telegram.warnings import PTBUserWarning
from telegram.ext import (
    CallbackContext,
    CallbackQueryHandler,
    CommandHandler,
    ConversationHandler,
    MessageHandler,
    filters,
)

from database import db
from handlers.common import is_user_admin
from handlers.cxp import _announce_level_up, calculate_level

logger = logging.getLogger(__name__)

# Conversation states for /question DM wizard
(
    Q_STATE_QUESTION,
    Q_STATE_ANSWER,
    Q_STATE_PRIZE,
    Q_STATE_COST,
    Q_STATE_CONFIRM,
) = range(5)

ANSWER_ALLOWED_RE = re.compile(r"^[a-zA-Z0-9 ]+$")

# Slot machine symbols: (emoji, 3-of-a-kind multiplier, label)
SLOT_SYMBOLS = [
    ("🍒", 2.0, "Cherries"),
    ("🍋", 2.5, "Lemons"),
    ("🍊", 3.0, "Oranges"),
    ("🍇", 4.0, "Grapes"),
    ("🔔", 6.0, "Bells"),
    ("🍀", 10.0, "Clovers"),
    ("💎", 16.0, "Diamonds"),
    ("7️⃣", 30.0, "Jackpot Sevens"),
]

# Center-row adjacent 2-of-a-kind multipliers (when center row is not already 3-of-a-kind)
CENTER_PAIR_MULTIPLIERS = {
    0: 1.5,  # 🍒🍒
    6: 2.0,  # 💎💎
    7: 3.0,  # 7️⃣7️⃣
}

SLOT_BASE_WEIGHTS = [25.0, 21.0, 18.0, 13.5, 9.5, 6.2, 4.3, 2.5]
SLOT_HIGH_WEIGHTS = [22.0, 19.5, 17.0, 14.5, 11.0, 7.2, 5.5, 3.3]


def _get_bot_owner_id() -> int | None:
    raw = os.getenv("BOT_OWNER_ID")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _normalize_answer(raw: str) -> str | None:
    """Validate a-z 0-9 & spaces only, returning lowercase single-spaced string or None."""
    if not raw:
        return None
    stripped = raw.strip()
    if not stripped or len(stripped) > 200:
        return None
    if not ANSWER_ALLOWED_RE.fullmatch(stripped):
        return None
    normalized = " ".join(stripped.lower().split())
    return normalized if normalized else None


def _parse_signed_int_amount(raw: str) -> int | None:
    """Parse a signed integer CXP value (allowing leading +/- and commas), or None if non-numeric."""
    if not raw:
        return None
    cleaned = raw.strip().replace(",", "").replace("_", "")
    if not cleaned:
        return None
    sign = 1
    if cleaned[0] in ("+", "-"):
        if cleaned[0] == "-":
            sign = -1
        cleaned = cleaned[1:]
    if not cleaned or not cleaned.isdigit():
        return None
    try:
        return sign * int(cleaned)
    except ValueError:
        return None


def _parse_int_amount(raw: str) -> int | None:
    """Parse a non-negative integer CXP value, tolerating commas (e.g. '1,000')."""
    val = _parse_signed_int_amount(raw)
    if val is None or val < 0:
        return None
    return val


async def _resolve_group_actor(update: Update):
    """
    Resolve the acting user in a group chat, supporting linked channels and anonymous group admins.
    Returns (actor_id, actor_name, user_obj_for_levelup, is_unlinked_channel).
    """
    if not update.message:
        return None, None, None, False

    if update.effective_user and not getattr(update.effective_user, "is_bot", True):
        u = update.effective_user
        full_name = u.first_name + (f" {u.last_name}" if u.last_name else "")
        return u.id, full_name, u, False

    if update.message.sender_chat:
        sender_chat = update.message.sender_chat
        if getattr(sender_chat, "type", "") == "channel":
            owner_id = await db.get_channel_owner(sender_chat.id)
            if owner_id:
                db_user = await db.get_user(owner_id)
                owner_name = (
                    (db_user.get("display_name") if db_user else None)
                    or sender_chat.title
                    or f"User {owner_id}"
                )

                class _MockUser:
                    def __init__(self, uid: int, name: str):
                        self.id = uid
                        self.first_name = name
                        self.last_name = None
                        self.is_bot = False

                return owner_id, owner_name, _MockUser(owner_id, owner_name), False
            return (
                sender_chat.id,
                sender_chat.title or f"Channel {sender_chat.id}",
                None,
                True,
            )

        return (
            sender_chat.id,
            sender_chat.title or "Group Admin",
            None,
            False,
        )

    return None, None, None, False


async def _maybe_announce_level_up(
    context: CallbackContext,
    user_id: int,
    old_cxp: int,
    new_cxp: int,
    fallback_user_obj=None,
):
    """Announce a level up in the group if new_cxp crosses into a higher level."""
    old_level = calculate_level(old_cxp)
    new_level = calculate_level(new_cxp)
    if new_level <= old_level:
        return

    user_obj = fallback_user_obj
    if not user_obj or getattr(user_obj, "id", None) != user_id:
        try:
            user_obj = await context.bot.get_chat(user_id)
        except Exception as e:
            logger.warning(
                "Could not fetch user %s for level up announcement: %s", user_id, e
            )
            return

    await _announce_level_up(context, user_obj, new_level)


# =====================================================================
# 1. SLOT MACHINE (/gamble)
# =====================================================================


def spin_slot_machine(bet: int) -> dict:
    """
    Spin a 3-reel x 3-row slot machine with 5 paylines (3 horizontal rows + 2 diagonals).
    Higher bets smoothly improve both symbol weights and reel-alignment luck.
    """
    luck = min(1.0, math.log(max(1, bet)) / math.log(10000))
    weights = [
        base + (high - base) * luck
        for base, high in zip(SLOT_BASE_WEIGHTS, SLOT_HIGH_WEIGHTS)
    ]

    # Spin 3 independent reels; each reel displays 3 distinct symbols (no confusing vertical triples)
    reels: list[list[int]] = []
    for _ in range(3):
        pool = list(range(len(SLOT_SYMBOLS)))
        w = list(weights)
        col = []
        for _ in range(3):
            idx = random.choices(range(len(pool)), weights=w, k=1)[0]
            col.append(pool.pop(idx))
            w.pop(idx)
        reels.append(col)

    # Bet-correlated reel alignment luck boost
    nudge_prob = 0.05 + 0.11 * luck
    if random.random() < nudge_prob:
        target_row = random.choice([0, 1, 2])
        sym = reels[0][target_row]
        for r in (1, 2):
            if sym in reels[r]:
                cur_pos = reels[r].index(sym)
                reels[r][target_row], reels[r][cur_pos] = (
                    reels[r][cur_pos],
                    reels[r][target_row],
                )
            elif random.random() < (0.45 + 0.20 * luck):
                reels[r][target_row] = sym

    # Convert reels[col][row] -> grid[row][col]
    grid_idx = [[reels[col][row] for col in range(3)] for row in range(3)]
    grid_emoji = [[SLOT_SYMBOLS[idx][0] for idx in row] for row in grid_idx]

    paylines = [
        ("Top Row", [grid_idx[0][0], grid_idx[0][1], grid_idx[0][2]]),
        ("Middle Row", [grid_idx[1][0], grid_idx[1][1], grid_idx[1][2]]),
        ("Bottom Row", [grid_idx[2][0], grid_idx[2][1], grid_idx[2][2]]),
        ("Diagonal ↘", [grid_idx[0][0], grid_idx[1][1], grid_idx[2][2]]),
        ("Diagonal ↗", [grid_idx[2][0], grid_idx[1][1], grid_idx[0][2]]),
    ]

    winning_lines = []
    total_multiplier = 0.0
    center_three_hit = False

    for line_name, line_syms in paylines:
        if line_syms[0] == line_syms[1] == line_syms[2]:
            sym_idx = line_syms[0]
            emoji, mult, _ = SLOT_SYMBOLS[sym_idx]
            total_multiplier += mult
            winning_lines.append((line_name, f"{emoji} {emoji} {emoji}", mult))
            if line_name == "Middle Row":
                center_three_hit = True

    if not center_three_hit:
        center = paylines[1][1]
        pair_sym = (
            center[0]
            if center[0] == center[1]
            else (center[1] if center[1] == center[2] else None)
        )
        if pair_sym is not None and pair_sym in CENTER_PAIR_MULTIPLIERS:
            emoji = SLOT_SYMBOLS[pair_sym][0]
            pair_mult = CENTER_PAIR_MULTIPLIERS[pair_sym]
            total_multiplier += pair_mult
            winning_lines.append(("Center Pair", f"{emoji} {emoji}", pair_mult))

    if total_multiplier > 0:
        payout = max(bet + 1, int(math.ceil(bet * total_multiplier)))
    else:
        payout = 0

    return {
        "grid": grid_emoji,
        "winning_lines": winning_lines,
        "total_multiplier": total_multiplier,
        "payout": payout,
        "luck": luck,
    }


def _format_multiplier(mult: float) -> str:
    return f"{int(mult)}x" if mult == int(mult) else f"{mult:g}x"


async def gamble_cmd(update: Update, context: CallbackContext):
    """Member command: /gamble <amount>. Bet CXP on the slot machine."""
    if not update.effective_chat or not update.message:
        return

    if update.effective_chat.type == "private":
        await update.message.reply_text(
            "🎰 Please use `/gamble <amount>` inside the group chat!",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    config = await db.get_config()
    main_group_id = config.get("main_group_id") if config else None
    if not main_group_id or update.effective_chat.id != main_group_id:
        return

    thread_id = update.message.message_thread_id
    actor_id, actor_name, user_obj, is_unlinked_channel = await _resolve_group_actor(
        update
    )
    if not actor_id:
        return

    if is_unlinked_channel:
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
            "⚠️ Please link your channel to your user account first before gambling CXP.",
            reply_markup=keyboard,
        )
        return

    if actor_id < 0:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            message_thread_id=thread_id,
            text="⚠️ Anonymous group admins don't have a personal CXP balance — please switch to your personal account (or a linked channel) to gamble!",
        )
        return

    if not context.args:
        help_text = (
            "🎰 <b>symm.city Slot Machine</b>\n\n"
            "<b>Usage:</b> <code>/gamble &lt;amount&gt;</code> (e.g. <code>/gamble 50</code>)\n"
            "<i>Higher bets unlock better symbol odds &amp; reel alignment luck!</i>\n\n"
            "<b>3-in-a-Row Paytable (Rows &amp; Diagonals):</b>\n"
            "🍒 <code>2x</code>  •  🍋 <code>2.5x</code>  •  🍊 <code>3x</code>  •  🍇 <code>4x</code>\n"
            "🔔 <code>6x</code>  •  🍀 <code>10x</code>  •  💎 <code>16x</code>  •  7️⃣ <code>30x</code>\n"
            "<i>Center-line pairs of</i> 🍒 <i>(1.5x),</i> 💎 <i>(2x), or</i> 7️⃣ <i>(3x) also win!</i>"
        )
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            message_thread_id=thread_id,
            text=help_text,
            parse_mode=ParseMode.HTML,
        )
        return

    safe_name = html.escape(actor_name)
    raw_arg = context.args[0].strip().lower()
    if raw_arg in ("all", "max"):
        user_data = await db.get_user(actor_id)
        bet = (user_data.get("cxp") or 0) if user_data else 0
        if bet <= 0:
            if bet < 0:
                err_msg = (
                    f"❌ You cannot gamble, <b>{safe_name}</b> — your balance is currently negative "
                    f"(<code>{bet:,} CXP</code>), and you need at least <code>1 CXP</code> available to place a bet."
                )
            else:
                err_msg = (
                    f"❌ You cannot gamble, <b>{safe_name}</b> — you currently have <code>0 CXP</code>, "
                    f"and you need at least <code>1 CXP</code> available to place a bet."
                )
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                message_thread_id=thread_id,
                text=err_msg,
                parse_mode=ParseMode.HTML,
            )
            return
    else:
        bet = _parse_signed_int_amount(raw_arg)
        if bet is None:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                message_thread_id=thread_id,
                text="❌ Invalid bet amount! Please enter a whole number of CXP (at least <code>1</code>), e.g. <code>/gamble 50</code>.",
                parse_mode=ParseMode.HTML,
            )
            return
        if bet < 1:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                message_thread_id=thread_id,
                text=f"❌ You cannot bet <code>{bet:,} CXP</code> — your bet must be at least <code>1 CXP</code>! Example: <code>/gamble 50</code>",
                parse_mode=ParseMode.HTML,
            )
            return

    spin = spin_slot_machine(bet)
    settlement = await db.place_bet_and_settle(actor_id, bet, spin["payout"])

    if not settlement.get("ok"):
        if settlement.get("reason") == "insufficient_cxp":
            cur_cxp = settlement.get("current_cxp", 0)
            if cur_cxp < 0:
                err_msg = (
                    f"❌ You cannot bet <code>{bet:,} CXP</code>, <b>{safe_name}</b> — "
                    f"your balance is currently negative (<code>{cur_cxp:,} CXP</code>), so you have no CXP available to gamble."
                )
            elif cur_cxp == 0:
                err_msg = (
                    f"❌ You cannot bet <code>{bet:,} CXP</code>, <b>{safe_name}</b> — "
                    f"you currently have <code>0 CXP</code> available to gamble."
                )
            else:
                err_msg = (
                    f"❌ You cannot bet <code>{bet:,} CXP</code>, <b>{safe_name}</b> — "
                    f"you only have <code>{cur_cxp:,} CXP</code> available."
                )
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                message_thread_id=thread_id,
                text=err_msg,
                parse_mode=ParseMode.HTML,
            )
        else:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                message_thread_id=thread_id,
                text="❌ Could not process your bet right now. Please try again.",
            )
        return

    grid = spin["grid"]
    winning_lines = spin["winning_lines"]
    payout = spin["payout"]
    net_delta = settlement["net_delta"]
    old_cxp = settlement["old_cxp"]
    new_cxp = settlement["new_cxp"]

    grid_display = (
        f"⬛ {grid[0][0]} │ {grid[0][1]} │ {grid[0][2]} ⬛\n"
        f"▶️ {grid[1][0]} │ {grid[1][1]} │ {grid[1][2]} ◀️\n"
        f"⬛ {grid[2][0]} │ {grid[2][1]} │ {grid[2][2]} ⬛"
    )

    if payout > 0:
        lines_summary = "\n".join(
            f"✨ <b>{line_name}:</b> {icons} (<code>{_format_multiplier(mult)}</code>)"
            for line_name, icons, mult in winning_lines
        )
        header = (
            "🚨 <b>JACKPOT!! SYMM SLOTS</b> 🚨"
            if spin["total_multiplier"] >= 16.0
            else "🎰 <b>SYMM SLOTS — WINNER!</b> 🎰"
        )
        msg = (
            f"{header}\n"
            f"👤 <b>{safe_name}</b> bet <code>{bet:,} CXP</code>\n"
            f"━━━━━━━━━━━━━\n"
            f"{grid_display}\n"
            f"━━━━━━━━━━━━━\n"
            f"{lines_summary}\n\n"
            f"🏆 <b>Payout:</b> <code>{payout:,} CXP</code> (Net: <code>+{net_delta:,} CXP</code>)\n"
            f"💰 <b>Balance:</b> <code>{new_cxp:,} CXP</code>"
        )
    else:
        msg = (
            f"🎰 <b>SYMM SLOTS</b> 🎰\n"
            f"👤 <b>{safe_name}</b> bet <code>{bet:,} CXP</code>\n"
            f"━━━━━━━━━━━━━\n"
            f"{grid_display}\n"
            f"━━━━━━━━━━━━━\n"
            f"💸 <b>No match!</b> Lost <code>{bet:,} CXP</code>.\n"
            f"💰 <b>Balance:</b> <code>{new_cxp:,} CXP</code>"
        )

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        message_thread_id=thread_id,
        text=msg,
        parse_mode=ParseMode.HTML,
    )

    if net_delta > 0:
        await _maybe_announce_level_up(
            context, actor_id, old_cxp, new_cxp, fallback_user_obj=user_obj
        )


# =====================================================================
# 2. GROUP TRIVIA QUESTIONS (/question, /answer, /cancel)
# =====================================================================


def _clear_question_flow(context: CallbackContext):
    if context.user_data is not None:
        context.user_data.pop("in_question_flow", None)
        context.user_data.pop("question_draft", None)


async def _resolve_asker_display_name(asker_id: int, cached_name: str | None, bot) -> str:
    if cached_name:
        return cached_name
    user_row = await db.get_user(asker_id)
    if user_row:
        if user_row.get("display_name"):
            return user_row["display_name"]
        if user_row.get("username"):
            return f"@{user_row['username']}"
    try:
        chat = await bot.get_chat(asker_id)
        if chat.first_name:
            return chat.first_name + (f" {chat.last_name}" if chat.last_name else "")
        if chat.title:
            return chat.title
    except Exception:
        pass
    return f"User {asker_id}"


def _build_group_question_html(question: dict, asker_name: str) -> str:
    safe_asker = html.escape(asker_name)
    safe_q = html.escape(question["question_text"])
    prize = question["prize_cxp"]
    cost = question["guess_cost"]
    cost_str = f"{cost:,} CXP" if cost > 0 else "FREE (0 CXP)"

    return (
        f"❓ <b>NEW TRIVIA QUESTION!</b> ❓\n\n"
        f"👤 <b>Asked by:</b> {safe_asker}\n"
        f"🏆 <b>Prize:</b> <code>{prize:,} CXP</code>\n"
        f"🎟️ <b>Cost to Guess:</b> <code>{cost_str}</code>\n\n"
        f"💬 <b>Question:</b>\n{safe_q}\n\n"
        f"👉 Guess with <code>/answer &lt;your guess&gt;</code>!\n"
        f"<i>(Case-insensitive • letters a-z, numbers 0-9 &amp; spaces only)</i>"
    )


async def _post_question_to_group(bot, question: dict) -> int | None:
    """Post a newly activated question to the main group (in cxp_topic_id if configured)."""
    config = await db.get_config()
    if not config or not config.get("main_group_id"):
        return None

    main_group_id = config["main_group_id"]
    cxp_topic_id = config.get("cxp_topic_id")
    asker_name = await _resolve_asker_display_name(
        question["asker_id"], question.get("asker_name"), bot
    )
    text = _build_group_question_html(question, asker_name)

    sent_msg = None
    if cxp_topic_id:
        try:
            sent_msg = await bot.send_message(
                chat_id=main_group_id,
                message_thread_id=cxp_topic_id,
                text=text,
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            logger.warning(
                "Failed sending question to cxp_topic_id %s, retrying without thread: %s",
                cxp_topic_id,
                e,
            )

    if sent_msg is None:
        sent_msg = await bot.send_message(
            chat_id=main_group_id,
            text=text,
            parse_mode=ParseMode.HTML,
        )

    await db.set_question_message_id(question["id"], sent_msg.message_id)
    return sent_msg.message_id


async def _activate_and_announce_next_question(bot, next_question: dict | None):
    """If a queued question was promoted to active, post it to the group and notify its author."""
    if not next_question:
        return

    try:
        await _post_question_to_group(bot, next_question)
    except Exception as e:
        logger.error("Failed to post next queued question %s: %s", next_question.get("id"), e)
        return

    try:
        await bot.send_message(
            chat_id=next_question["asker_id"],
            text=(
                "🚀 <b>Your queued trivia question is now LIVE in the group chat!</b>\n\n"
                f"❓ {html.escape(next_question['question_text'])}"
            ),
            parse_mode=ParseMode.HTML,
        )
    except Exception:
        pass


async def question_group_cmd(update: Update, context: CallbackContext):
    """If a user types /question in the group chat, direct them to DM the bot privately."""
    if not update.effective_chat or not update.message:
        return
    if update.effective_chat.type == "private":
        return

    config = await db.get_config()
    main_group_id = config.get("main_group_id") if config else None
    if main_group_id and update.effective_chat.id != main_group_id:
        return

    bot_username = context.bot.username
    keyboard = (
        InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "💬 DM Bot to Ask a Question",
                        url=f"https://t.me/{bot_username}?start=question",
                    )
                ]
            ]
        )
        if bot_username
        else None
    )

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        message_thread_id=update.message.message_thread_id,
        text=(
            "🤫 To keep your answer secret, send <code>/question</code> to me in a "
            "<b>Direct Message</b>!"
        ),
        parse_mode=ParseMode.HTML,
        reply_markup=keyboard,
    )


async def question_start_dm(update: Update, context: CallbackContext) -> int:
    """Entry point in Private DM for /question (or /start question)."""
    if not update.effective_chat or update.effective_chat.type != "private":
        return ConversationHandler.END
    if not update.effective_user or not update.message:
        return ConversationHandler.END

    _clear_question_flow(context)

    config = await db.get_config()
    main_group_id = config.get("main_group_id") if config else None
    if not main_group_id:
        await update.message.reply_text(
            "❌ The bot's main group is not configured yet."
        )
        return ConversationHandler.END

    user = update.effective_user
    user_id = user.id
    owner_id = _get_bot_owner_id()

    # Verify the user is a member of the main group
    if user_id != owner_id:
        try:
            member = await context.bot.get_chat_member(main_group_id, user_id)
            if member.status in (ChatMember.LEFT, ChatMember.BANNED):
                await update.message.reply_text(
                    "❌ You must be a member of the group chat to create trivia questions."
                )
                return ConversationHandler.END
        except Exception as e:
            logger.warning("Membership check failed for user %s: %s", user_id, e)
            await update.message.reply_text(
                "❌ Could not verify your membership in the group chat."
            )
            return ConversationHandler.END

    full_name = user.first_name + (f" {user.last_name}" if user.last_name else "")
    await db.update_user_display_name(user_id, full_name, user.username)

    user_data = await db.get_user(user_id)
    current_cxp = (user_data.get("cxp") or 0) if user_data else 0
    if current_cxp < 1:
        if current_cxp < 0:
            err_msg = (
                f"❌ You cannot create a trivia question right now because your balance is negative "
                f"(<code>{current_cxp:,} CXP</code>). You need at least <code>1 CXP</code> available to offer as a prize!"
            )
        else:
            err_msg = (
                "❌ You cannot create a trivia question right now because you have <code>0 CXP</code>. "
                "You need at least <code>1 CXP</code> available to offer as a prize!"
            )
        await update.message.reply_text(err_msg, parse_mode=ParseMode.HTML)
        return ConversationHandler.END

    active_q = await db.get_active_question()
    queued_count = await db.get_queued_questions_count()
    queue_note = ""
    if active_q:
        queue_note = (
            f"\n\n<i>ℹ️ Note: A question is currently live in the group"
            f"{f' ({queued_count} queued)' if queued_count else ''}. "
            f"Yours will automatically enter the queue once confirmed!</i>"
        )

    context.user_data["in_question_flow"] = True
    context.user_data["question_draft"] = {
        "asker_id": user_id,
        "asker_name": full_name,
    }

    await update.message.reply_text(
        f"❓ <b>Create a Group Trivia Question (Step 1/4)</b>\n\n"
        f"💰 <b>Your Balance:</b> <code>{current_cxp:,} CXP</code>{queue_note}\n\n"
        f"Please send the <b>question</b> you want to ask the group chat:\n"
        f"<i>(Type /cancel at any time to abort)</i>",
        parse_mode=ParseMode.HTML,
    )
    return Q_STATE_QUESTION


async def receive_question_text(update: Update, context: CallbackContext) -> int:
    text = (update.message.text or "").strip()
    if not text:
        await update.message.reply_text(
            "⚠️ Question cannot be empty. Please send your question text (or /cancel to abort):"
        )
        return Q_STATE_QUESTION

    if len(text) > 1000:
        await update.message.reply_text(
            f"⚠️ That question is too long ({len(text)} chars). Please keep it under 1,000 characters:"
        )
        return Q_STATE_QUESTION

    context.user_data["question_draft"]["question_text"] = text

    await update.message.reply_text(
        "✅ <b>Question saved! (Step 2/4)</b>\n\n"
        "Now send the <b>correct answer</b> for your question.\n"
        "• Matching is <b>case-insensitive</b>\n"
        "• Only letters (<code>a-z</code>), numbers (<code>0-9</code>), and spaces are allowed\n\n"
        "<i>(Type /cancel to abort)</i>",
        parse_mode=ParseMode.HTML,
    )
    return Q_STATE_ANSWER


async def receive_answer_text(update: Update, context: CallbackContext) -> int:
    raw = update.message.text or ""
    normalized = _normalize_answer(raw)
    if not normalized:
        await update.message.reply_text(
            "⚠️ <b>Invalid answer!</b>\n"
            "The answer must be 1–200 characters and may <b>only</b> contain letters (<code>a-z</code>), "
            "numbers (<code>0-9</code>), and spaces (no punctuation or symbols).\n\n"
            "Please enter a valid answer (or /cancel to abort):",
            parse_mode=ParseMode.HTML,
        )
        return Q_STATE_ANSWER

    context.user_data["question_draft"]["answer_text"] = normalized

    user_data = await db.get_user(update.effective_user.id)
    current_cxp = (user_data.get("cxp") or 0) if user_data else 0
    if current_cxp < 1:
        _clear_question_flow(context)
        await update.message.reply_text(
            f"❌ Question creation cancelled — your balance is now <code>{current_cxp:,} CXP</code>, "
            f"so you no longer have at least <code>1 CXP</code> available to offer as a prize.",
            parse_mode=ParseMode.HTML,
        )
        return ConversationHandler.END

    await update.message.reply_text(
        f"✅ <b>Answer saved:</b> <code>{html.escape(normalized)}</code> <b>(Step 3/4)</b>\n\n"
        f"How much <b>CXP</b> are you offering as the <b>prize</b>?\n"
        f"💰 <b>Available Balance:</b> <code>{current_cxp:,} CXP</code>\n\n"
        f"<i>(Enter a whole number ≥ 1, or /cancel to abort)</i>",
        parse_mode=ParseMode.HTML,
    )
    return Q_STATE_PRIZE


async def receive_prize_amount(update: Update, context: CallbackContext) -> int:
    user_data = await db.get_user(update.effective_user.id)
    current_cxp = (user_data.get("cxp") or 0) if user_data else 0
    if current_cxp < 1:
        _clear_question_flow(context)
        await update.message.reply_text(
            f"❌ Question creation cancelled — your balance dropped to <code>{current_cxp:,} CXP</code>, "
            f"so you no longer have at least <code>1 CXP</code> available to offer as a prize.",
            parse_mode=ParseMode.HTML,
        )
        return ConversationHandler.END

    prize = _parse_signed_int_amount(update.message.text or "")
    if prize is None:
        await update.message.reply_text(
            f"❌ Invalid prize amount! Please enter a whole number of at least <code>1 CXP</code> "
            f"(up to your available balance of <code>{current_cxp:,} CXP</code>, or /cancel to abort):",
            parse_mode=ParseMode.HTML,
        )
        return Q_STATE_PRIZE

    if prize < 1:
        await update.message.reply_text(
            f"❌ You cannot offer <code>{prize:,} CXP</code> as a prize — the prize must be at least <code>1 CXP</code>!\n\n"
            f"Please enter an amount between <code>1</code> and <code>{current_cxp:,}</code> (or /cancel to abort):",
            parse_mode=ParseMode.HTML,
        )
        return Q_STATE_PRIZE

    if prize > current_cxp:
        await update.message.reply_text(
            f"❌ You cannot offer a prize of <code>{prize:,} CXP</code> because you only have "
            f"<code>{current_cxp:,} CXP</code> available.\n\n"
            f"Please enter an amount up to <code>{current_cxp:,}</code> (or /cancel to abort):",
            parse_mode=ParseMode.HTML,
        )
        return Q_STATE_PRIZE

    context.user_data["question_draft"]["prize_cxp"] = prize

    await update.message.reply_text(
        f"✅ <b>Prize set to <code>{prize:,} CXP</code>! (Step 4/4)</b>\n\n"
        f"How much <b>CXP</b> should players pay <b>per guess</b> to try answering?\n"
        f"<i>(All CXP paid for guesses goes to you! Enter a whole number ≥ 0, e.g. <code>25</code>, or /cancel to abort)</i>",
        parse_mode=ParseMode.HTML,
    )
    return Q_STATE_COST


async def receive_guess_cost(update: Update, context: CallbackContext) -> int:
    cost = _parse_signed_int_amount(update.message.text or "")
    if cost is None:
        await update.message.reply_text(
            "❌ Invalid guess price! Please enter a whole number of CXP (<code>0</code> or higher, e.g. <code>25</code>, or /cancel to abort):",
            parse_mode=ParseMode.HTML,
        )
        return Q_STATE_COST

    if cost < 0:
        await update.message.reply_text(
            f"❌ You cannot set a negative guess price (<code>{cost:,} CXP</code>)! "
            f"Please enter <code>0</code> (for free guesses) or a positive whole number (or /cancel to abort):",
            parse_mode=ParseMode.HTML,
        )
        return Q_STATE_COST

    draft = context.user_data["question_draft"]
    draft["guess_cost"] = cost

    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✅ Confirm & Submit", callback_data="qconfirm_yes"
                ),
                InlineKeyboardButton("❌ Cancel", callback_data="qconfirm_no"),
            ]
        ]
    )

    cost_display = f"{cost:,} CXP" if cost > 0 else "FREE (0 CXP)"
    summary = (
        f"📋 <b>Confirm Your Trivia Question</b>\n\n"
        f"❓ <b>Question:</b>\n{html.escape(draft['question_text'])}\n\n"
        f"🔑 <b>Correct Answer:</b> <code>{html.escape(draft['answer_text'])}</code> <i>(case-insensitive)</i>\n"
        f"🏆 <b>Prize Offered:</b> <code>{draft['prize_cxp']:,} CXP</code> <i>(escrowed from your CXP now)</i>\n"
        f"🎟️ <b>Price to Guess:</b> <code>{cost_display}</code> <i>(paid to you!)</i>\n\n"
        f"Ready to submit this question?"
    )

    await update.message.reply_text(
        summary,
        parse_mode=ParseMode.HTML,
        reply_markup=keyboard,
    )
    return Q_STATE_CONFIRM


async def _finalize_question_submission(
    update: Update, context: CallbackContext, reply_func
) -> int:
    draft = (context.user_data or {}).get("question_draft")
    _clear_question_flow(context)

    if not draft:
        await reply_func("❌ Question draft expired. Please start again with /question.")
        return ConversationHandler.END

    res = await db.create_question(
        asker_id=draft["asker_id"],
        asker_name=draft["asker_name"],
        question_text=draft["question_text"],
        answer_text=draft["answer_text"],
        prize_cxp=draft["prize_cxp"],
        guess_cost=draft["guess_cost"],
    )

    if not res.get("ok"):
        if res.get("reason") == "insufficient_cxp":
            cur_cxp = res.get("current_cxp", 0)
            prize = draft["prize_cxp"]
            if cur_cxp <= 0:
                err_msg = (
                    f"❌ Cannot submit your question: you offered a prize of <code>{prize:,} CXP</code>, "
                    f"but your current balance is <code>{cur_cxp:,} CXP</code>, so you have no CXP available to offer."
                )
            else:
                err_msg = (
                    f"❌ Cannot submit your question: you offered a prize of <code>{prize:,} CXP</code>, "
                    f"which is more than your available balance of <code>{cur_cxp:,} CXP</code>."
                )
            await reply_func(err_msg, parse_mode=ParseMode.HTML)
        else:
            await reply_func("❌ Database error while submitting your question.")
        return ConversationHandler.END

    question = res["question"]
    remaining_cxp = res["remaining_cxp"]

    if res["became_active"]:
        try:
            await _post_question_to_group(context.bot, question)
        except Exception as e:
            logger.error("Failed to post newly created question to group: %s", e)
            # Refund immediately so user never loses escrowed CXP on send failure
            await db.cancel_active_question(actor_id=draft["asker_id"], is_admin=True)
            await reply_func(
                "❌ Failed to post question to the group chat. Your prize CXP has been refunded."
            )
            return ConversationHandler.END

        await reply_func(
            f"🚀 <b>Confirmed &amp; Posted!</b>\n\n"
            f"Your question is now <b>LIVE</b> in the group chat!\n"
            f"🏆 <b>Escrowed Prize:</b> <code>{draft['prize_cxp']:,} CXP</code>\n"
            f"💰 <b>Remaining Balance:</b> <code>{remaining_cxp:,} CXP</code>",
            parse_mode=ParseMode.HTML,
        )
    else:
        pos = res["queue_position"]
        await reply_func(
            f"⏳ <b>Confirmed &amp; Queued!</b>\n\n"
            f"Another question is currently active, so your question is <b>#{pos} in the queue</b> "
            f"and will be posted automatically once the preceding question closes.\n"
            f"🏆 <b>Escrowed Prize:</b> <code>{draft['prize_cxp']:,} CXP</code>\n"
            f"💰 <b>Remaining Balance:</b> <code>{remaining_cxp:,} CXP</code>",
            parse_mode=ParseMode.HTML,
        )

    return ConversationHandler.END


async def confirm_question_callback(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    await query.answer()

    if query.data == "qconfirm_no":
        _clear_question_flow(context)
        await query.edit_message_text(
            "❌ Question creation cancelled. No CXP was deducted."
        )
        return ConversationHandler.END

    return await _finalize_question_submission(
        update, context, query.edit_message_text
    )


async def confirm_question_text(update: Update, context: CallbackContext) -> int:
    text = (update.message.text or "").strip().lower()
    if text in ("yes", "y", "confirm", "post", "submit"):
        return await _finalize_question_submission(
            update, context, update.message.reply_text
        )
    if text in ("no", "n", "cancel", "abort"):
        _clear_question_flow(context)
        await update.message.reply_text(
            "❌ Question creation cancelled. No CXP was deducted."
        )
        return ConversationHandler.END

    await update.message.reply_text(
        "Please tap <b>✅ Confirm &amp; Submit</b> or <b>❌ Cancel</b> above (or type <code>yes</code> / <code>/cancel</code>).",
        parse_mode=ParseMode.HTML,
    )
    return Q_STATE_CONFIRM


async def cancel_question_draft(update: Update, context: CallbackContext) -> int:
    """Fallback /cancel while inside the DM /question wizard."""
    _clear_question_flow(context)
    if update.message:
        await update.message.reply_text(
            "❌ Question creation cancelled. No CXP was deducted."
        )
    return ConversationHandler.END


async def non_text_in_question_flow(update: Update, context: CallbackContext):
    """Handle unexpected media/stickers during the /question text steps without dropping state."""
    if update.message:
        await update.message.reply_text(
            "⚠️ Please send plain text for this step (or type /cancel to abort)."
        )
    return None


async def unknown_command_in_question_flow(update: Update, context: CallbackContext):
    """Remind user they are in the /question wizard if they send another slash command."""
    if update.message:
        await update.message.reply_text(
            "ℹ️ You are currently creating a trivia question. "
            "Please reply with plain text for the current step, or type /cancel to abort."
        )
    return None


async def handle_stale_question_callback(update: Update, context: CallbackContext):
    """Gracefully answer expired confirmation buttons (e.g. after a bot restart)."""
    query = update.callback_query
    if query:
        await query.answer()
        _clear_question_flow(context)
        try:
            await query.edit_message_text(
                "⚠️ This question draft expired. Please send /question to start a new one."
            )
        except Exception:
            pass


def get_question_conversation_handler() -> ConversationHandler:
    warnings.filterwarnings("ignore", category=PTBUserWarning)
    return ConversationHandler(
        entry_points=[
            CommandHandler(
                "question", question_start_dm, filters=filters.ChatType.PRIVATE
            ),
            CommandHandler(
                "start",
                question_start_dm,
                filters=filters.ChatType.PRIVATE
                & filters.Regex(r"^/start(?:@\w+)?\s+question\s*$"),
            ),
        ],
        states={
            Q_STATE_QUESTION: [
                MessageHandler(
                    filters.ChatType.PRIVATE & filters.TEXT & ~filters.COMMAND,
                    receive_question_text,
                )
            ],
            Q_STATE_ANSWER: [
                MessageHandler(
                    filters.ChatType.PRIVATE & filters.TEXT & ~filters.COMMAND,
                    receive_answer_text,
                )
            ],
            Q_STATE_PRIZE: [
                MessageHandler(
                    filters.ChatType.PRIVATE & filters.TEXT & ~filters.COMMAND,
                    receive_prize_amount,
                )
            ],
            Q_STATE_COST: [
                MessageHandler(
                    filters.ChatType.PRIVATE & filters.TEXT & ~filters.COMMAND,
                    receive_guess_cost,
                )
            ],
            Q_STATE_CONFIRM: [
                CallbackQueryHandler(
                    confirm_question_callback, pattern=r"^qconfirm_(yes|no)$"
                ),
                MessageHandler(
                    filters.ChatType.PRIVATE & filters.TEXT & ~filters.COMMAND,
                    confirm_question_text,
                ),
            ],
        },
        fallbacks=[
            CommandHandler(
                "cancel", cancel_question_draft, filters=filters.ChatType.PRIVATE
            ),
            MessageHandler(
                filters.ChatType.PRIVATE & ~filters.TEXT & ~filters.COMMAND,
                non_text_in_question_flow,
            ),
            MessageHandler(
                filters.ChatType.PRIVATE & filters.COMMAND,
                unknown_command_in_question_flow,
            ),
        ],
        allow_reentry=True,
        per_message=False,
    )


async def answer_cmd(update: Update, context: CallbackContext):
    """Group command: /answer <guess> to pay the guess fee and try winning the active question."""
    if not update.effective_chat or not update.message:
        return

    if update.effective_chat.type == "private":
        await update.message.reply_text(
            "💬 Please use `/answer <your guess>` inside the group chat!",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    config = await db.get_config()
    main_group_id = config.get("main_group_id") if config else None
    if not main_group_id or update.effective_chat.id != main_group_id:
        return

    thread_id = update.message.message_thread_id
    actor_id, actor_name, user_obj, is_unlinked_channel = await _resolve_group_actor(
        update
    )
    if not actor_id:
        return

    if is_unlinked_channel:
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
            "⚠️ Please link your channel to your user account first before answering trivia questions.",
            reply_markup=keyboard,
        )
        return

    # If invoked without arguments, display the current live question
    if not context.args:
        active_q = await db.get_active_question()
        if not active_q:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                message_thread_id=thread_id,
                text="ℹ️ There is no active trivia question right now! DM me <code>/question</code> to ask one.",
                parse_mode=ParseMode.HTML,
            )
            return

        queued_count = await db.get_queued_questions_count()
        asker_name = await _resolve_asker_display_name(
            active_q["asker_id"], active_q.get("asker_name"), context.bot
        )
        cost = active_q["guess_cost"]
        cost_str = f"{cost:,} CXP" if cost > 0 else "FREE (0 CXP)"
        queue_line = (
            f"\n📚 <b>Queued Next:</b> <code>{queued_count}</code>"
            if queued_count
            else ""
        )

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            message_thread_id=thread_id,
            text=(
                f"❓ <b>Active Trivia Question</b>\n\n"
                f"👤 <b>Asked by:</b> {html.escape(asker_name)}\n"
                f"🏆 <b>Prize:</b> <code>{active_q['prize_cxp']:,} CXP</code>\n"
                f"🎟️ <b>Cost to Guess:</b> <code>{cost_str}</code>\n"
                f"🔢 <b>Guesses So Far:</b> <code>{active_q.get('guess_count') or 0}</code>{queue_line}\n\n"
                f"💬 <b>Question:</b>\n{html.escape(active_q['question_text'])}\n\n"
                f"👉 Submit a guess with <code>/answer &lt;your guess&gt;</code>!"
            ),
            parse_mode=ParseMode.HTML,
        )
        return

    if actor_id < 0:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            message_thread_id=thread_id,
            text="⚠️ Anonymous group admins don't have a personal CXP account — please switch to your personal account (or a linked channel) to submit an answer!",
        )
        return

    raw_guess = " ".join(context.args)
    normalized_guess = _normalize_answer(raw_guess)
    if not normalized_guess:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            message_thread_id=thread_id,
            text=(
                "⚠️ Answers only use letters (<code>a-z</code>), numbers (<code>0-9</code>), and spaces "
                "(no punctuation or symbols). <i>You were not charged — try again!</i>"
            ),
            parse_mode=ParseMode.HTML,
        )
        return

    res = await db.submit_question_answer(actor_id, normalized_guess)
    if not res.get("ok"):
        reason = res.get("reason")
        if reason == "no_active_question":
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                message_thread_id=thread_id,
                text="ℹ️ There is currently no active trivia question! DM me <code>/question</code> to ask one.",
                parse_mode=ParseMode.HTML,
            )
        elif reason == "self_answer":
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                message_thread_id=thread_id,
                text="🚫 You can't answer your own trivia question! (Use <code>/cancel</code> if you want to cancel it.)",
                parse_mode=ParseMode.HTML,
            )
        elif reason == "insufficient_cxp":
            cur_cxp = res.get("current_cxp", 0)
            cost = res.get("guess_cost", 0)
            safe_guesser_err = html.escape(actor_name)
            if cur_cxp < 0:
                err_msg = (
                    f"❌ You cannot submit a guess, <b>{safe_guesser_err}</b> — guessing costs <code>{cost:,} CXP</code>, "
                    f"and your balance is currently negative (<code>{cur_cxp:,} CXP</code>)."
                )
            elif cur_cxp == 0:
                err_msg = (
                    f"❌ You cannot submit a guess, <b>{safe_guesser_err}</b> — guessing costs <code>{cost:,} CXP</code>, "
                    f"and you currently have <code>0 CXP</code> available."
                )
            else:
                err_msg = (
                    f"❌ You cannot submit a guess, <b>{safe_guesser_err}</b> — guessing costs <code>{cost:,} CXP</code>, "
                    f"and you only have <code>{cur_cxp:,} CXP</code> available."
                )
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                message_thread_id=thread_id,
                text=err_msg,
                parse_mode=ParseMode.HTML,
            )
        else:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                message_thread_id=thread_id,
                text="❌ Could not submit your guess right now. Please try again.",
            )
        return

    question = res["question"]
    asker_id = question["asker_id"]
    asker_name = await _resolve_asker_display_name(
        asker_id, question.get("asker_name"), context.bot
    )
    safe_guesser = html.escape(actor_name)
    safe_asker = html.escape(asker_name)
    cost = res["guess_cost"]

    if not res["correct"]:
        cost_note = (
            f" (<code>-{cost:,} CXP</code> paid to <b>{safe_asker}</b>)"
            if cost > 0
            else ""
        )
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            message_thread_id=thread_id,
            text=f"❌ Incorrect guess, <b>{safe_guesser}</b>!{cost_note}",
            parse_mode=ParseMode.HTML,
        )
        return

    # Correct answer!
    prize = res["prize_cxp"]
    collected = res["collected_cxp"]
    guesses = res["guess_count"]
    cost_paid_note = f" (paid <code>{cost:,} CXP</code> to guess)" if cost > 0 else ""

    win_msg = (
        f"🎉 <b>CORRECT ANSWER!</b> 🎉\n\n"
        f"🏆 <b>{safe_guesser}</b> answered correctly: <code>{html.escape(question['answer_text'])}</code>!\n"
        f"💰 <b>Prize Won:</b> <code>+{prize:,} CXP</code>{cost_paid_note}\n"
        f"🏦 <b>{safe_asker}</b> collected <code>{collected:,} CXP</code> across <code>{guesses}</code> guess(es)!"
    )
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        message_thread_id=thread_id,
        text=win_msg,
        parse_mode=ParseMode.HTML,
    )

    # Check level-ups for both the winner and the question creator
    await _maybe_announce_level_up(
        context,
        actor_id,
        res["guesser_old_cxp"],
        res["guesser_new_cxp"],
        fallback_user_obj=user_obj,
    )
    if collected > 0:
        await _maybe_announce_level_up(
            context,
            asker_id,
            res["asker_old_cxp"],
            res["asker_new_cxp"],
        )

    # Notify question creator in DM
    try:
        await context.bot.send_message(
            chat_id=asker_id,
            text=(
                f"🎉 <b>Your trivia question was answered!</b>\n\n"
                f"🏆 <b>Winner:</b> {safe_guesser}\n"
                f"🔑 <b>Answer:</b> <code>{html.escape(question['answer_text'])}</code>\n"
                f"🏦 <b>Guess Fees Earned:</b> <code>+{collected:,} CXP</code> ({guesses} guess(es))"
            ),
            parse_mode=ParseMode.HTML,
        )
    except Exception:
        pass

    # Advance queue if another question is waiting
    await _activate_and_announce_next_question(context.bot, res.get("next_question"))


async def cancel_question_cmd(update: Update, context: CallbackContext):
    """
    /cancel command for admins and the active question's author.
    Cancels the current active question, returns the offered prize to the asker,
    pays out any collected guess fees to the asker, and activates the next queued question.
    """
    if not update.effective_chat or not update.message:
        return

    config = await db.get_config()
    main_group_id = config.get("main_group_id") if config else None
    if not main_group_id:
        return

    chat_type = update.effective_chat.type
    if chat_type in ("group", "supergroup") and update.effective_chat.id != main_group_id:
        return

    if chat_type == "private":
        if not update.effective_user:
            return
        actor_id = update.effective_user.id
        actor_name = update.effective_user.first_name + (
            f" {update.effective_user.last_name}"
            if update.effective_user.last_name
            else ""
        )
    else:
        actor_id, actor_name, _, _ = await _resolve_group_actor(update)
        if not actor_id:
            return

    owner_id = _get_bot_owner_id()
    is_admin = (
        (owner_id is not None and actor_id == owner_id)
        or actor_id == main_group_id  # Anonymous admin posting as the group itself
        or await is_user_admin(actor_id, main_group_id, context)
    )

    res = await db.cancel_active_question(actor_id=actor_id, is_admin=is_admin)
    if not res.get("ok"):
        reason = res.get("reason")
        reply_thread = (
            update.message.message_thread_id if chat_type != "private" else None
        )
        if reason == "no_active_question":
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                message_thread_id=reply_thread,
                text="ℹ️ There is currently no active trivia question to cancel.",
            )
        elif reason == "forbidden":
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                message_thread_id=reply_thread,
                text="❌ Only group admins or the person who asked the current question can use /cancel.",
            )
        else:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                message_thread_id=reply_thread,
                text="❌ Failed to cancel the active question.",
            )
        return

    cancelled_q = res["cancelled_question"]
    asker_id = cancelled_q["asker_id"]
    asker_name = await _resolve_asker_display_name(
        asker_id, cancelled_q.get("asker_name"), context.bot
    )
    safe_asker = html.escape(asker_name)
    safe_actor = html.escape(actor_name)
    prize_returned = res["prize_returned"]
    collected_paid = res["collected_paid"]
    guess_count = res["guess_count"]

    cancel_msg = (
        f"🛑 <b>Trivia Question Cancelled</b>\n\n"
        f"The active question by <b>{safe_asker}</b> was cancelled by <b>{safe_actor}</b>.\n"
        f"↩️ <b>Prize Returned:</b> <code>+{prize_returned:,} CXP</code> refunded to <b>{safe_asker}</b>.\n"
        f"🏦 <b>Paid Guesses:</b> <code>+{collected_paid:,} CXP</code> ({guess_count} guess(es)) paid to <b>{safe_asker}</b>."
    )

    if chat_type == "private":
        # Confirm in DM and also announce in the main group
        await update.message.reply_text(
            f"✅ Cancelled the active trivia question! <code>{prize_returned:,} CXP</code> prize refunded "
            f"(plus <code>{collected_paid:,} CXP</code> from guesses).",
            parse_mode=ParseMode.HTML,
        )
        cxp_topic_id = config.get("cxp_topic_id")
        try:
            await context.bot.send_message(
                chat_id=main_group_id,
                message_thread_id=cxp_topic_id if cxp_topic_id else None,
                text=cancel_msg,
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            await context.bot.send_message(
                chat_id=main_group_id,
                text=cancel_msg,
                parse_mode=ParseMode.HTML,
            )
    else:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            message_thread_id=update.message.message_thread_id,
            text=cancel_msg,
            parse_mode=ParseMode.HTML,
        )

    if collected_paid > 0:
        await _maybe_announce_level_up(
            context,
            asker_id,
            res["asker_old_cxp"],
            res["asker_new_cxp"],
        )

    # Move onto the next queued question if one exists
    await _activate_and_announce_next_question(context.bot, res.get("next_question"))
