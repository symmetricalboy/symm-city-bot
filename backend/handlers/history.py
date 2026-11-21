import asyncio
import calendar
import html
import io
import json
import logging
import re
from datetime import date, datetime, time as dt_time, timedelta, timezone

import numpy as np
import matplotlib.dates as mdates
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, MessageLimit, ParseMode
from telegram.ext import CallbackContext

from database import db
from handlers.common import resolve_username
from handlers.cxp import _get_member_tag_string, calculate_level

logger = logging.getLogger(__name__)

MONTH_LOOKUP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

TWO_WORD_TIMESPANS = {
    ("this", "day"): "today",
    ("this", "week"): "week",
    ("last", "week"): "last_week",
    ("past", "week"): "7d",
    ("this", "month"): "month",
    ("last", "month"): "last_month",
    ("past", "month"): "30d",
    ("this", "year"): "year",
    ("last", "year"): "last_year",
    ("past", "year"): "365d",
    ("all", "time"): "all",
}

ONE_WORD_TIMESPANS = {
    "today": "today",
    "day": "today",
    "daily": "today",
    "yesterday": "yesterday",
    "week": "week",
    "weekly": "week",
    "month": "month",
    "monthly": "month",
    "year": "year",
    "yearly": "year",
    "contest": "contest",
    "all": "all",
    "alltime": "all",
    "all-time": "all",
    "lifetime": "all",
    "ever": "all",
    "global": "all",
}

DURATION_TOKEN_RE = re.compile(
    r"^(\d{1,4})(h|hr|hrs|hour|hours|d|day|days|w|wk|wks|week|weeks)$",
    re.IGNORECASE,
)


def _ensure_utc(dt_val: datetime | None) -> datetime | None:
    if dt_val is None:
        return None
    if dt_val.tzinfo is None:
        return dt_val.replace(tzinfo=timezone.utc)
    return dt_val.astimezone(timezone.utc)


def _fmt_signed(val: int) -> str:
    if val > 0:
        return f"+{val:,}"
    return f"{val:,}"


def _fmt_compact_signed(val: int, show_plus: bool = True) -> str:
    sign = "+" if (val > 0 and show_plus) else ("-" if val < 0 else "")
    abs_v = abs(val)
    if abs_v >= 1_000_000:
        return f"{sign}{abs_v / 1_000_000:.1f}M"
    if abs_v >= 10_000:
        return f"{sign}{abs_v / 1_000:.1f}k"
    return f"{sign}{abs_v:,}"


def _parse_explicit_date(token: str) -> date | None:
    if token.startswith("@"):
        return None
    for fmt in ("%m/%d/%y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(token, fmt).date()
        except ValueError:
            continue
    return None


def _build_window_from_spec(
    spec_key: str,
    now_utc: datetime,
    config: dict | None,
    custom_dates: tuple[date, date] | None = None,
    month_year: tuple[int, int] | None = None,
) -> dict:
    """
    Convert a parsed timespan specification into:
      {
        "mode": str,
        "start_dt": datetime | None,   # None means All-Time
        "end_dt": datetime,
        "label": str,
        "error": str | None,
      }
    """
    today_utc = now_utc.date()

    if spec_key == "custom_dates" and custom_dates:
        d_start, d_end = custom_dates
        start_dt = datetime.combine(d_start, dt_time.min, tzinfo=timezone.utc)
        raw_end = datetime.combine(d_end, dt_time.max, tzinfo=timezone.utc)
        end_dt = min(raw_end, now_utc) if start_dt <= now_utc else raw_end
        if d_start == d_end:
            label = d_start.strftime("%b %d, %Y")
        else:
            label = f"{d_start.strftime('%b %d, %Y')} – {d_end.strftime('%b %d, %Y')}"
        return {
            "mode": "range",
            "start_dt": start_dt,
            "end_dt": end_dt,
            "label": label,
            "error": None,
        }

    if spec_key == "month_year" and month_year:
        target_month, target_year = month_year
        last_day = calendar.monthrange(target_year, target_month)[1]
        d_start = date(target_year, target_month, 1)
        d_end = date(target_year, target_month, last_day)
        start_dt = datetime.combine(d_start, dt_time.min, tzinfo=timezone.utc)
        raw_end = datetime.combine(d_end, dt_time.max, tzinfo=timezone.utc)
        end_dt = min(raw_end, now_utc) if start_dt <= now_utc else raw_end
        month_name = calendar.month_name[target_month]
        return {
            "mode": "range",
            "start_dt": start_dt,
            "end_dt": end_dt,
            "label": f"{month_name} {target_year}",
            "error": None,
        }

    if spec_key == "all":
        return {
            "mode": "all",
            "start_dt": None,
            "end_dt": now_utc,
            "label": "All-Time",
            "error": None,
        }

    if spec_key == "contest":
        if not config or not config.get("contest_start") or not config.get("contest_end"):
            return {
                "mode": "contest",
                "start_dt": None,
                "end_dt": now_utc,
                "label": "Contest",
                "error": "ℹ️ No contest timeframe is currently configured.",
            }
        c_start: date = config["contest_start"]
        c_end: date = config["contest_end"]
        start_dt = datetime.combine(c_start, dt_time.min, tzinfo=timezone.utc)
        raw_end = datetime.combine(c_end, dt_time.max, tzinfo=timezone.utc)
        end_dt = min(raw_end, now_utc) if start_dt <= now_utc else raw_end
        label = f"Contest ({c_start.strftime('%b %d')} – {c_end.strftime('%b %d, %Y')})"
        return {
            "mode": "range",
            "start_dt": start_dt,
            "end_dt": end_dt,
            "label": label,
            "error": None,
        }

    if spec_key == "today":
        start_dt = datetime.combine(today_utc, dt_time.min, tzinfo=timezone.utc)
        return {
            "mode": "range",
            "start_dt": start_dt,
            "end_dt": now_utc,
            "label": f"Today ({today_utc.strftime('%b %d, %Y')} UTC)",
            "error": None,
        }

    if spec_key == "yesterday":
        yest = today_utc - timedelta(days=1)
        start_dt = datetime.combine(yest, dt_time.min, tzinfo=timezone.utc)
        end_dt = datetime.combine(yest, dt_time.max, tzinfo=timezone.utc)
        return {
            "mode": "range",
            "start_dt": start_dt,
            "end_dt": end_dt,
            "label": f"Yesterday ({yest.strftime('%b %d, %Y')} UTC)",
            "error": None,
        }

    if spec_key == "week":
        start_of_week = today_utc - timedelta(days=today_utc.weekday())
        start_dt = datetime.combine(start_of_week, dt_time.min, tzinfo=timezone.utc)
        return {
            "mode": "range",
            "start_dt": start_dt,
            "end_dt": now_utc,
            "label": f"This Week ({start_of_week.strftime('%b %d')} – {today_utc.strftime('%b %d')})",
            "error": None,
        }

    if spec_key == "last_week":
        start_of_last_week = today_utc - timedelta(days=today_utc.weekday() + 7)
        end_of_last_week = start_of_last_week + timedelta(days=6)
        start_dt = datetime.combine(start_of_last_week, dt_time.min, tzinfo=timezone.utc)
        end_dt = datetime.combine(end_of_last_week, dt_time.max, tzinfo=timezone.utc)
        return {
            "mode": "range",
            "start_dt": start_dt,
            "end_dt": end_dt,
            "label": f"Last Week ({start_of_last_week.strftime('%b %d')} – {end_of_last_week.strftime('%b %d')})",
            "error": None,
        }

    if spec_key == "month":
        start_of_month = today_utc.replace(day=1)
        start_dt = datetime.combine(start_of_month, dt_time.min, tzinfo=timezone.utc)
        return {
            "mode": "range",
            "start_dt": start_dt,
            "end_dt": now_utc,
            "label": f"This Month ({today_utc.strftime('%B %Y')})",
            "error": None,
        }

    if spec_key == "last_month":
        first_this = today_utc.replace(day=1)
        end_last = first_this - timedelta(days=1)
        start_last = end_last.replace(day=1)
        start_dt = datetime.combine(start_last, dt_time.min, tzinfo=timezone.utc)
        end_dt = datetime.combine(end_last, dt_time.max, tzinfo=timezone.utc)
        return {
            "mode": "range",
            "start_dt": start_dt,
            "end_dt": end_dt,
            "label": f"Last Month ({start_last.strftime('%B %Y')})",
            "error": None,
        }

    if spec_key == "year":
        start_of_year = date(today_utc.year, 1, 1)
        start_dt = datetime.combine(start_of_year, dt_time.min, tzinfo=timezone.utc)
        return {
            "mode": "range",
            "start_dt": start_dt,
            "end_dt": now_utc,
            "label": f"This Year ({today_utc.year})",
            "error": None,
        }

    if spec_key == "last_year":
        prev_year = today_utc.year - 1
        start_dt = datetime(prev_year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end_dt = datetime(prev_year, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc)
        return {
            "mode": "range",
            "start_dt": start_dt,
            "end_dt": end_dt,
            "label": f"Last Year ({prev_year})",
            "error": None,
        }

    # Duration strings like "24h", "7d", "30d", "2w"
    m = DURATION_TOKEN_RE.match(spec_key)
    if m:
        amount = max(1, min(3650, int(m.group(1))))
        unit = m.group(2).lower()
        if unit.startswith("h"):
            start_dt = now_utc - timedelta(hours=amount)
            unit_lbl = "Hour" if amount == 1 else "Hours"
            return {
                "mode": "range",
                "start_dt": start_dt,
                "end_dt": now_utc,
                "label": f"Last {amount} {unit_lbl}",
                "error": None,
            }
        if unit.startswith("w"):
            days = amount * 7
            start_day = today_utc - timedelta(days=days - 1)
            start_dt = datetime.combine(start_day, dt_time.min, tzinfo=timezone.utc)
            unit_lbl = "Week" if amount == 1 else "Weeks"
            return {
                "mode": "range",
                "start_dt": start_dt,
                "end_dt": now_utc,
                "label": f"Last {amount} {unit_lbl} ({days}d)",
                "error": None,
            }
        # Days
        if amount == 1:
            start_dt = now_utc - timedelta(hours=24)
            return {
                "mode": "range",
                "start_dt": start_dt,
                "end_dt": now_utc,
                "label": "Last 24 Hours",
                "error": None,
            }
        start_day = today_utc - timedelta(days=amount - 1)
        start_dt = datetime.combine(start_day, dt_time.min, tzinfo=timezone.utc)
        return {
            "mode": "range",
            "start_dt": start_dt,
            "end_dt": now_utc,
            "label": f"Last {amount} Days",
            "error": None,
        }

    return {
        "mode": "all",
        "start_dt": None,
        "end_dt": now_utc,
        "label": "All-Time",
        "error": None,
    }


def parse_history_args(
    args: list[str] | None,
    config: dict | None,
    now_utc: datetime | None = None,
    protected_user_text: str | None = None,
) -> tuple[dict, str | None]:
    """
    Parse mixed-order /history arguments into (window_spec_dict, user_query_or_None).
    Supports any order of timespan and username/display-name, e.g.:
      /history
      /history today
      /history @symmetricalboy today
      /history month symmetricalboy
      /history symmetricalboy last week
      /history 7d @symmetricalboy
      /history 11/01/26 11/20/26 symmetricalboy
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    if not args:
        return _build_window_from_spec("all", now_utc, config), None

    tokens = [t.strip() for t in args if t and t.strip()]
    if not tokens:
        return _build_window_from_spec("all", now_utc, config), None

    # Mark indices that should never be interpreted as timespans (e.g. @mentions or text_mentions)
    protected_indices: set[int] = set()
    for idx, tok in enumerate(tokens):
        if tok.startswith("@"):
            protected_indices.add(idx)

    if protected_user_text:
        prot_parts = protected_user_text.strip().split()
        if prot_parts:
            plen = len(prot_parts)
            for i in range(len(tokens) - plen + 1):
                if [t.lower() for t in tokens[i : i + plen]] == [
                    p.lower() for p in prot_parts
                ]:
                    for j in range(i, i + plen):
                        protected_indices.add(j)
                    break

    consumed: set[int] = set()
    window_spec: dict | None = None

    # 1. Check explicit dates (e.g. 11/01/26 or 11/01/26 11/20/26)
    date_matches: list[tuple[int, date]] = []
    for idx, tok in enumerate(tokens):
        if idx in protected_indices:
            continue
        parsed_d = _parse_explicit_date(tok)
        if parsed_d is not None:
            date_matches.append((idx, parsed_d))

    if len(date_matches) >= 2:
        (i1, d1), (i2, d2) = date_matches[0], date_matches[1]
        consumed.update({i1, i2})
        d_start, d_end = (d1, d2) if d1 <= d2 else (d2, d1)
        window_spec = _build_window_from_spec(
            "custom_dates", now_utc, config, custom_dates=(d_start, d_end)
        )
    elif len(date_matches) == 1:
        i1, d1 = date_matches[0]
        consumed.add(i1)
        window_spec = _build_window_from_spec(
            "custom_dates", now_utc, config, custom_dates=(d1, d1)
        )

    # 2. Check 2-word timespans (e.g. "last week", "this month", "all time", "october 2026")
    if window_spec is None:
        for i in range(len(tokens) - 1):
            if i in protected_indices or (i + 1) in protected_indices:
                continue
            w1, w2 = tokens[i].lower(), tokens[i + 1].lower()
            pair = (w1, w2)
            if pair in TWO_WORD_TIMESPANS:
                consumed.update({i, i + 1})
                window_spec = _build_window_from_spec(
                    TWO_WORD_TIMESPANS[pair], now_utc, config
                )
                break
            if w1 in MONTH_LOOKUP and re.fullmatch(r"20\d{2}", w2):
                consumed.update({i, i + 1})
                window_spec = _build_window_from_spec(
                    "month_year",
                    now_utc,
                    config,
                    month_year=(MONTH_LOOKUP[w1], int(w2)),
                )
                break

    # 3. Check 1-word timespans (e.g. "today", "month", "7d", "24h", "november", "contest")
    if window_spec is None:
        for i, tok in enumerate(tokens):
            if i in protected_indices:
                continue
            low = tok.lower()
            if low in ONE_WORD_TIMESPANS:
                consumed.add(i)
                window_spec = _build_window_from_spec(
                    ONE_WORD_TIMESPANS[low], now_utc, config
                )
                break
            if low in MONTH_LOOKUP:
                consumed.add(i)
                target_month = MONTH_LOOKUP[low]
                target_year = now_utc.year
                if target_month > now_utc.month:
                    target_year -= 1
                window_spec = _build_window_from_spec(
                    "month_year",
                    now_utc,
                    config,
                    month_year=(target_month, target_year),
                )
                break
            if DURATION_TOKEN_RE.match(low):
                consumed.add(i)
                window_spec = _build_window_from_spec(low, now_utc, config)
                break

    if window_spec is None:
        window_spec = _build_window_from_spec("all", now_utc, config)

    remaining = [tok for idx, tok in enumerate(tokens) if idx not in consumed]
    user_query = " ".join(remaining).strip() if remaining else None
    return window_spec, user_query


def _decode_metadata(raw_meta) -> dict:
    if not raw_meta:
        return {}
    if isinstance(raw_meta, dict):
        return raw_meta
    if isinstance(raw_meta, str):
        try:
            parsed = json.loads(raw_meta)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def build_cxp_history_report(
    user_id: int,
    raw_data: dict,
    window_spec: dict,
    now_utc: datetime | None = None,
) -> dict:
    """
    Reconstruct a continuous CXP balance timeline from:
      - granular `cxp_history` ledger rows (exact timestamps, categories, balances)
      - historical `questions` records (for pre-ledger trivia wins/escrows/collections)
      - historical `user_daily_cxp` daily net records (for pre-ledger daily curves)
      - current `users.cxp` balance
    Then slice to `window_spec` and compute category breakdowns, high/low points, and plot series.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    user_row = raw_data.get("user") or {}
    current_cxp = int(user_row.get("cxp") or 0)
    daily_rows = raw_data.get("daily") or []
    ledger_rows = raw_data.get("ledger") or []
    question_rows = raw_data.get("questions") or []

    # Normalize ledger rows
    normalized_ledger: list[dict] = []
    recorded_trivia_refs: set[tuple[str | None, int]] = set()
    ledger_day_sums: dict[date, int] = {}

    for r in ledger_rows:
        ts = _ensure_utc(r["created_at"]) or now_utc
        delta = int(r["delta_cxp"] or 0)
        bal_after = int(r["balance_after"] or 0)
        cat = r["category"] or "other"
        sub = r.get("sub_type")
        ref_id = r.get("reference_id")
        meta = _decode_metadata(r.get("metadata"))

        if cat == "trivia" and ref_id is not None:
            recorded_trivia_refs.add((sub, int(ref_id)))

        d_key = ts.date()
        ledger_day_sums[d_key] = ledger_day_sums.get(d_key, 0) + delta

        normalized_ledger.append(
            {
                "ts": ts,
                "delta": delta,
                "balance_before": bal_after - delta,
                "balance_after": bal_after,
                "category": cat,
                "sub_type": sub,
                "metadata": meta,
                "is_synthetic": False,
            }
        )

    first_ledger_ts = normalized_ledger[0]["ts"] if normalized_ledger else None
    first_ledger_date = first_ledger_ts.date() if first_ledger_ts else None
    anchor_balance = (
        normalized_ledger[0]["balance_before"] if normalized_ledger else current_cxp
    )

    # Extract pre-ledger trivia events from `questions` that aren't already in `cxp_history`
    pre_trivia_events: list[dict] = []
    pre_trivia_day_sums: dict[date, int] = {}

    def _add_pre_trivia(ts_raw: datetime | None, delta: int, sub_type: str, meta: dict):
        if delta == 0 and sub_type != "win":
            return
        ts = _ensure_utc(ts_raw)
        if ts is None:
            return
        if first_ledger_ts is not None and ts > first_ledger_ts:
            return
        if first_ledger_ts is not None and ts >= first_ledger_ts:
            ts = first_ledger_ts - timedelta(milliseconds=1)
        d_key = ts.date()
        pre_trivia_day_sums[d_key] = pre_trivia_day_sums.get(d_key, 0) + delta
        pre_trivia_events.append(
            {
                "ts": ts,
                "delta": delta,
                "balance_before": 0,
                "balance_after": 0,
                "category": "trivia",
                "sub_type": sub_type,
                "metadata": meta,
                "is_synthetic": False,
            }
        )

    for q in question_rows:
        qid = int(q["id"])
        prize = int(q["prize_cxp"] or 0)
        cost = int(q["guess_cost"] or 0)
        collected = int(q["collected_cxp"] or 0)
        guesses = int(q["guess_count"] or 0)
        status = q.get("status")

        if q.get("asker_id") == user_id and ("escrow_prize", qid) not in recorded_trivia_refs:
            _add_pre_trivia(
                q.get("created_at"),
                -prize,
                "escrow_prize",
                {"prize_cxp": prize, "guess_cost": cost},
            )

        if q.get("closed_at") is not None:
            if status == "answered":
                if q.get("winner_id") == user_id and ("win", qid) not in recorded_trivia_refs:
                    _add_pre_trivia(
                        q.get("closed_at"),
                        prize - cost,
                        "win",
                        {"prize_cxp": prize, "guess_cost": cost},
                    )
                if (
                    q.get("asker_id") == user_id
                    and collected > 0
                    and ("collect_fees", qid) not in recorded_trivia_refs
                ):
                    _add_pre_trivia(
                        q.get("closed_at"),
                        collected,
                        "collect_fees",
                        {"collected_cxp": collected, "guess_count": guesses},
                    )
            elif status == "cancelled":
                if (
                    q.get("asker_id") == user_id
                    and ("cancel_refund", qid) not in recorded_trivia_refs
                ):
                    _add_pre_trivia(
                        q.get("closed_at"),
                        prize + collected,
                        "cancel_refund",
                        {
                            "prize_returned": prize,
                            "collected_paid": collected,
                            "guess_count": guesses,
                        },
                    )

    # Reconcile pre-ledger daily net CXP from `user_daily_cxp`
    daily_map: dict[date, int] = {}
    for d_row in daily_rows:
        d_val = d_row["date"]
        daily_map[d_val] = int(d_row["cxp_gained"] or 0)

    candidate_pre_dates = set(daily_map.keys()) | set(pre_trivia_day_sums.keys())
    legacy_daily_events: list[dict] = []

    for d_val in sorted(candidate_pre_dates):
        if first_ledger_date is not None and d_val > first_ledger_date:
            continue
        daily_total = daily_map.get(d_val, 0)
        ledger_sum = ledger_day_sums.get(d_val, 0)
        pre_trivia_sum = pre_trivia_day_sums.get(d_val, 0)
        residual = daily_total - ledger_sum - pre_trivia_sum
        if residual == 0:
            continue

        day_start = datetime.combine(d_val, dt_time.min, tzinfo=timezone.utc)
        if first_ledger_ts is not None and d_val == first_ledger_date:
            span = max(timedelta(seconds=1), (first_ledger_ts - day_start) / 2)
            syn_ts = min(day_start + span, first_ledger_ts - timedelta(milliseconds=2))
        elif d_val == now_utc.date():
            span = max(timedelta(seconds=1), (now_utc - day_start) / 2)
            syn_ts = min(day_start + span, now_utc)
        else:
            syn_ts = datetime.combine(
                d_val, dt_time(23, 59, 0), tzinfo=timezone.utc
            )

        legacy_daily_events.append(
            {
                "ts": syn_ts,
                "delta": residual,
                "balance_before": 0,
                "balance_after": 0,
                "category": "legacy",
                "sub_type": "daily_net",
                "metadata": {},
                "is_synthetic": True,
            }
        )

    pre_events = sorted(
        pre_trivia_events + legacy_daily_events, key=lambda e: e["ts"]
    )

    # Walk backwards from `anchor_balance` to populate exact balances on pre-ledger events
    running_bal = anchor_balance
    for ev in reversed(pre_events):
        ev["balance_after"] = running_bal
        ev["balance_before"] = running_bal - ev["delta"]
        running_bal = ev["balance_before"]

    initial_baseline_balance = running_bal
    all_events = pre_events + normalized_ledger

    # If external DB edits ever caused final ledger balance to differ from users.cxp, reconcile cleanly
    if all_events and all_events[-1]["balance_after"] != current_cxp:
        drift = current_cxp - all_events[-1]["balance_after"]
        all_events.append(
            {
                "ts": max(now_utc, all_events[-1]["ts"] + timedelta(milliseconds=1)),
                "delta": drift,
                "balance_before": all_events[-1]["balance_after"],
                "balance_after": current_cxp,
                "category": "other",
                "sub_type": "adjustment",
                "metadata": {},
                "is_synthetic": True,
            }
        )

    # Determine effective [window_start, window_end]
    window_end: datetime = window_spec["end_dt"]
    if window_spec["start_dt"] is None:
        if all_events:
            first_ts = all_events[0]["ts"]
            first_day_start = datetime.combine(
                first_ts.date(), dt_time.min, tzinfo=timezone.utc
            )
            window_start = min(first_day_start, first_ts - timedelta(hours=1))
        else:
            window_start = window_end - timedelta(days=1)
    else:
        window_start = window_spec["start_dt"]

    if window_end <= window_start:
        window_end = window_start + timedelta(minutes=1)

    # Compute start_balance at window_start
    start_balance = initial_baseline_balance
    for ev in all_events:
        if ev["ts"] < window_start:
            start_balance = ev["balance_after"]
        else:
            break

    window_events = [
        ev for ev in all_events if window_start <= ev["ts"] <= window_end
    ]
    end_balance = (
        window_events[-1]["balance_after"] if window_events else start_balance
    )
    net_change = end_balance - start_balance

    # Compute exact High & Low across the entire timeframe (including start_balance & end_balance)
    balance_checkpoints: list[tuple[datetime, int]] = [(window_start, start_balance)]
    for ev in window_events:
        if ev["delta"] != 0:
            balance_checkpoints.append((ev["ts"], ev["balance_after"]))
    balance_checkpoints.append((window_end, end_balance))

    high_ts, high_balance = max(balance_checkpoints, key=lambda p: p[1])
    low_ts, low_balance = min(balance_checkpoints, key=lambda p: p[1])

    # Aggregate rich category breakdowns
    categories = {
        "message": {"net": 0, "gained": 0, "lost": 0, "count": 0},
        "reaction": {"net": 0, "gained": 0, "lost": 0, "count": 0},
        "gamble": {
            "net": 0,
            "gained": 0,
            "lost": 0,
            "count": 0,
            "wins": 0,
            "losses": 0,
            "wagered": 0,
            "best_win": 0,
        },
        "steal": {
            "net": 0,
            "gained": 0,
            "lost": 0,
            "count": 0,
            "stole_count": 0,
            "robbed_count": 0,
            "blocked_count": 0,
            "defended_count": 0,
        },
        "trivia": {
            "net": 0,
            "gained": 0,
            "lost": 0,
            "count": 0,
            "wins": 0,
            "questions_asked": 0,
            "wrong_guesses": 0,
        },
        "admin_give": {"net": 0, "gained": 0, "lost": 0, "count": 0},
        "other": {"net": 0, "gained": 0, "lost": 0, "count": 0},
        "legacy": {"net": 0, "gained": 0, "lost": 0, "count": 0},
    }

    for ev in window_events:
        cat_key = ev["category"] if ev["category"] in categories else "other"
        bucket = categories[cat_key]
        d = ev["delta"]
        sub = ev.get("sub_type")
        meta = ev.get("metadata") or {}

        bucket["net"] += d
        if d > 0:
            bucket["gained"] += d
        elif d < 0:
            bucket["lost"] += d
        bucket["count"] += 1

        if cat_key == "gamble":
            if sub == "win" or d > 0:
                bucket["wins"] += 1
                if d > bucket["best_win"]:
                    bucket["best_win"] = d
            elif sub == "loss" or d < 0:
                bucket["losses"] += 1
            bucket["wagered"] += int(meta.get("bet") or abs(d))
        elif cat_key == "steal":
            if sub == "stole" or d > 0:
                bucket["stole_count"] += 1
            elif sub == "robbed" or d < 0:
                bucket["robbed_count"] += 1
            elif sub == "blocked":
                bucket["blocked_count"] += 1
            elif sub == "defended":
                bucket["defended_count"] += 1
        elif cat_key == "trivia":
            if sub == "win":
                bucket["wins"] += 1
            elif sub == "escrow_prize":
                bucket["questions_asked"] += 1
            elif sub == "wrong_guesses" or sub == "wrong_guess":
                bucket["wrong_guesses"] += 1

    # Build trajectory vertices for plotting
    plot_points: list[tuple[datetime, int, bool]] = [
        (window_start, start_balance, False)
    ]
    nonzero_events = [ev for ev in window_events if ev["delta"] != 0]

    # If there are thousands of granular events over long periods, downsample intermediate points
    # while ALWAYS preserving exact global High/Low and per-bucket open/high/low/close!
    if len(nonzero_events) <= 450:
        for ev in nonzero_events:
            if not ev["is_synthetic"]:
                # Step hold right up to the exact event timestamp so idle stretches look flat
                prev_ts = plot_points[-1][0]
                if (ev["ts"] - prev_ts) > timedelta(minutes=2):
                    plot_points.append(
                        (ev["ts"] - timedelta(seconds=1), ev["balance_before"], False)
                    )
            plot_points.append((ev["ts"], ev["balance_after"], not ev["is_synthetic"]))
    else:
        # Bucket into ~200 slices preserving first, min, max, last in each slice
        chunk_size = max(2, len(nonzero_events) // 180)
        for i in range(0, len(nonzero_events), chunk_size):
            chunk = nonzero_events[i : i + chunk_size]
            pick_map = {
                0: chunk[0],
                len(chunk) - 1: chunk[-1],
            }
            max_idx = max(range(len(chunk)), key=lambda k: chunk[k]["balance_after"])
            min_idx = min(range(len(chunk)), key=lambda k: chunk[k]["balance_after"])
            pick_map[max_idx] = chunk[max_idx]
            pick_map[min_idx] = chunk[min_idx]
            for idx in sorted(pick_map.keys()):
                ev = pick_map[idx]
                plot_points.append((ev["ts"], ev["balance_after"], False))

    if plot_points[-1][0] < window_end:
        plot_points.append((window_end, end_balance, False))

    return {
        "current_cxp": current_cxp,
        "start_balance": start_balance,
        "end_balance": end_balance,
        "net_change": net_change,
        "high_balance": high_balance,
        "high_ts": high_ts,
        "low_balance": low_balance,
        "low_ts": low_ts,
        "window_start": window_start,
        "window_end": window_end,
        "window_label": window_spec["label"],
        "event_count": len(nonzero_events),
        "categories": categories,
        "plot_points": plot_points,
    }


def _sanitize_chart_text(text: str) -> str:
    """Keep characters guaranteed to render cleanly in DejaVu Sans (strip color emoji surrogates)."""
    cleaned = []
    for ch in text:
        cp = ord(ch)
        # Skip supplemental plane emojis/pictographs and variation selectors that render as boxes in Agg
        if cp >= 0x1F000 or cp in (0xFE0F, 0xFE0E, 0x200D):
            continue
        cleaned.append(ch)
    res = "".join(cleaned).strip()
    return res if res else "Member"


def render_history_chart_png(target_name: str, report: dict) -> io.BytesIO:
    """
    Render a sleek dark-mode two-panel PNG chart in memory using Matplotlib Figure + FigureCanvasAgg:
      - Top Panel: Continuous CXP Balance Trajectory with above/below period-start shading & High/Low callouts
      - Bottom Panel: Diverging Gains (+) vs Losses (-) horizontal bar chart across all CXP categories
    """
    bg_fig = "#0b0f19"
    bg_card = "#111827"
    grid_col = "#1f2937"
    spine_col = "#334155"
    text_main = "#f8fafc"
    text_muted = "#94a3b8"
    col_gain = "#10b981"
    col_gain_bright = "#34d399"
    col_loss = "#f43f5e"
    col_loss_bright = "#fb7185"
    col_neutral = "#38bdf8"

    fig = Figure(figsize=(10.0, 6.8), dpi=150, facecolor=bg_fig)
    FigureCanvasAgg(fig)

    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[2.05, 1.25],
        hspace=0.34,
        left=0.12,
        right=0.83,
        top=0.84,
        bottom=0.09,
    )

    ax_top = fig.add_subplot(gs[0], facecolor=bg_card)
    ax_bot = fig.add_subplot(gs[1], facecolor=bg_card)

    # ── Header Banner ──
    clean_name = _sanitize_chart_text(target_name)
    if len(clean_name) > 32:
        clean_name = clean_name[:30] + "…"

    fig.text(
        0.12,
        0.938,
        f"symm.city  •  {clean_name}",
        fontsize=15,
        fontweight="bold",
        color=text_main,
        ha="left",
        va="top",
    )
    fig.text(
        0.12,
        0.892,
        f"CXP History  —  {report['window_label']}",
        fontsize=10.5,
        color=text_muted,
        ha="left",
        va="top",
    )

    net = report["net_change"]
    net_col = col_gain_bright if net > 0 else (col_loss_bright if net < 0 else col_neutral)
    end_lvl = calculate_level(report["end_balance"])
    fig.text(
        0.945,
        0.938,
        f"{_fmt_signed(net)} CXP",
        fontsize=16,
        fontweight="bold",
        color=net_col,
        ha="right",
        va="top",
    )
    fig.text(
        0.945,
        0.892,
        f"Balance: {report['end_balance']:,} CXP (Lvl {end_lvl})",
        fontsize=10,
        color=text_muted,
        ha="right",
        va="top",
    )

    # ── Top Panel: Balance Trajectory ──
    pts = report["plot_points"]
    x_dts = [p[0] for p in pts]
    y_vals = [p[1] for p in pts]
    x_nums = mdates.date2num(x_dts)
    y_arr = np.asarray(y_vals, dtype=float)
    start_bal = float(report["start_balance"])

    # Period start reference line
    ax_top.axhline(
        start_bal,
        color="#64748b",
        linestyle="--",
        linewidth=1.1,
        alpha=0.75,
        zorder=2,
    )

    # Profit / Loss fill relative to period start
    ax_top.fill_between(
        x_nums,
        y_arr,
        start_bal,
        where=(y_arr >= start_bal),
        interpolate=True,
        color=col_gain,
        alpha=0.22,
        zorder=3,
    )
    ax_top.fill_between(
        x_nums,
        y_arr,
        start_bal,
        where=(y_arr <= start_bal),
        interpolate=True,
        color=col_loss,
        alpha=0.22,
        zorder=3,
    )

    line_color = col_gain_bright if net > 0 else (col_loss_bright if net < 0 else col_neutral)
    ax_top.plot(x_nums, y_arr, color=line_color, linewidth=2.3, zorder=4)

    # Plot subtle markers on actual granular events when count is modest
    event_dots_x = [mdates.date2num(p[0]) for p in pts if p[2]]
    event_dots_y = [p[1] for p in pts if p[2]]
    if 0 < len(event_dots_x) <= 60:
        ax_top.scatter(
            event_dots_x,
            event_dots_y,
            s=20,
            color=line_color,
            edgecolors=bg_card,
            linewidths=0.8,
            zorder=5,
        )

    # Start & End endpoint dots
    ax_top.scatter(
        [x_nums[0], x_nums[-1]],
        [y_arr[0], y_arr[-1]],
        s=36,
        color=[text_muted, line_color],
        edgecolors=bg_card,
        linewidths=1.0,
        zorder=6,
    )

    # Y-axis range with generous headroom for High/Low callout badges
    high_val = report["high_balance"]
    low_val = report["low_balance"]
    val_span = high_val - low_val
    pad = max(60.0, val_span * 0.28 if val_span > 0 else max(60.0, abs(high_val) * 0.08))
    ax_top.set_ylim(low_val - pad, high_val + pad)

    x_min_num, x_max_num = x_nums[0], x_nums[-1]
    x_span_num = max(1e-4, x_max_num - x_min_num)
    ax_top.set_xlim(x_min_num - x_span_num * 0.02, x_max_num + x_span_num * 0.02)

    # High & Low Callouts
    mid_x = (x_min_num + x_max_num) / 2.0
    if high_val == low_val:
        ax_top.annotate(
            f"High / Low: {high_val:,} CXP",
            xy=(mid_x, high_val),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color=text_main,
            bbox=dict(
                boxstyle="round,pad=0.32",
                facecolor="#1e293b",
                edgecolor=col_neutral,
                linewidth=1.0,
            ),
            zorder=7,
        )
        if report["event_count"] == 0:
            ax_top.text(
                mid_x,
                high_val - pad * 0.42,
                "No CXP changes recorded in this timeframe",
                ha="center",
                va="center",
                fontsize=9,
                color=text_muted,
                style="italic",
            )
    else:
        high_x = mdates.date2num(report["high_ts"])
        low_x = mdates.date2num(report["low_ts"])

        high_ha = "right" if high_x > (mid_x + x_span_num * 0.22) else (
            "left" if high_x < (mid_x - x_span_num * 0.22) else "center"
        )
        high_dx = -8 if high_ha == "right" else (8 if high_ha == "left" else 0)

        low_ha = "right" if low_x > (mid_x + x_span_num * 0.22) else (
            "left" if low_x < (mid_x - x_span_num * 0.22) else "center"
        )
        low_dx = -8 if low_ha == "right" else (8 if low_ha == "left" else 0)

        ax_top.scatter(
            [high_x],
            [high_val],
            s=52,
            color=col_gain_bright,
            edgecolors=bg_card,
            linewidths=1.2,
            zorder=7,
        )
        ax_top.annotate(
            f"▲ High: {high_val:,}",
            xy=(high_x, high_val),
            xytext=(high_dx, 9),
            textcoords="offset points",
            ha=high_ha,
            va="bottom",
            fontsize=8.6,
            fontweight="bold",
            color="#ecfdf5",
            bbox=dict(
                boxstyle="round,pad=0.28",
                facecolor="#064e3b",
                edgecolor=col_gain,
                linewidth=1.0,
            ),
            zorder=8,
        )

        ax_top.scatter(
            [low_x],
            [low_val],
            s=52,
            color=col_loss_bright,
            edgecolors=bg_card,
            linewidths=1.2,
            zorder=7,
        )
        ax_top.annotate(
            f"▼ Low: {low_val:,}",
            xy=(low_x, low_val),
            xytext=(low_dx, -10),
            textcoords="offset points",
            ha=low_ha,
            va="top",
            fontsize=8.6,
            fontweight="bold",
            color="#fff1f2",
            bbox=dict(
                boxstyle="round,pad=0.28",
                facecolor="#4c0519",
                edgecolor=col_loss,
                linewidth=1.0,
            ),
            zorder=8,
        )

    # Top panel axes & date formatting
    duration_hours = (
        report["window_end"] - report["window_start"]
    ).total_seconds() / 3600.0
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8, tz=timezone.utc)
    ax_top.xaxis.set_major_locator(locator)

    if duration_hours <= 36:
        date_fmt = mdates.DateFormatter("%H:%M", tz=timezone.utc)
    elif duration_hours <= 24 * 180:
        date_fmt = mdates.DateFormatter("%b %d", tz=timezone.utc)
    else:
        date_fmt = mdates.DateFormatter("%b '%y", tz=timezone.utc)
    ax_top.xaxis.set_major_formatter(date_fmt)

    ax_top.yaxis.set_major_formatter(
        FuncFormatter(lambda val, _pos: f"{int(round(val)):,}")
    )
    ax_top.set_ylabel("CXP Balance", fontsize=9.5, color=text_muted, labelpad=8)
    ax_top.grid(True, color=grid_col, linestyle="-", linewidth=0.8, alpha=0.85, zorder=1)
    ax_top.tick_params(axis="both", colors=text_muted, labelsize=8.5)
    for spine in ("top", "right"):
        ax_top.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax_top.spines[spine].set_color(spine_col)

    # Period start subtle right-margin tag
    ax_top.text(
        1.012,
        start_bal,
        f"Start\n{int(start_bal):,}",
        transform=ax_top.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=8,
        color=text_muted,
    )

    # ── Bottom Panel: Category Gains vs Losses Diverging Bar Chart ──
    cats = report["categories"]
    row_defs = [
        ("Messages", cats["message"]),
        ("Reactions", cats["reaction"]),
        ("Gambling", cats["gamble"]),
        ("Stealing", cats["steal"]),
        ("Trivia", cats["trivia"]),
        ("Admin Gives", cats["admin_give"]),
    ]
    if cats["other"]["net"] != 0:
        row_defs.append(("Channel Link", cats["other"]))
    if cats["legacy"]["net"] != 0:
        row_defs.append(("Pre-Ledger", cats["legacy"]))

    # Reverse so Messages is at the top of the horizontal bar chart
    row_defs_rev = list(reversed(row_defs))
    y_pos = np.arange(len(row_defs_rev))
    labels = [r[0] for r in row_defs_rev]
    gains = [r[1]["gained"] for r in row_defs_rev]
    losses = [r[1]["lost"] for r in row_defs_rev]  # negative or 0
    nets = [r[1]["net"] for r in row_defs_rev]

    max_extent = max(
        [100] + [abs(g) for g in gains] + [abs(l) for l in losses]
    )
    x_limit = max_extent * 1.32

    ax_bot.axvline(0, color="#475569", linewidth=1.2, zorder=3)
    ax_bot.barh(
        y_pos,
        gains,
        height=0.58,
        color=col_gain,
        alpha=0.88,
        zorder=4,
    )
    ax_bot.barh(
        y_pos,
        losses,
        height=0.58,
        color=col_loss,
        alpha=0.88,
        zorder=4,
    )

    label_offset = x_limit * 0.022
    for idx, (g, l, n) in enumerate(zip(gains, losses, nets)):
        if g > 0:
            ax_bot.text(
                g + label_offset,
                idx,
                _fmt_compact_signed(g, show_plus=True),
                va="center",
                ha="left",
                fontsize=8.2,
                fontweight="bold",
                color=col_gain_bright,
                zorder=5,
            )
        if l < 0:
            ax_bot.text(
                l - label_offset,
                idx,
                _fmt_compact_signed(l, show_plus=False),
                va="center",
                ha="right",
                fontsize=8.2,
                fontweight="bold",
                color=col_loss_bright,
                zorder=5,
            )

        # Right-hand aligned Net column outside plot area
        n_col = (
            col_gain_bright
            if n > 0
            else (col_loss_bright if n < 0 else text_muted)
        )
        ax_bot.text(
            1.015,
            idx,
            f"Net: {_fmt_signed(n)}",
            transform=ax_bot.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=8.4,
            fontweight="bold" if n != 0 else "normal",
            color=n_col,
        )

    ax_bot.set_xlim(-x_limit, x_limit)
    ax_bot.set_yticks(y_pos)
    ax_bot.set_yticklabels(labels, fontsize=9, color=text_main)
    ax_bot.xaxis.set_major_formatter(
        FuncFormatter(lambda val, _pos: _fmt_compact_signed(int(round(val)), show_plus=True) if val != 0 else "0")
    )
    ax_bot.set_title(
        "CXP Gains (+) vs Losses (-) by Source",
        fontsize=9.8,
        color=text_muted,
        loc="left",
        pad=6,
    )
    ax_bot.grid(True, axis="x", color=grid_col, linestyle="-", linewidth=0.8, alpha=0.85, zorder=1)
    ax_bot.tick_params(axis="x", colors=text_muted, labelsize=8.2)
    ax_bot.tick_params(axis="y", length=0)
    for spine in ("top", "right"):
        ax_bot.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax_bot.spines[spine].set_color(spine_col)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=bg_fig)
    buf.seek(0)
    return buf


def format_history_caption_html(target_name: str, report: dict) -> str:
    """Build the rich HTML breakdown caption (staying cleanly within Telegram's 1,024-char photo caption limit)."""
    safe_name = html.escape(target_name)
    safe_label = html.escape(report["window_label"])

    start_bal = report["start_balance"]
    end_bal = report["end_balance"]
    net = report["net_change"]
    high_val = report["high_balance"]
    low_val = report["low_balance"]

    lvl = calculate_level(end_bal)
    lvl_tag = _get_member_tag_string(lvl)
    lvl_str = f"Lvl {lvl} • {lvl_tag}" if lvl_tag else f"Lvl {lvl}"

    net_icon = "📈" if net > 0 else ("📉" if net < 0 else "➖")
    cats = report["categories"]

    # 1. Messages
    m = cats["message"]
    msg_detail = (
        f" <i>({m['count']:,} {'msg' if m['count'] == 1 else 'msgs'})</i>"
        if m["count"] > 0
        else ""
    )
    line_messages = f"💬 <b>Messages:</b> <code>{_fmt_signed(m['net'])} CXP</code>{msg_detail}"

    # 2. Reactions
    r = cats["reaction"]
    if r["gained"] > 0 and r["lost"] < 0:
        react_detail = f" <i>(+{r['gained']:,} / {r['lost']:,})</i>"
    elif r["count"] > 0:
        react_detail = f" <i>({r['count']:,} {'event' if r['count'] == 1 else 'events'})</i>"
    else:
        react_detail = ""
    line_reactions = f"❤️ <b>Reactions:</b> <code>{_fmt_signed(r['net'])} CXP</code>{react_detail}"

    # 3. Gambling
    g = cats["gamble"]
    if g["count"] > 0:
        won_str = f"+{g['gained']:,}" if g["gained"] > 0 else "0"
        lost_str = f"{g['lost']:,}" if g["lost"] < 0 else "0"
        gamble_detail = (
            f" <i>({won_str} won / {lost_str} lost • {g['wins']}W-{g['losses']}L)</i>"
        )
    else:
        gamble_detail = ""
    line_gambling = f"🎰 <b>Gambling:</b> <code>{_fmt_signed(g['net'])} CXP</code>{gamble_detail}"

    # 4. Stealing
    s = cats["steal"]
    if s["stole_count"] > 0 and s["robbed_count"] > 0:
        steal_detail = f" <i>(+{s['gained']:,} stolen / {s['lost']:,} robbed)</i>"
    elif s["stole_count"] > 0:
        steal_detail = f" <i>({s['stole_count']} {'steal' if s['stole_count'] == 1 else 'steals'})</i>"
    elif s["robbed_count"] > 0:
        steal_detail = f" <i>(robbed {s['robbed_count']}x)</i>"
    elif s["blocked_count"] > 0 or s["defended_count"] > 0:
        shields = s["blocked_count"] + s["defended_count"]
        steal_detail = f" <i>({shields} shielded)</i>"
    else:
        steal_detail = ""
    line_stealing = f"🕵️ <b>Stealing:</b> <code>{_fmt_signed(s['net'])} CXP</code>{steal_detail}"

    # 5. Trivia
    t = cats["trivia"]
    if t["gained"] > 0 and t["lost"] < 0:
        wins_note = f" • {t['wins']}W" if t["wins"] > 0 else ""
        trivia_detail = f" <i>(+{t['gained']:,} / {t['lost']:,}{wins_note})</i>"
    elif t["wins"] > 0:
        trivia_detail = f" <i>({t['wins']} {'win' if t['wins'] == 1 else 'wins'})</i>"
    elif t["questions_asked"] > 0:
        trivia_detail = f" <i>({t['questions_asked']} {'question' if t['questions_asked'] == 1 else 'questions'} hosted)</i>"
    elif t["wrong_guesses"] > 0:
        trivia_detail = f" <i>({t['wrong_guesses']} {'guess' if t['wrong_guesses'] == 1 else 'guesses'})</i>"
    else:
        trivia_detail = ""
    line_trivia = f"❓ <b>Trivia:</b> <code>{_fmt_signed(t['net'])} CXP</code>{trivia_detail}"

    # 6. Admin Gives
    a = cats["admin_give"]
    if a["gained"] > 0 and a["lost"] < 0:
        admin_detail = f" <i>(+{a['gained']:,} / {a['lost']:,})</i>"
    else:
        admin_detail = ""
    line_admin = f"🎁 <b>Admin Gives:</b> <code>{_fmt_signed(a['net'])} CXP</code>{admin_detail}"

    extra_lines: list[str] = []
    if cats["other"]["net"] != 0:
        extra_lines.append(
            f"🔗 <b>Channel Link:</b> <code>{_fmt_signed(cats['other']['net'])} CXP</code>"
        )
    if cats["legacy"]["net"] != 0:
        extra_lines.append(
            f"🕰️ <b>Pre-Ledger Net:</b> <code>{_fmt_signed(cats['legacy']['net'])} CXP</code>"
        )

    extra_block = ("\n" + "\n".join(extra_lines)) if extra_lines else ""

    caption = (
        f"📊 <b>CXP History — {safe_name}</b>\n"
        f"🗓️ <b>Timeframe:</b> {safe_label}\n\n"
        f"💰 <b>Balance:</b> <code>{start_bal:,}</code> → <code>{end_bal:,} CXP</code> <i>({html.escape(lvl_str)})</i>\n"
        f"{net_icon} <b>Net Change:</b> <code>{_fmt_signed(net)} CXP</code>\n"
        f"🔼 <b>High:</b> <code>{high_val:,} CXP</code>  •  🔽 <b>Low:</b> <code>{low_val:,} CXP</code>\n\n"
        f"<b>━━━ Breakdown by Source ━━━</b>\n"
        f"{line_messages}\n"
        f"{line_reactions}\n"
        f"{line_gambling}\n"
        f"{line_stealing}\n"
        f"{line_trivia}\n"
        f"{line_admin}"
        f"{extra_block}"
    )
    return caption


async def _resolve_history_target(
    update: Update, context: CallbackContext, user_query: str | None, has_text_mention: bool
) -> tuple[int | None, str | None, bool]:
    """
    Resolve target (target_id, target_name, is_unlinked_self_channel).
    Priority:
      1. Explicit user_query / text_mention in arguments
      2. Replied-to message author (ignoring forum topic root replies)
      3. Command caller (resolving linked channel to owner)
    """
    # 1. Explicit argument or text_mention
    if user_query or has_text_mention:
        resolved_id, resolved_name = await resolve_username(
            user_query or "", update, context
        )
        if not resolved_id:
            return None, None, False
        if resolved_id < 0:
            owner_id = await db.get_channel_owner(resolved_id)
            if owner_id:
                db_owner = await db.get_user(owner_id)
                owner_name = (
                    (db_owner.get("display_name") if db_owner else None)
                    or (
                        f"@{db_owner['username']}"
                        if db_owner and db_owner.get("username")
                        else None
                    )
                    or f"Owner {owner_id}"
                )
                return owner_id, f"{owner_name} (via {resolved_name})", False
        return resolved_id, resolved_name, False

    # 2. Reply targeting another user
    if update.message and update.message.reply_to_message and not getattr(
        update.message, "is_automatic_forward", False
    ):
        is_topic_root_reply = (
            update.message.is_topic_message
            and update.message.reply_to_message.message_id
            == update.message.message_thread_id
        )
        if not is_topic_root_reply:
            reply_msg = update.message.reply_to_message
            if reply_msg.sender_chat:
                cid = reply_msg.sender_chat.id
                cname = (
                    reply_msg.sender_chat.title
                    or reply_msg.sender_chat.username
                    or f"Channel {cid}"
                )
                if getattr(reply_msg.sender_chat, "type", "") == "channel":
                    owner_id = await db.get_channel_owner(cid)
                    if owner_id:
                        db_owner = await db.get_user(owner_id)
                        owner_name = (
                            (db_owner.get("display_name") if db_owner else None)
                            or f"Owner {owner_id}"
                        )
                        return owner_id, f"{owner_name} (via {cname})", False
                return cid, cname, False
            elif reply_msg.from_user and not getattr(reply_msg.from_user, "is_bot", False):
                ru = reply_msg.from_user
                rname = ru.first_name + (f" {ru.last_name}" if ru.last_name else "")
                return ru.id, rname, False

    # 3. Caller themselves
    if getattr(update.effective_user, "is_bot", True) and update.message.sender_chat:
        sc = update.message.sender_chat
        cid = sc.id
        cname = sc.title or sc.username or f"Channel {cid}"
        if getattr(sc, "type", "") == "channel":
            owner_id = await db.get_channel_owner(cid)
            if owner_id:
                db_owner = await db.get_user(owner_id)
                owner_name = (
                    (db_owner.get("display_name") if db_owner else None)
                    or f"Owner {owner_id}"
                )
                return owner_id, f"{owner_name} (via {cname})", False
            return cid, cname, True
        return cid, cname, False

    u = update.effective_user
    uname = u.first_name + (f" {u.last_name}" if u.last_name else "")
    return u.id, uname, False


async def history_cmd(update: Update, context: CallbackContext):
    """
    Handler for /history [timespan] [@username | username] (in any argument order).
    Generates a CXP trajectory & source breakdown chart with detailed totals and high/low points.
    """
    if not update.effective_chat or not update.message or not update.effective_user:
        return

    if (
        getattr(update.effective_user, "is_bot", True)
        and not update.message.sender_chat
    ):
        return

    chat_id = update.effective_chat.id
    thread_id = (
        update.message.message_thread_id
        if update.effective_chat.type != "private"
        else None
    )

    config = await db.get_config()

    # Check if message contains a text_mention entity so we never confuse a person's name with a timespan
    protected_user_text = None
    has_text_mention = False
    entities = update.message.parse_entities(["text_mention"])
    for _ent, ent_text in entities.items():
        has_text_mention = True
        protected_user_text = ent_text
        break

    now_utc = datetime.now(timezone.utc)
    window_spec, user_query = parse_history_args(
        context.args,
        config,
        now_utc=now_utc,
        protected_user_text=protected_user_text,
    )

    if window_spec.get("error"):
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=thread_id,
            text=window_spec["error"],
        )
        return

    target_id, target_name, is_unlinked_channel = await _resolve_history_target(
        update, context, user_query, has_text_mention
    )

    if is_unlinked_channel:
        bot_username = context.bot.username
        keyboard = (
            InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "🔗 Link Channel",
                            url=f"https://t.me/{bot_username}?start=setchannel",
                        )
                    ]
                ]
            )
            if bot_username
            else None
        )
        await update.message.reply_text(
            "⚠️ Please link your channel to your user account first to track and view CXP history.",
            reply_markup=keyboard,
        )
        return

    if not target_id:
        safe_q = html.escape(user_query or "")
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=thread_id,
            text=(
                f"❌ Could not find a user or timeframe matching <code>{safe_q}</code>.\n\n"
                f"<b>Examples:</b>\n"
                f"• <code>/history</code> or <code>/history today</code>\n"
                f"• <code>/history @username week</code>\n"
                f"• <code>/history month username</code>\n"
                f"<i>Timeframes: today, yesterday, week, last week, month, last month, 24h, 7d, 30d, year, contest, all</i>"
            ),
            parse_mode=ParseMode.HTML,
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

    # Ensure user exists if querying self, then fetch all CXP history data
    raw_data = await db.get_user_cxp_history_data(target_id)
    if not raw_data or (
        not raw_data.get("user")
        and not raw_data.get("daily")
        and not raw_data.get("ledger")
    ):
        if target_id == update.effective_user.id:
            await db.get_user(target_id)
            raw_data = await db.get_user_cxp_history_data(target_id)
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=thread_id,
                text="ℹ️ No CXP records found for that user yet.",
            )
            return

    user_row = raw_data.get("user") or {}
    if not target_name:
        target_name = (
            user_row.get("display_name")
            or (f"@{user_row['username']}" if user_row.get("username") else None)
            or f"User {target_id}"
        )

    report = build_cxp_history_report(
        user_id=target_id,
        raw_data=raw_data,
        window_spec=window_spec,
        now_utc=now_utc,
    )

    chart_buf = await asyncio.to_thread(
        render_history_chart_png, target_name, report
    )
    caption_html = format_history_caption_html(target_name, report)

    if len(caption_html) <= MessageLimit.CAPTION_LENGTH:
        await context.bot.send_photo(
            chat_id=chat_id,
            message_thread_id=thread_id,
            photo=chart_buf,
            filename="cxp_history.png",
            caption=caption_html,
            parse_mode=ParseMode.HTML,
        )
    else:
        short_caption = (
            f"📊 <b>CXP History — {html.escape(target_name)}</b> "
            f"({html.escape(report['window_label'])})"
        )
        await context.bot.send_photo(
            chat_id=chat_id,
            message_thread_id=thread_id,
            photo=chart_buf,
            filename="cxp_history.png",
            caption=short_caption,
            parse_mode=ParseMode.HTML,
        )
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=thread_id,
            text=caption_html,
            parse_mode=ParseMode.HTML,
        )
