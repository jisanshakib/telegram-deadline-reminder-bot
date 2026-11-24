"""Telegram update handlers for channel monitoring and the private Admin Control Panel."""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional, Union

from telegram import Chat, InlineKeyboardMarkup, Update
from telegram.constants import ChatType, ParseMode
from telegram.error import BadRequest, Forbidden, TelegramError
from telegram.ext import (
    Application,
    ApplicationHandlerStop,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    TypeHandler,
    filters,
)

from database import Database, Reminder, extract_internal_channel_id
from gemini_parser import GeminiDeadlineParser
from keyboards import (
    WINDOW_OPTIONS,
    build_active_reminders_keyboard,
    build_delete_confirm_keyboard,
    build_detection_confirm_keyboard,
    build_instant_reminder_list_keyboard,
    build_main_menu_keyboard,
    build_manual_cache_miss_keyboard,
    build_manual_cancel_keyboard,
    build_manual_final_confirm_keyboard,
    build_manual_mode_keyboard,
    build_manual_window_keyboard,
    build_overview_keyboard,
    build_post_activation_keyboard,
    build_repeat_mode_keyboard,
    build_window_selection_keyboard,
)
from scheduler import post_reminder_to_channel

logger = logging.getLogger(__name__)

_TELEGRAM_MAX_MESSAGE_LEN = 4096
_TELEGRAM_MAX_CALLBACK_ALERT_LEN = 195

# Matches private links: https://t.me/c/1234567890/42 or https://t.me/c/1234567890/1/42
_PRIVATE_LINK_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:t\.me|telegram\.me)/c/(\d+)/(?:\d+/)?(\d+)(?=[/?#\s]|$)",
    re.IGNORECASE,
)

# Matches public links: https://t.me/channel_name/42 or https://t.me/s/channel_name/42
_PUBLIC_LINK_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:t\.me|telegram\.me)/(?:s/)?([A-Za-z][A-Za-z0-9_]{3,})/(\d+)(?=[/?#\s]|$)",
    re.IGNORECASE,
)

_RESERVED_PUBLIC_SLUGS = frozenset({"c", "s", "addstickers", "joinchat", "boost", "share", "iv"})

_ACCEPTED_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",  # 2026-12-31
    "%d-%m-%Y",  # 31-12-2026
    "%d %B %Y",  # 31 December 2026
    "%d %b %Y",  # 31 Dec 2026
    "%d %B, %Y",  # 31 December, 2026
    "%d %b, %Y",  # 31 Dec, 2026
    "%B %d, %Y",  # December 31, 2026
    "%b %d, %Y",  # Dec 31, 2026
    "%B %d %Y",  # December 31 2026
    "%b %d %Y",  # Dec 31 2026
    "%d/%m/%Y",  # 31/12/2026
    "%Y/%m/%d",  # 2026/12/31
    "%d.%m.%Y",  # 31.12.2026
    "%Y.%m.%d",  # 2026.12.31
)


@dataclass(slots=True)
class ParsedTelegramLink:
    """Normalized components extracted from a Telegram channel post URL."""

    post_link: str
    message_id: int
    channel_id: Optional[str] = None
    channel_username: Optional[str] = None
    internal_channel_id: Optional[str] = None


def build_post_link(chat: Chat, message_id: int) -> str:
    """Build canonical Telegram link for public or private channel posts."""
    if chat.username:
        return f"https://t.me/{chat.username}/{message_id}"
    internal_id = extract_internal_channel_id(chat.id) or str(abs(chat.id))
    return f"https://t.me/c/{internal_id}/{message_id}"


def parse_telegram_post_link(raw_text: str) -> Optional[ParsedTelegramLink]:
    """Parse a public or private Telegram channel post link from arbitrary text."""
    text = raw_text.strip()
    if not text:
        return None

    private_match = _PRIVATE_LINK_RE.search(text)
    if private_match:
        internal_id = private_match.group(1)
        message_id = int(private_match.group(2))
        if message_id >= 1:
            return ParsedTelegramLink(
                post_link=f"https://t.me/c/{internal_id}/{message_id}",
                message_id=message_id,
                channel_id=f"-100{internal_id}",
                internal_channel_id=internal_id,
            )

    for public_match in _PUBLIC_LINK_RE.finditer(text):
        username = public_match.group(1)
        if username.lower() in _RESERVED_PUBLIC_SLUGS:
            continue
        message_id = int(public_match.group(2))
        if message_id < 1:
            continue
        return ParsedTelegramLink(
            post_link=f"https://t.me/{username}/{message_id}",
            message_id=message_id,
            channel_username=username.lower(),
        )

    return None


def parse_user_date(raw_text: str) -> Optional[date]:
    """Parse manually typed End Date in YYYY-MM-DD, DD-MM-YYYY, or '31 December 2026' formats."""
    cleaned = " ".join(raw_text.strip().split())
    if not cleaned:
        return None
    # Strip English ordinal suffixes such as 1st, 2nd, 3rd, 31st
    cleaned = re.sub(r"\b(\d{1,2})(?:st|nd|rd|th)\b", r"\1", cleaned, flags=re.IGNORECASE)
    # Normalize spaces around commas (e.g. "31 December , 2026" -> "31 December, 2026")
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)

    for fmt in _ACCEPTED_DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def _matches_channel(chat: Chat, configured_channel_id: Union[int, str]) -> bool:
    """Check whether an incoming channel_post originates from the configured CHANNEL_ID."""
    if isinstance(configured_channel_id, int):
        return chat.id == configured_channel_id

    target = str(configured_channel_id).strip()
    if target.lstrip("-").isdigit():
        return chat.id == int(target)

    if chat.username:
        return chat.username.lower() == target.lstrip("@").lower()

    return False


def _is_private_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Return True strictly when the update comes from ADMIN_USER_ID in a private chat."""
    user = update.effective_user
    chat = update.effective_chat
    admin_id: int = context.bot_data["admin_user_id"]
    return (
        user is not None
        and user.id == admin_id
        and chat is not None
        and chat.type == ChatType.PRIVATE
    )


def _get_db(context: ContextTypes.DEFAULT_TYPE) -> Database:
    return context.bot_data["db"]


def _get_parser(context: ContextTypes.DEFAULT_TYPE) -> GeminiDeadlineParser:
    return context.bot_data["parser"]


def _get_channel_id(context: ContextTypes.DEFAULT_TYPE) -> Union[int, str]:
    return context.bot_data["channel_id"]


def _parse_positive_int(raw: str) -> Optional[int]:
    try:
        value = int(raw)
        return value if value >= 0 else None
    except ValueError:
        return None


async def _safe_answer(
    query,
    text: Optional[str] = None,
    *,
    show_alert: bool = False,
) -> None:
    """Safely answer callback queries while respecting Telegram's 200-char alert limit."""
    safe_text = None
    if text:
        safe_text = (
            text
            if len(text) <= _TELEGRAM_MAX_CALLBACK_ALERT_LEN
            else f"{text[: _TELEGRAM_MAX_CALLBACK_ALERT_LEN - 1]}…"
        )
    try:
        await query.answer(text=safe_text, show_alert=show_alert)
    except BadRequest as exc:
        # Ignore expired callback query timeouts cleanly
        logger.debug("Callback answer skipped: %s", exc)


async def _safe_edit_message(
    query,
    text: str,
    reply_markup: Optional[InlineKeyboardMarkup] = None,
) -> None:
    try:
        await query.edit_message_text(
            text=text,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup,
            disable_web_page_preview=True,
        )
    except BadRequest as exc:
        if "Message is not modified" in str(exc):
            return
        raise


def _format_days_left_label(days_left: int) -> str:
    if days_left < 0:
        return f"Expired ({abs(days_left)} day(s) ago)"
    if days_left == 0:
        return "0 days (Ends today)"
    unit = "day" if days_left == 1 else "days"
    return f"{days_left} {unit}"


def _format_duration_text(
    start_date: Optional[str],
    end_date: str,
    today: date,
) -> str:
    end_dt = date.fromisoformat(end_date)
    days_left = (end_dt - today).days
    rem_label = _format_days_left_label(days_left)

    if start_date:
        start_dt = date.fromisoformat(start_date)
        span = (end_dt - start_dt).days
        if span == 0:
            span_str = "Same-day event"
        elif span > 0:
            unit = "day" if span == 1 else "days"
            span_str = f"{span} {unit}"
        else:
            span_str = f"{abs(span)} days"
        return f"{span_str} ({rem_label})"

    if days_left > 0:
        return f"{rem_label} remaining from today"
    return rem_label


def _escape_and_fit_post_details(post_text: str, budget_chars: int) -> str:
    """HTML-escape post text and guarantee the escaped output fits inside budget_chars."""
    cleaned = post_text.strip()
    escaped = html.escape(cleaned)
    if len(escaped) <= budget_chars:
        return escaped

    suffix = "...\n[truncated]"
    max_body = max(100, budget_chars - len(suffix))
    # Trim raw text iteratively so we never split an HTML entity like &amp;
    low, high = 0, len(cleaned)
    while low < high:
        mid = (low + high + 1) // 2
        if len(html.escape(cleaned[:mid])) <= max_body:
            low = mid
        else:
            high = mid - 1
    return f"{html.escape(cleaned[:low].rstrip())}{suffix}"


def _format_detection_message(reminder: Reminder, today: date) -> str:
    duration_text = _format_duration_text(reminder.start_date, reminder.end_date, today)
    safe_link = html.escape(reminder.post_link, quote=True)

    header = (
        "🔔 <b>New Event Deadline Detected</b>\n\n"
        f"📌 <b>Title:</b> {html.escape(reminder.title)}\n"
        f"🗓 <b>Start Date:</b> {html.escape(reminder.start_date or 'N/A')}\n"
        f"⏳ <b>End Date:</b> {html.escape(reminder.end_date)}\n"
        f"⏱ <b>Total Duration:</b> {html.escape(duration_text)}\n"
        f'🔗 <b>Original Post:</b> <a href="{safe_link}">Open Post</a>\n\n'
        "📝 <b>Post Details:</b>\n<blockquote>"
    )
    footer = "</blockquote>\n\nWould you like to activate automated reminders for this event?"

    budget = min(2400, _TELEGRAM_MAX_MESSAGE_LEN - len(header) - len(footer) - 64)
    post_preview = _escape_and_fit_post_details(reminder.post_text, budget)
    return f"{header}{post_preview}{footer}"


def _format_overview_message(
    reminder: Reminder,
    today: date,
    banner: Optional[str] = None,
) -> str:
    days_left = reminder.days_left(today)
    safe_link = html.escape(reminder.post_link, quote=True)
    header = f"{html.escape(banner)}\n\n" if banner else ""

    return (
        f"{header}"
        "📄 <b>Full Event Overview</b>\n\n"
        f"📌 <b>Event Title:</b> {html.escape(reminder.title)}\n"
        f"🗓 <b>Start Date:</b> {html.escape(reminder.start_date or 'N/A')}\n"
        f"⏳ <b>End Date:</b> {html.escape(reminder.end_date)}\n"
        f"📆 <b>Remaining Days:</b> {html.escape(_format_days_left_label(days_left))}\n"
        f'🔗 <b>Original Post Link:</b> <a href="{safe_link}">Open</a>\n\n'
        "⚙️ <b>Saved Reminder Settings</b>\n"
        f"• <b>Window:</b> {html.escape(reminder.window_label)}\n"
        f"• <b>Repeat Mode:</b> {html.escape(reminder.repeat_mode_label)}\n"
        f"• <b>Status:</b> {html.escape(reminder.status.capitalize())}\n"
        f"• <b>Last Reminded Date:</b> {html.escape(reminder.last_reminded_on or 'Never')}"
    )


def _format_active_list_message(reminders: list[Reminder], today: date) -> str:
    if not reminders:
        return (
            "📋 <b>Active Reminders</b>\n\n"
            "No active reminders right now.\n"
            "New channel posts with deadlines will appear for approval automatically, "
            "or you can add one via <b>Set Reminder Post</b>."
        )

    lines = [f"📋 <b>Active Reminders ({len(reminders)})</b>\n"]
    shown = 0
    footer = "\nSelect a reminder to open its Overview or click <b>Delete</b> to cancel it."

    for idx, rem in enumerate(reminders[:15], start=1):
        days_left = rem.days_left(today)
        short_title = rem.title if len(rem.title) <= 65 else f"{rem.title[:64].rstrip()}…"
        entry = (
            f"{idx}. <b>{html.escape(short_title)}</b>\n"
            f"   ⏳ End: <code>{html.escape(rem.end_date)}</code> ({html.escape(_format_days_left_label(days_left))}) | "
            f"🪟 {html.escape(rem.window_label)} | 🔁 {html.escape(rem.repeat_mode_label)}"
        )
        # Keep total message length well within Telegram's 4096-char ceiling
        projected_len = sum(len(l) + 1 for l in lines) + len(entry) + len(footer) + 120
        if projected_len > _TELEGRAM_MAX_MESSAGE_LEN:
            break
        lines.append(entry)
        shown += 1

    if len(reminders) > shown:
        lines.append(
            f"\n<i>Showing summary of first {shown} items. "
            f"Use pagination buttons below to browse all {len(reminders)}.</i>"
        )

    lines.append(footer)
    return "\n".join(lines)


def _format_instant_list_message(reminders: list[Reminder]) -> str:
    if not reminders:
        return (
            "⚡ <b>Instant Reminder</b>\n\n"
            "There are no active reminders to dispatch.\n"
            "Confirm a detected event or use <b>Set Reminder Post</b> first."
        )
    return (
        f"⚡ <b>Instant Reminder ({len(reminders)} active)</b>\n\n"
        "Choose an event below to inspect its <b>Full Event Overview</b>, "
        "edit reminder settings, or send an instant reminder to the channel."
    )


def _format_manual_overview(flow: dict[str, Any], today: date) -> str:
    title = flow.get("title") or "Untitled Event"
    start_date = flow.get("start_date") or "N/A"
    end_date = flow["end_date"]
    post_link = flow["post_link"]
    window_days = flow.get("window_days")
    repeat_mode = flow.get("repeat_mode")

    days_left = (date.fromisoformat(end_date) - today).days
    window_str = (
        f"Last {window_days} {'Day' if window_days == 1 else 'Days'}"
        if window_days
        else "Not selected yet"
    )
    mode_str = (
        "Normal (Daily)"
        if repeat_mode == "normal"
        else ("Alternate Day" if repeat_mode == "alternate" else "Not selected yet")
    )
    safe_link = html.escape(post_link, quote=True)

    extra_note = ""
    if days_left == 0:
        extra_note = (
            "\n\nℹ️ <i>Note: This event ends today (0 days left). Automated reminders run "
            "when 1 ≤ days_left ≤ window_days, so you can send an Instant Reminder after saving.</i>"
        )

    return (
        "📝 <b>Manual Reminder — Event Overview</b>\n\n"
        f"📌 <b>Event Title:</b> {html.escape(title)}\n"
        f"🗓 <b>Start Date:</b> {html.escape(start_date)}\n"
        f"⏳ <b>End Date:</b> {html.escape(end_date)}\n"
        f"📆 <b>Remaining Days:</b> {html.escape(_format_days_left_label(days_left))}\n"
        f'🔗 <b>Original Post Link:</b> <a href="{safe_link}">Open</a>\n\n'
        "⚙️ <b>Reminder Configuration</b>\n"
        f"• <b>Window:</b> {html.escape(window_str)}\n"
        f"• <b>Repeat Mode:</b> {html.escape(mode_str)}"
        f"{extra_note}"
    )


def _fallback_title_from_text(post_text: str) -> str:
    for line in post_text.splitlines():
        cleaned = " ".join(line.strip().split())
        if cleaned:
            return cleaned[:80]
    return "Channel Event"


# ======================================================================
# Global Strict Private Admin-Only Access Gatekeeper (group=-1)
# ======================================================================


async def enforce_admin_access(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Strictly block all non-admin users and non-private admin interactions.

    Channel posts (`channel_post` / `edited_channel_post`) are allowed through to the
    channel monitor, which separately verifies `CHANNEL_ID`. Every other update
    (commands, messages, button clicks, group chats) MUST originate from `ADMIN_USER_ID`
    inside a private chat; otherwise it is rejected immediately and halted via
    `ApplicationHandlerStop`.
    """
    if update.channel_post is not None or update.edited_channel_post is not None:
        return

    if _is_private_admin(update, context):
        return

    user = update.effective_user
    chat = update.effective_chat
    logger.warning(
        "Blocked unauthorized update (user_id=%s, chat_id=%s, chat_type=%s)",
        user.id if user else None,
        chat.id if chat else None,
        chat.type if chat else None,
    )

    if update.callback_query is not None:
        await _safe_answer(
            update.callback_query,
            "⛔ Access denied. This bot is restricted to its administrator.",
            show_alert=True,
        )
    elif update.effective_message is not None and chat is not None and chat.type == ChatType.PRIVATE:
        try:
            await update.effective_message.reply_text(
                "⛔ Access denied. This bot is strictly private and restricted to its administrator."
            )
        except (Forbidden, TelegramError):
            pass

    raise ApplicationHandlerStop


# ======================================================================
# A. Channel Post Listener (Automatic Detection & Post Cache)
# ======================================================================


async def handle_channel_post(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Monitor incoming channel posts, cache text, and detect event deadlines via Gemini."""
    post = update.channel_post
    if post is None:
        return

    configured_channel = _get_channel_id(context)
    if not _matches_channel(post.chat, configured_channel):
        logger.debug(
            "Ignoring channel_post from unconfigured chat %s (%s)",
            post.chat.id,
            post.chat.username,
        )
        return

    post_text = (post.text or post.caption or "").strip()
    if not post_text:
        return

    db = _get_db(context)
    post_link = build_post_link(post.chat, post.message_id)

    # Always cache incoming channel posts for manual link lookup
    await db.cache_channel_post(
        channel_id=post.chat.id,
        channel_username=post.chat.username,
        message_id=post.message_id,
        post_text=post_text,
        post_link=post_link,
    )

    # Avoid duplicate pending/active reminders for the same post
    if await db.has_open_reminder_for_post(post.chat.id, post.message_id, post_link=post_link):
        logger.debug("Post %s/%d already has an open reminder", post.chat.id, post.message_id)
        return

    parser = _get_parser(context)
    today = db.today()

    try:
        extraction = await parser.extract_deadline(post_text=post_text, current_date=today)
    except Exception:
        logger.exception(
            "Gemini extraction failed for channel post %s/%d",
            post.chat.id,
            post.message_id,
        )
        return

    if not (extraction.has_deadline and extraction.end_date):
        logger.info("No deadline detected in channel post %s/%d", post.chat.id, post.message_id)
        return

    # Do not notify the admin for events whose extracted deadline is already in the past
    if date.fromisoformat(extraction.end_date) < today:
        logger.info(
            "Ignoring past deadline (%s < %s) detected in channel post %s/%d",
            extraction.end_date,
            today.isoformat(),
            post.chat.id,
            post.message_id,
        )
        return

    title = extraction.title or _fallback_title_from_text(post_text)
    reminder = await db.create_reminder(
        channel_id=post.chat.id,
        message_id=post.message_id,
        post_link=post_link,
        title=title,
        post_text=post_text,
        start_date=extraction.start_date,
        end_date=extraction.end_date,
        status="pending",
    )

    admin_id: int = context.bot_data["admin_user_id"]
    try:
        await context.bot.send_message(
            chat_id=admin_id,
            text=_format_detection_message(reminder, today),
            parse_mode=ParseMode.HTML,
            reply_markup=build_detection_confirm_keyboard(reminder.id),
            disable_web_page_preview=True,
        )
        logger.info(
            "Sent pending reminder #%d (%s, end=%s) to admin %d",
            reminder.id,
            reminder.title,
            reminder.end_date,
            admin_id,
        )
    except Exception:
        logger.exception("Failed to notify admin %d about reminder #%d", admin_id, reminder.id)


async def handle_edited_channel_post(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Keep the channel post cache updated when channel posts are edited."""
    post = update.edited_channel_post
    if post is None:
        return
    if not _matches_channel(post.chat, _get_channel_id(context)):
        return
    post_text = (post.text or post.caption or "").strip()
    if not post_text:
        return
    db = _get_db(context)
    await db.cache_channel_post(
        channel_id=post.chat.id,
        channel_username=post.chat.username,
        message_id=post.message_id,
        post_text=post_text,
        post_link=build_post_link(post.chat, post.message_id),
    )


# ======================================================================
# C. Admin Control Panel Commands (/start, /admin, /cancel, unknown)
# ======================================================================


async def _render_main_menu_text(context: ContextTypes.DEFAULT_TYPE) -> str:
    db = _get_db(context)
    await db.expire_past_reminders()
    active = await db.list_active_reminders()
    tz_name = context.bot_data["timezone"]
    daily_hour = context.bot_data["daily_hour"]
    daily_minute = context.bot_data["daily_minute"]

    return (
        "🛠 <b>Deadline Reminder — Admin Control Panel</b>\n\n"
        f"📊 <b>Active Reminders:</b> {len(active)}\n"
        f"🕒 <b>Daily Dispatch:</b> {daily_hour:02d}:{daily_minute:02d} ({html.escape(tz_name)})\n\n"
        "Select an option below:"
    )


async def cmd_admin_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start and /admin commands (strictly restricted to ADMIN_USER_ID)."""
    if not _is_private_admin(update, context):
        return

    context.user_data.pop("manual_flow", None)
    text = await _render_main_menu_text(context)
    if update.effective_message:
        await update.effective_message.reply_text(
            text=text,
            parse_mode=ParseMode.HTML,
            reply_markup=build_main_menu_keyboard(),
        )


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Cancel any active manual reminder entry step."""
    if not _is_private_admin(update, context):
        return

    had_flow = context.user_data.pop("manual_flow", None) is not None
    note = "✅ Cancelled manual reminder setup.\n\n" if had_flow else ""
    text = note + await _render_main_menu_text(context)
    if update.effective_message:
        await update.effective_message.reply_text(
            text=text,
            parse_mode=ParseMode.HTML,
            reply_markup=build_main_menu_keyboard(),
        )


async def cmd_unknown_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Guide the admin when an unrecognized command is entered."""
    if not _is_private_admin(update, context):
        return
    if update.effective_message:
        await update.effective_message.reply_text(
            "Unknown command. Use /admin (or /start) to open the Admin Control Panel, "
            "or /cancel to exit an active setup."
        )


# ======================================================================
# B & C. Callback Query Router (Confirm/Decline, Menus, Overview, Edit)
# ======================================================================


async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Central router for all inline keyboard actions."""
    query = update.callback_query
    if query is None:
        return

    if not _is_private_admin(update, context):
        await _safe_answer(query, "⛔ Unauthorized. Admin access required.", show_alert=True)
        return

    data = (query.data or "").strip()
    if data == "noop":
        await _safe_answer(query)
        return

    parts = data.split(":")
    db = _get_db(context)
    today = db.today()

    # --- Main Menu Navigation ---
    if data == "menu:home":
        await _safe_answer(query)
        context.user_data.pop("manual_flow", None)
        await _safe_edit_message(
            query,
            await _render_main_menu_text(context),
            reply_markup=build_main_menu_keyboard(),
        )
        return

    if len(parts) == 3 and parts[0] == "menu" and parts[1] == "active":
        page = _parse_positive_int(parts[2])
        if page is None:
            await _safe_answer(query, "Invalid page.", show_alert=True)
            return
        await _safe_answer(query)
        await db.expire_past_reminders()
        reminders = await db.list_active_reminders()
        await _safe_edit_message(
            query,
            _format_active_list_message(reminders, today),
            reply_markup=build_active_reminders_keyboard(reminders, page=page),
        )
        return

    if len(parts) == 3 and parts[0] == "menu" and parts[1] == "instant":
        page = _parse_positive_int(parts[2])
        if page is None:
            await _safe_answer(query, "Invalid page.", show_alert=True)
            return
        await _safe_answer(query)
        await db.expire_past_reminders()
        reminders = await db.list_active_reminders()
        await _safe_edit_message(
            query,
            _format_instant_list_message(reminders),
            reply_markup=build_instant_reminder_list_keyboard(reminders, page=page),
        )
        return

    if data == "menu:set_post":
        await _safe_answer(query)
        context.user_data["manual_flow"] = {"step": "awaiting_link"}
        await _safe_edit_message(
            query,
            "🔗 <b>Set Reminder Post (Manual Link Entry)</b>\n\n"
            "Send the Telegram channel post link you want to track.\n\n"
            "<b>Accepted examples:</b>\n"
            "• <code>https://t.me/channel_username/123</code>\n"
            "• <code>https://t.me/c/1234567890/123</code>\n\n"
            "Use /cancel or click Cancel anytime to exit.",
            reply_markup=build_manual_cancel_keyboard(),
        )
        return

    # --- Auto-Detection Multi-Step Confirmation Flow ---
    if len(parts) == 2 and parts[0] == "detect_back":
        reminder_id = _parse_positive_int(parts[1])
        if reminder_id is None:
            await _safe_answer(query, "Invalid reminder reference.", show_alert=True)
            return
        reminder = await db.get_reminder(reminder_id)
        if reminder is None:
            await _safe_answer(query, "Reminder no longer exists.", show_alert=True)
            return
        if reminder.status != "pending":
            await _safe_answer(
                query,
                f"This reminder is already {reminder.status}.",
                show_alert=True,
            )
            return
        await _safe_answer(query)
        await _safe_edit_message(
            query,
            _format_detection_message(reminder, today),
            reply_markup=build_detection_confirm_keyboard(reminder.id),
        )
        return

    if len(parts) == 2 and parts[0] == "decline":
        reminder_id = _parse_positive_int(parts[1])
        if reminder_id is None:
            await _safe_answer(query, "Invalid reminder reference.", show_alert=True)
            return
        reminder = await db.get_reminder(reminder_id)
        if reminder is None:
            await _safe_answer(query, "Reminder no longer exists.", show_alert=True)
            return
        if reminder.status != "pending":
            await _safe_answer(
                query,
                f"Cannot decline: reminder is already {reminder.status}.",
                show_alert=True,
            )
            return
        await db.update_reminder_status(reminder_id, "declined")
        await _safe_answer(query, "Reminder declined.")
        await _safe_edit_message(
            query,
            f"❌ <b>Reminder Declined</b>\n\n"
            f"📌 <b>Title:</b> {html.escape(reminder.title)}\n"
            f"⏳ <b>End Date:</b> {html.escape(reminder.end_date)}\n\n"
            "No reminders will be scheduled for this post.",
            reply_markup=None,
        )
        return

    if len(parts) == 2 and parts[0] == "confirm":
        reminder_id = _parse_positive_int(parts[1])
        if reminder_id is None:
            await _safe_answer(query, "Invalid reminder reference.", show_alert=True)
            return
        reminder = await db.get_reminder(reminder_id)
        if reminder is None:
            await _safe_answer(query, "Reminder not found.", show_alert=True)
            return
        if reminder.status != "pending":
            await _safe_answer(
                query,
                f"This reminder is already {reminder.status}.",
                show_alert=True,
            )
            return
        await _safe_answer(query)
        duration_text = _format_duration_text(reminder.start_date, reminder.end_date, today)
        await _safe_edit_message(
            query,
            "🪟 <b>Step 1 of 2 — Select Reminder Window</b>\n\n"
            f"📌 <b>Event:</b> {html.escape(reminder.title)}\n"
            f"🗓 <b>Start Date:</b> {html.escape(reminder.start_date or 'N/A')}\n"
            f"⏳ <b>End Date:</b> {html.escape(reminder.end_date)}\n"
            f"⏱ <b>Total Duration:</b> {html.escape(duration_text)}\n\n"
            "How many days before the deadline should reminders run?",
            reply_markup=build_window_selection_keyboard(
                reminder.id,
                prefix="win",
                back_callback=f"detect_back:{reminder.id}",
            ),
        )
        return

    if len(parts) == 3 and parts[0] == "win":
        reminder_id = _parse_positive_int(parts[1])
        window_days = _parse_positive_int(parts[2])
        if reminder_id is None or window_days not in WINDOW_OPTIONS:
            await _safe_answer(query, "Invalid window option.", show_alert=True)
            return
        reminder = await db.get_reminder(reminder_id)
        if reminder is None:
            await _safe_answer(query, "Reminder not found.", show_alert=True)
            return
        if reminder.status != "pending":
            await _safe_answer(
                query,
                f"This reminder is already {reminder.status}.",
                show_alert=True,
            )
            return
        await _safe_answer(query)
        suffix = "Day" if window_days == 1 else "Days"
        await _safe_edit_message(
            query,
            "🔁 <b>Step 2 of 2 — Select Repeat Mode</b>\n\n"
            f"📌 <b>Event:</b> {html.escape(reminder.title)}\n"
            f"⏳ <b>End Date:</b> {html.escape(reminder.end_date)}\n"
            f"🪟 <b>Selected Window:</b> Last {window_days} {suffix}\n\n"
            "• <b>Normal (Daily):</b> 1 reminder every day during the selected window.\n"
            "• <b>Alternate Day:</b> 1 reminder every 2 days starting from the first day of the window.",
            reply_markup=build_repeat_mode_keyboard(
                reminder.id,
                window_days,
                prefix="mode",
                back_callback=f"confirm:{reminder.id}",
            ),
        )
        return

    if len(parts) == 4 and parts[0] == "mode":
        reminder_id = _parse_positive_int(parts[1])
        window_days = _parse_positive_int(parts[2])
        repeat_mode = parts[3]
        if (
            reminder_id is None
            or window_days not in WINDOW_OPTIONS
            or repeat_mode not in {"normal", "alternate"}
        ):
            await _safe_answer(query, "Invalid reminder configuration.", show_alert=True)
            return

        existing = await db.get_reminder(reminder_id)
        if existing is None:
            await _safe_answer(query, "Reminder not found.", show_alert=True)
            return
        if existing.status != "pending":
            await _safe_answer(
                query,
                f"This reminder is already {existing.status}.",
                show_alert=True,
            )
            return

        updated = await db.update_reminder_settings(
            reminder_id,
            window_days=window_days,
            repeat_mode=repeat_mode,
            activate=True,
        )
        if updated is None:
            await _safe_answer(query, "Reminder not found.", show_alert=True)
            return

        await _safe_answer(query, "Reminder activated!")
        safe_link = html.escape(updated.post_link, quote=True)
        await _safe_edit_message(
            query,
            "✅ <b>Reminder Confirmed & Activated</b>\n\n"
            f"📌 <b>Event Title:</b> {html.escape(updated.title)}\n"
            f"🗓 <b>Start Date:</b> {html.escape(updated.start_date or 'N/A')}\n"
            f"⏳ <b>End Date:</b> {html.escape(updated.end_date)}\n"
            f"📆 <b>Remaining Days:</b> {html.escape(_format_days_left_label(updated.days_left(today)))}\n"
            f"🪟 <b>Reminder Window:</b> {html.escape(updated.window_label)}\n"
            f"🔁 <b>Repeat Mode:</b> {html.escape(updated.repeat_mode_label)}\n"
            f'🔗 <b>Original Post:</b> <a href="{safe_link}">Open</a>',
            reply_markup=build_post_activation_keyboard(updated.id),
        )
        return

    # --- Full Event Overview & Instant Dispatch ---
    if len(parts) == 3 and parts[0] == "overview":
        reminder_id = _parse_positive_int(parts[1])
        source = parts[2] if parts[2] in {"active", "instant"} else "instant"
        if reminder_id is None:
            await _safe_answer(query, "Invalid reminder reference.", show_alert=True)
            return
        reminder = await db.get_reminder(reminder_id)
        if reminder is None:
            await _safe_answer(query, "Reminder no longer exists.", show_alert=True)
            reminders = await db.list_active_reminders()
            if source == "active":
                await _safe_edit_message(
                    query,
                    _format_active_list_message(reminders, today),
                    reply_markup=build_active_reminders_keyboard(reminders, page=0),
                )
            else:
                await _safe_edit_message(
                    query,
                    _format_instant_list_message(reminders),
                    reply_markup=build_instant_reminder_list_keyboard(reminders, page=0),
                )
            return

        await _safe_answer(query)
        await _safe_edit_message(
            query,
            _format_overview_message(reminder, today),
            reply_markup=build_overview_keyboard(reminder.id, source=source),
        )
        return

    if len(parts) == 3 and parts[0] == "instant_send":
        reminder_id = _parse_positive_int(parts[1])
        source = parts[2] if parts[2] in {"active", "instant"} else "instant"
        if reminder_id is None:
            await _safe_answer(query, "Invalid reminder reference.", show_alert=True)
            return
        reminder = await db.get_reminder(reminder_id)
        if reminder is None:
            await _safe_answer(query, "Reminder not found.", show_alert=True)
            return

        target_channel = _get_channel_id(context)
        try:
            await post_reminder_to_channel(
                bot=context.bot,
                channel_id=target_channel,
                reminder=reminder,
                db=db,
                today=today,
            )
        except Exception as exc:
            logger.exception("Instant reminder dispatch failed for reminder #%d", reminder.id)
            await _safe_answer(query, f"❌ Failed to send: {exc}", show_alert=True)
            return

        refreshed = await db.get_reminder(reminder.id) or reminder
        await _safe_answer(query, "🚀 Instant reminder posted to channel!", show_alert=True)
        await _safe_edit_message(
            query,
            _format_overview_message(
                refreshed,
                today,
                banner="✅ Instant reminder sent to the channel!",
            ),
            reply_markup=build_overview_keyboard(refreshed.id, source=source),
        )
        return

    # --- Edit Settings for Existing Reminder ---
    if len(parts) == 3 and parts[0] == "edit_win":
        reminder_id = _parse_positive_int(parts[1])
        source = parts[2] if parts[2] in {"active", "instant"} else "instant"
        if reminder_id is None:
            await _safe_answer(query, "Invalid reminder reference.", show_alert=True)
            return
        reminder = await db.get_reminder(reminder_id)
        if reminder is None:
            await _safe_answer(query, "Reminder not found.", show_alert=True)
            return
        await _safe_answer(query)
        await _safe_edit_message(
            query,
            "⚙️ <b>Edit Reminder Settings — Step 1 of 2</b>\n\n"
            f"📌 <b>Event:</b> {html.escape(reminder.title)}\n"
            f"🪟 <b>Current Window:</b> {html.escape(reminder.window_label)}\n"
            f"🔁 <b>Current Mode:</b> {html.escape(reminder.repeat_mode_label)}\n\n"
            "Select the new reminder window:",
            reply_markup=build_window_selection_keyboard(
                reminder.id,
                prefix="edit_mode",
                source=source,
                current_window=reminder.window_days,
                back_callback=f"overview:{reminder.id}:{source}",
            ),
        )
        return

    if len(parts) == 4 and parts[0] == "edit_mode":
        reminder_id = _parse_positive_int(parts[1])
        window_days = _parse_positive_int(parts[2])
        source = parts[3] if parts[3] in {"active", "instant"} else "instant"
        if reminder_id is None or window_days not in WINDOW_OPTIONS:
            await _safe_answer(query, "Invalid window selection.", show_alert=True)
            return
        reminder = await db.get_reminder(reminder_id)
        if reminder is None:
            await _safe_answer(query, "Reminder not found.", show_alert=True)
            return
        await _safe_answer(query)
        suffix = "Day" if window_days == 1 else "Days"
        await _safe_edit_message(
            query,
            "⚙️ <b>Edit Reminder Settings — Step 2 of 2</b>\n\n"
            f"📌 <b>Event:</b> {html.escape(reminder.title)}\n"
            f"🪟 <b>New Window:</b> Last {window_days} {suffix}\n\n"
            "Select the repeat mode:",
            reply_markup=build_repeat_mode_keyboard(
                reminder.id,
                window_days,
                prefix="edit_save",
                source=source,
                current_mode=reminder.repeat_mode,
                back_callback=f"edit_win:{reminder.id}:{source}",
            ),
        )
        return

    if len(parts) == 5 and parts[0] == "edit_save":
        reminder_id = _parse_positive_int(parts[1])
        window_days = _parse_positive_int(parts[2])
        repeat_mode = parts[3]
        source = parts[4] if parts[4] in {"active", "instant"} else "instant"
        if (
            reminder_id is None
            or window_days not in WINDOW_OPTIONS
            or repeat_mode not in {"normal", "alternate"}
        ):
            await _safe_answer(query, "Invalid settings.", show_alert=True)
            return

        updated = await db.update_reminder_settings(
            reminder_id,
            window_days=window_days,
            repeat_mode=repeat_mode,
            activate=False,
        )
        if updated is None:
            await _safe_answer(query, "Reminder not found.", show_alert=True)
            return

        await _safe_answer(query, "✅ Reminder settings updated!")
        await _safe_edit_message(
            query,
            _format_overview_message(
                updated,
                today,
                banner="✅ Reminder settings updated successfully!",
            ),
            reply_markup=build_overview_keyboard(updated.id, source=source),
        )
        return

    # --- Delete Reminder Confirmation ---
    if len(parts) in {3, 4} and parts[0] == "del_ask":
        reminder_id = _parse_positive_int(parts[1])
        source = parts[2] if parts[2] in {"active", "instant"} else "active"
        origin = parts[3] if len(parts) == 4 and parts[3] in {"list", "overview"} else "overview"
        if reminder_id is None:
            await _safe_answer(query, "Invalid reminder reference.", show_alert=True)
            return
        reminder = await db.get_reminder(reminder_id)
        if reminder is None:
            await _safe_answer(query, "Reminder already deleted.", show_alert=True)
            return
        await _safe_answer(query)
        await _safe_edit_message(
            query,
            "⚠️ <b>Confirm Deletion</b>\n\n"
            "Are you sure you want to permanently delete this reminder?\n\n"
            f"📌 <b>Title:</b> {html.escape(reminder.title)}\n"
            f"⏳ <b>End Date:</b> {html.escape(reminder.end_date)}",
            reply_markup=build_delete_confirm_keyboard(
                reminder.id,
                source=source,
                origin=origin,
            ),
        )
        return

    if len(parts) == 3 and parts[0] == "del_yes":
        reminder_id = _parse_positive_int(parts[1])
        source = parts[2] if parts[2] in {"active", "instant"} else "active"
        if reminder_id is None:
            await _safe_answer(query, "Invalid reminder reference.", show_alert=True)
            return
        deleted = await db.delete_reminder(reminder_id)
        await _safe_answer(
            query,
            "🗑 Reminder deleted." if deleted else "Reminder was already removed.",
        )
        reminders = await db.list_active_reminders()
        if source == "instant":
            await _safe_edit_message(
                query,
                _format_instant_list_message(reminders),
                reply_markup=build_instant_reminder_list_keyboard(reminders, page=0),
            )
        else:
            await _safe_edit_message(
                query,
                _format_active_list_message(reminders, today),
                reply_markup=build_active_reminders_keyboard(reminders, page=0),
            )
        return

    # --- Manual Post Entry Callbacks ---
    if data == "manual:cancel":
        await _safe_answer(query, "Cancelled.")
        context.user_data.pop("manual_flow", None)
        await _safe_edit_message(
            query,
            await _render_main_menu_text(context),
            reply_markup=build_main_menu_keyboard(),
        )
        return

    if data == "manual:skip_text":
        flow = context.user_data.get("manual_flow")
        if not flow:
            await _safe_answer(
                query,
                "Session expired. Start again from the Admin Panel.",
                show_alert=True,
            )
            return
        await _safe_answer(query)
        flow["post_text"] = flow.get("post_text") or ""
        flow["step"] = "awaiting_end_date"
        await _safe_edit_message(
            query,
            "⏳ <b>Enter Event End Date</b>\n\n"
            "Type the deadline date in any of these accepted formats:\n"
            "• <code>YYYY-MM-DD</code> (e.g. <code>2026-12-31</code>)\n"
            "• <code>DD-MM-YYYY</code> (e.g. <code>31-12-2026</code>)\n"
            "• <code>DD Month YYYY</code> (e.g. <code>31 December 2026</code>)",
            reply_markup=build_manual_cancel_keyboard(),
        )
        return

    if data == "manual:choose_window":
        flow = context.user_data.get("manual_flow")
        if not flow or not flow.get("end_date"):
            await _safe_answer(
                query,
                "Session expired. Start again from the Admin Panel.",
                show_alert=True,
            )
            return
        await _safe_answer(query)
        flow["step"] = "configuring"
        await _safe_edit_message(
            query,
            _format_manual_overview(flow, today) + "\n\n🪟 <b>Select Reminder Window:</b>",
            reply_markup=build_manual_window_keyboard(flow.get("window_days")),
        )
        return

    if len(parts) == 2 and parts[0] == "manual_win":
        flow = context.user_data.get("manual_flow")
        if not flow or not flow.get("end_date"):
            await _safe_answer(
                query,
                "Session expired. Start again from the Admin Panel.",
                show_alert=True,
            )
            return
        window_days = _parse_positive_int(parts[1])
        if window_days not in WINDOW_OPTIONS:
            await _safe_answer(query, "Invalid window.", show_alert=True)
            return
        await _safe_answer(query)
        flow["window_days"] = window_days
        flow["step"] = "configuring"
        await _safe_edit_message(
            query,
            _format_manual_overview(flow, today)
            + "\n\n🔁 <b>Select Repeat Mode:</b>\n"
            "• <b>Normal (Daily):</b> 1 reminder every day inside the window.\n"
            "• <b>Alternate Day:</b> 1 reminder every 2 days from the start of the window.",
            reply_markup=build_manual_mode_keyboard(flow.get("repeat_mode")),
        )
        return

    if len(parts) == 2 and parts[0] == "manual_mode":
        flow = context.user_data.get("manual_flow")
        if not flow or not flow.get("end_date") or not flow.get("window_days"):
            await _safe_answer(
                query,
                "Session expired. Start again from the Admin Panel.",
                show_alert=True,
            )
            return
        repeat_mode = parts[1]
        if repeat_mode not in {"normal", "alternate"}:
            await _safe_answer(query, "Invalid repeat mode.", show_alert=True)
            return
        await _safe_answer(query)
        flow["repeat_mode"] = repeat_mode
        flow["step"] = "ready_to_confirm"
        await _safe_edit_message(
            query,
            _format_manual_overview(flow, today)
            + "\n\nReview the details above and click <b>Confirm & Activate</b> to save.",
            reply_markup=build_manual_final_confirm_keyboard(),
        )
        return

    if data == "manual:confirm":
        # Pop immediately to prevent duplicate inserts from rapid double-clicks
        flow = context.user_data.pop("manual_flow", None)
        if (
            not flow
            or not flow.get("end_date")
            or not flow.get("title")
            or not flow.get("window_days")
            or not flow.get("repeat_mode")
        ):
            await _safe_answer(
                query,
                "Session incomplete or already confirmed.",
                show_alert=True,
            )
            return

        target_channel_str = flow.get("channel_id") or str(_get_channel_id(context))
        if await db.has_open_reminder_for_post(
            target_channel_str,
            int(flow["message_id"]),
            post_link=flow["post_link"],
        ):
            await _safe_answer(
                query,
                "An active or pending reminder already exists for this post.",
                show_alert=True,
            )
            await _safe_edit_message(
                query,
                "⚠️ <b>Duplicate Reminder Prevented</b>\n\n"
                "An open reminder already exists for that Telegram post link.\n"
                "Manage or edit it from <b>Active Reminders</b>.",
                reply_markup=build_main_menu_keyboard(),
            )
            return

        try:
            reminder = await db.create_reminder(
                channel_id=target_channel_str,
                message_id=int(flow["message_id"]),
                post_link=flow["post_link"],
                title=flow["title"],
                post_text=flow.get("post_text") or flow["title"],
                start_date=flow.get("start_date"),
                end_date=flow["end_date"],
                window_days=int(flow["window_days"]),
                repeat_mode=flow["repeat_mode"],
                status="active",
            )
        except Exception as exc:
            # Restore session state on transient DB failure so the admin can retry
            context.user_data["manual_flow"] = flow
            logger.exception("Failed to save manual reminder to database")
            await _safe_answer(query, f"❌ Database error: {exc}", show_alert=True)
            return

        await _safe_answer(query, "✅ Reminder saved and activated!")
        safe_link = html.escape(reminder.post_link, quote=True)
        await _safe_edit_message(
            query,
            "✅ <b>Manual Reminder Saved & Activated</b>\n\n"
            f"📌 <b>Event Title:</b> {html.escape(reminder.title)}\n"
            f"🗓 <b>Start Date:</b> {html.escape(reminder.start_date or 'N/A')}\n"
            f"⏳ <b>End Date:</b> {html.escape(reminder.end_date)}\n"
            f"📆 <b>Remaining Days:</b> {html.escape(_format_days_left_label(reminder.days_left(today)))}\n"
            f"🪟 <b>Window:</b> {html.escape(reminder.window_label)}\n"
            f"🔁 <b>Repeat Mode:</b> {html.escape(reminder.repeat_mode_label)}\n"
            f'🔗 <b>Original Post:</b> <a href="{safe_link}">Open</a>',
            reply_markup=build_post_activation_keyboard(reminder.id),
        )
        return

    await _safe_answer(query, "Unknown or expired action.", show_alert=True)


# ======================================================================
# C.3. Manual Link Entry Private Message Handler
# ======================================================================


async def _run_manual_gemini_and_advance(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    flow: dict[str, Any],
    post_text: str,
) -> None:
    """Analyze text with Gemini during manual flow and advance to fallback or overview."""
    message = update.effective_message
    assert message is not None

    db = _get_db(context)
    parser = _get_parser(context)
    today = db.today()

    flow["post_text"] = post_text

    extraction = None
    try:
        extraction = await parser.extract_deadline(post_text=post_text, current_date=today)
    except Exception:
        logger.exception("Gemini extraction error during manual post entry")

    if extraction is not None:
        if extraction.title:
            flow["title"] = extraction.title
        if extraction.start_date:
            flow["start_date"] = extraction.start_date
        if extraction.end_date and date.fromisoformat(extraction.end_date) >= today:
            flow["end_date"] = extraction.end_date

    # Missing End Date Fallback
    if not flow.get("end_date"):
        flow["step"] = "awaiting_end_date"
        detected_title_note = (
            f"📌 <b>Detected Title:</b> {html.escape(flow['title'])}\n\n"
            if flow.get("title")
            else ""
        )
        await message.reply_text(
            "⚠️ <b>Could not automatically detect an upcoming End Date</b>\n\n"
            f"{detected_title_note}"
            "Please type the <b>End Date</b> directly using one of these formats:\n"
            "• <code>YYYY-MM-DD</code> (e.g. <code>2026-12-31</code>)\n"
            "• <code>DD-MM-YYYY</code> (e.g. <code>31-12-2026</code>)\n"
            "• <code>DD Month YYYY</code> (e.g. <code>31 December 2026</code>)",
            parse_mode=ParseMode.HTML,
            reply_markup=build_manual_cancel_keyboard(),
        )
        return

    # Missing Title Fallback
    if not flow.get("title"):
        flow["step"] = "awaiting_title"
        await message.reply_text(
            f"⏳ <b>Detected End Date:</b> <code>{html.escape(flow['end_date'])}</code>\n\n"
            "No concise title could be extracted. Please type the <b>Event Title</b>:",
            parse_mode=ParseMode.HTML,
            reply_markup=build_manual_cancel_keyboard(),
        )
        return

    # Both End Date and Title are ready -> show Event Overview & Window selection
    flow["step"] = "configuring"
    await message.reply_text(
        _format_manual_overview(flow, today)
        + "\n\n🪟 <b>Step 1 of 2 — Select Reminder Window:</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=build_manual_window_keyboard(),
        disable_web_page_preview=True,
    )


async def handle_private_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process admin text/forwarded input during the Set Reminder Post flow."""
    message = update.effective_message
    if message is None or not _is_private_admin(update, context):
        return

    flow: Optional[dict[str, Any]] = context.user_data.get("manual_flow")
    if not flow:
        await message.reply_text("Use /admin or /start to open the Admin Control Panel.")
        return

    step = flow.get("step")
    db = _get_db(context)
    today = db.today()

    # Step 1: Receive Telegram channel post link
    if step == "awaiting_link":
        raw_text = (message.text or message.caption or "").strip()
        parsed = parse_telegram_post_link(raw_text)
        if parsed is None:
            await message.reply_text(
                "❌ <b>Invalid Telegram post link.</b>\n\n"
                "Please send a valid channel post URL, for example:\n"
                "• <code>https://t.me/my_channel/123</code>\n"
                "• <code>https://t.me/c/1234567890/123</code>",
                parse_mode=ParseMode.HTML,
                reply_markup=build_manual_cancel_keyboard(),
            )
            return

        target_channel_str = parsed.channel_id or str(_get_channel_id(context))
        if await db.has_open_reminder_for_post(
            target_channel_str,
            parsed.message_id,
            post_link=parsed.post_link,
        ):
            context.user_data.pop("manual_flow", None)
            await message.reply_text(
                "⚠️ <b>Reminder Already Exists</b>\n\n"
                "That post already has an active or pending reminder. "
                "You can manage or edit it from <b>Active Reminders</b>.",
                parse_mode=ParseMode.HTML,
                reply_markup=build_main_menu_keyboard(),
            )
            return

        flow["post_link"] = parsed.post_link
        flow["message_id"] = parsed.message_id
        flow["channel_id"] = target_channel_str
        flow["channel_username"] = parsed.channel_username
        flow["internal_channel_id"] = parsed.internal_channel_id

        cached = await db.get_cached_post(
            message_id=parsed.message_id,
            channel_id=parsed.channel_id,
            channel_username=parsed.channel_username,
            internal_channel_id=parsed.internal_channel_id,
        )

        if cached is not None:
            flow["channel_id"] = cached.channel_id
            await message.reply_text(
                "📦 Found this post in the cache! Extracting event details with Gemini..."
            )
            await _run_manual_gemini_and_advance(update, context, flow, cached.post_text)
            return

        flow["step"] = "awaiting_post_text"
        await message.reply_text(
            "ℹ️ <b>Post not found in cache</b>\n\n"
            "Because this bot uses the Telegram Bot API (bot token only), it cannot fetch "
            "older channel posts published before the bot saw them.\n\n"
            "Please <b>forward the channel post</b> here or <b>paste the post text</b> below "
            "so Gemini can extract the title and deadline — or click the button below to "
            "enter the title and date manually.",
            parse_mode=ParseMode.HTML,
            reply_markup=build_manual_cache_miss_keyboard(),
        )
        return

    # Step 2: Receive forwarded post or pasted text on cache miss
    if step == "awaiting_post_text":
        post_text = (message.text or message.caption or "").strip()
        if not post_text:
            await message.reply_text(
                "⚠️ That message has no text or caption. Please forward a text/captioned post, "
                "paste the text, or click <b>Skip & Enter Title/Date Manually</b>.",
                parse_mode=ParseMode.HTML,
                reply_markup=build_manual_cache_miss_keyboard(),
            )
            return

        await db.cache_channel_post(
            channel_id=flow["channel_id"],
            channel_username=flow.get("channel_username"),
            message_id=int(flow["message_id"]),
            post_text=post_text,
            post_link=flow["post_link"],
        )

        await message.reply_text("🤖 Analyzing post text with Gemini...")
        await _run_manual_gemini_and_advance(update, context, flow, post_text)
        return

    # Step 3a: Manual End Date fallback
    if step == "awaiting_end_date":
        raw_date = (message.text or "").strip()
        parsed_date = parse_user_date(raw_date)
        if parsed_date is None:
            await message.reply_text(
                "❌ <b>Could not parse that date.</b>\n\n"
                "Please enter the <b>End Date</b> in one of these accepted formats:\n"
                "• <code>YYYY-MM-DD</code> (e.g. <code>2026-12-31</code>)\n"
                "• <code>DD-MM-YYYY</code> (e.g. <code>31-12-2026</code>)\n"
                "• <code>DD Month YYYY</code> (e.g. <code>31 December 2026</code>)",
                parse_mode=ParseMode.HTML,
                reply_markup=build_manual_cancel_keyboard(),
            )
            return

        if parsed_date < today:
            await message.reply_text(
                f"❌ <b>Past date rejected:</b> <code>{parsed_date.isoformat()}</code> is before "
                f"today (<code>{today.isoformat()}</code>).\n\n"
                "Please enter today or a future End Date:",
                parse_mode=ParseMode.HTML,
                reply_markup=build_manual_cancel_keyboard(),
            )
            return

        flow["end_date"] = parsed_date.isoformat()

        if not flow.get("title"):
            flow["step"] = "awaiting_title"
            await message.reply_text(
                f"✅ End Date set to <code>{html.escape(flow['end_date'])}</code>.\n\n"
                "Now please type a concise <b>Event Title</b>:",
                parse_mode=ParseMode.HTML,
                reply_markup=build_manual_cancel_keyboard(),
            )
            return

        flow["step"] = "configuring"
        await message.reply_text(
            _format_manual_overview(flow, today)
            + "\n\n🪟 <b>Step 1 of 2 — Select Reminder Window:</b>",
            parse_mode=ParseMode.HTML,
            reply_markup=build_manual_window_keyboard(),
            disable_web_page_preview=True,
        )
        return

    # Step 3b: Manual Title fallback
    if step == "awaiting_title":
        title = " ".join((message.text or "").strip().split())
        if not title:
            await message.reply_text(
                "⚠️ Title cannot be empty. Please type a concise <b>Event Title</b>:",
                parse_mode=ParseMode.HTML,
                reply_markup=build_manual_cancel_keyboard(),
            )
            return

        flow["title"] = title[:120]
        flow["step"] = "configuring"
        await message.reply_text(
            _format_manual_overview(flow, today)
            + "\n\n🪟 <b>Step 1 of 2 — Select Reminder Window:</b>",
            parse_mode=ParseMode.HTML,
            reply_markup=build_manual_window_keyboard(),
            disable_web_page_preview=True,
        )
        return

    await message.reply_text(
        "Please use the inline buttons above to complete your selection, or send /cancel to exit."
    )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global error handler for unexpected Telegram or handler exceptions."""
    logger.exception("Unhandled exception while processing update %s", update, exc_info=context.error)


def register_handlers(application: Application) -> None:
    """Attach access gatekeeper, channel monitors, admin commands, callbacks, and error handler."""
    # Group -1: Ironclad private admin-only gatekeeper (runs BEFORE all handlers)
    application.add_handler(TypeHandler(Update, enforce_admin_access), group=-1)

    # Group 0: Channel monitoring
    application.add_handler(
        MessageHandler(filters.UpdateType.CHANNEL_POST, handle_channel_post)
    )
    application.add_handler(
        MessageHandler(filters.UpdateType.EDITED_CHANNEL_POST, handle_edited_channel_post)
    )

    # Group 0: Admin Control Panel commands (private chat only)
    application.add_handler(
        CommandHandler(["start", "admin"], cmd_admin_menu, filters=filters.ChatType.PRIVATE)
    )
    application.add_handler(
        CommandHandler("cancel", cmd_cancel, filters=filters.ChatType.PRIVATE)
    )
    application.add_handler(
        MessageHandler(filters.ChatType.PRIVATE & filters.COMMAND, cmd_unknown_admin)
    )

    # Group 0: Inline button callbacks
    application.add_handler(CallbackQueryHandler(handle_callback_query))

    # Group 0: Manual link / text / date / title input in private chat
    application.add_handler(
        MessageHandler(
            filters.ChatType.PRIVATE & ~filters.COMMAND,
            handle_private_message,
        )
    )

    application.add_error_handler(error_handler)
