"""Inline keyboard builders for the Admin Control Panel and confirmation flows."""

from __future__ import annotations

from typing import Optional, Sequence

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from database import Reminder

WINDOW_OPTIONS: tuple[int, ...] = (1, 2, 3, 5, 7, 10)
ITEMS_PER_PAGE = 8


def get_allowed_window_options(max_days: Optional[int]) -> tuple[int, ...]:
    """Return valid reminder window day options capped by max_days (always >= 1)."""
    if max_days is None:
        return WINDOW_OPTIONS
    cap = max(1, int(max_days))
    allowed = tuple(days for days in WINDOW_OPTIONS if days <= cap)
    return allowed or (1,)


def _truncate(text: str, max_len: int = 34) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_len:
        return cleaned
    return f"{cleaned[: max_len - 1].rstrip()}…"


def build_main_menu_keyboard() -> InlineKeyboardMarkup:
    """Main Admin Control Panel menu."""
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("📋 Active Reminders", callback_data="menu:active:0")],
            [InlineKeyboardButton("⚡ Instant Reminder", callback_data="menu:instant:0")],
            [InlineKeyboardButton("🔗 Set Reminder Post", callback_data="menu:set_post")],
        ]
    )


def build_detection_confirm_keyboard(reminder_id: int) -> InlineKeyboardMarkup:
    """Initial Confirm / Decline buttons when a new deadline post is detected."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("✅ Confirm", callback_data=f"confirm:{reminder_id}"),
                InlineKeyboardButton("❌ Decline", callback_data=f"decline:{reminder_id}"),
            ]
        ]
    )


def build_window_selection_keyboard(
    reminder_id: int,
    *,
    prefix: str = "win",
    source: Optional[str] = None,
    current_window: Optional[int] = None,
    back_callback: Optional[str] = None,
    max_days: Optional[int] = None,
) -> InlineKeyboardMarkup:
    """Window selection keyboard dynamically capped to <= max_days (1, 2, 3, 5, 7, 10 Days)."""
    rows: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []

    for days in get_allowed_window_options(max_days):
        suffix = "Day" if days == 1 else "Days"
        label = f"{days} {suffix}"
        if current_window == days:
            label = f"✓ {label}"

        if source is not None:
            cb = f"{prefix}:{reminder_id}:{days}:{source}"
        else:
            cb = f"{prefix}:{reminder_id}:{days}"

        row.append(InlineKeyboardButton(label, callback_data=cb))
        if len(row) == 3:
            rows.append(row)
            row = []

    if row:
        rows.append(row)

    if back_callback:
        rows.append([InlineKeyboardButton("⬅️ Back", callback_data=back_callback)])

    return InlineKeyboardMarkup(rows)


def build_repeat_mode_keyboard(
    reminder_id: int,
    window_days: int,
    *,
    prefix: str = "mode",
    source: Optional[str] = None,
    current_mode: Optional[str] = None,
    back_callback: Optional[str] = None,
) -> InlineKeyboardMarkup:
    """Repeat mode selection keyboard (Normal Daily vs Alternate Day)."""
    normal_label = "📅 Normal (Daily)"
    alt_label = "📆 Alternate Day"
    if current_mode == "normal":
        normal_label = f"✓ {normal_label}"
    elif current_mode == "alternate":
        alt_label = f"✓ {alt_label}"

    if source is not None:
        normal_cb = f"{prefix}:{reminder_id}:{window_days}:normal:{source}"
        alt_cb = f"{prefix}:{reminder_id}:{window_days}:alternate:{source}"
    else:
        normal_cb = f"{prefix}:{reminder_id}:{window_days}:normal"
        alt_cb = f"{prefix}:{reminder_id}:{window_days}:alternate"

    rows = [
        [InlineKeyboardButton(normal_label, callback_data=normal_cb)],
        [InlineKeyboardButton(alt_label, callback_data=alt_cb)],
    ]
    if back_callback:
        rows.append([InlineKeyboardButton("⬅️ Back", callback_data=back_callback)])
    return InlineKeyboardMarkup(rows)


def build_post_activation_keyboard(reminder_id: int) -> InlineKeyboardMarkup:
    """Navigation buttons shown after a reminder is confirmed and activated."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "📄 View Overview",
                    callback_data=f"overview:{reminder_id}:active",
                )
            ],
            [InlineKeyboardButton("🏠 Admin Panel", callback_data="menu:home")],
        ]
    )


def build_active_reminders_keyboard(
    reminders: Sequence[Reminder],
    page: int = 0,
) -> InlineKeyboardMarkup:
    """List active reminders with Overview and Delete actions + pagination."""
    total = len(reminders)
    total_pages = max(1, (total + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
    page = max(0, min(page, total_pages - 1))
    start = page * ITEMS_PER_PAGE
    chunk = reminders[start : start + ITEMS_PER_PAGE]

    rows: list[list[InlineKeyboardButton]] = []
    for rem in chunk:
        rows.append(
            [
                InlineKeyboardButton(
                    f"📌 {_truncate(rem.title, 28)}",
                    callback_data=f"overview:{rem.id}:active",
                ),
                InlineKeyboardButton(
                    "🗑 Delete",
                    callback_data=f"del_ask:{rem.id}:active:list",
                ),
            ]
        )

    if total_pages > 1:
        nav_row: list[InlineKeyboardButton] = []
        if page > 0:
            nav_row.append(
                InlineKeyboardButton("◀️ Prev", callback_data=f"menu:active:{page - 1}")
            )
        nav_row.append(
            InlineKeyboardButton(f"{page + 1}/{total_pages}", callback_data="noop")
        )
        if page < total_pages - 1:
            nav_row.append(
                InlineKeyboardButton("Next ▶️", callback_data=f"menu:active:{page + 1}")
            )
        rows.append(nav_row)

    rows.append(
        [
            InlineKeyboardButton("🔄 Refresh", callback_data=f"menu:active:{page}"),
            InlineKeyboardButton("🏠 Main Menu", callback_data="menu:home"),
        ]
    )
    return InlineKeyboardMarkup(rows)


def build_instant_reminder_list_keyboard(
    reminders: Sequence[Reminder],
    page: int = 0,
) -> InlineKeyboardMarkup:
    """List active reminders by Gemini-generated title for Instant Reminder overview."""
    total = len(reminders)
    total_pages = max(1, (total + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
    page = max(0, min(page, total_pages - 1))
    start = page * ITEMS_PER_PAGE
    chunk = reminders[start : start + ITEMS_PER_PAGE]

    rows: list[list[InlineKeyboardButton]] = []
    for rem in chunk:
        rows.append(
            [
                InlineKeyboardButton(
                    f"⚡ {_truncate(rem.title, 44)}",
                    callback_data=f"overview:{rem.id}:instant",
                )
            ]
        )

    if total_pages > 1:
        nav_row: list[InlineKeyboardButton] = []
        if page > 0:
            nav_row.append(
                InlineKeyboardButton("◀️ Prev", callback_data=f"menu:instant:{page - 1}")
            )
        nav_row.append(
            InlineKeyboardButton(f"{page + 1}/{total_pages}", callback_data="noop")
        )
        if page < total_pages - 1:
            nav_row.append(
                InlineKeyboardButton("Next ▶️", callback_data=f"menu:instant:{page + 1}")
            )
        rows.append(nav_row)

    rows.append(
        [
            InlineKeyboardButton("🔄 Refresh", callback_data=f"menu:instant:{page}"),
            InlineKeyboardButton("🏠 Main Menu", callback_data="menu:home"),
        ]
    )
    return InlineKeyboardMarkup(rows)


def build_overview_keyboard(reminder_id: int, source: str = "instant") -> InlineKeyboardMarkup:
    """Full Event Overview action buttons."""
    back_cb = "menu:active:0" if source == "active" else "menu:instant:0"
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "🚀 Send Instant Reminder",
                    callback_data=f"instant_send:{reminder_id}:{source}",
                )
            ],
            [
                InlineKeyboardButton(
                    "⚙️ Edit Settings",
                    callback_data=f"edit_win:{reminder_id}:{source}",
                ),
                InlineKeyboardButton(
                    "🗑 Delete Reminder",
                    callback_data=f"del_ask:{reminder_id}:{source}:overview",
                ),
            ],
            [
                InlineKeyboardButton("⬅️ Back to List", callback_data=back_cb),
                InlineKeyboardButton("🏠 Main Menu", callback_data="menu:home"),
            ],
        ]
    )


def build_delete_confirm_keyboard(
    reminder_id: int,
    source: str = "active",
    origin: str = "overview",
) -> InlineKeyboardMarkup:
    """Confirmation dialog before permanently deleting a reminder."""
    if origin == "list":
        cancel_cb = "menu:active:0" if source == "active" else "menu:instant:0"
    else:
        cancel_cb = f"overview:{reminder_id}:{source}"

    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✅ Yes, Delete Permanently",
                    callback_data=f"del_yes:{reminder_id}:{source}",
                )
            ],
            [InlineKeyboardButton("↩️ Cancel", callback_data=cancel_cb)],
        ]
    )


def build_manual_cancel_keyboard() -> InlineKeyboardMarkup:
    """Cancel button for manual post entry steps."""
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("❌ Cancel", callback_data="manual:cancel")]]
    )


def build_manual_cache_miss_keyboard() -> InlineKeyboardMarkup:
    """Options when a manually linked post is not in the local cache."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✍️ Skip & Enter Title/Date Manually",
                    callback_data="manual:skip_text",
                )
            ],
            [InlineKeyboardButton("❌ Cancel", callback_data="manual:cancel")],
        ]
    )


def build_manual_window_keyboard(
    current_window: Optional[int] = None,
    *,
    max_days: Optional[int] = None,
) -> InlineKeyboardMarkup:
    """Select reminder window during manual post setup, capped to <= max_days."""
    rows: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for days in get_allowed_window_options(max_days):
        suffix = "Day" if days == 1 else "Days"
        label = f"{days} {suffix}"
        if current_window == days:
            label = f"✓ {label}"
        row.append(InlineKeyboardButton(label, callback_data=f"manual_win:{days}"))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("❌ Cancel", callback_data="manual:cancel")])
    return InlineKeyboardMarkup(rows)


def build_manual_mode_keyboard(current_mode: Optional[str] = None) -> InlineKeyboardMarkup:
    """Select repeat mode during manual post setup."""
    normal_label = "📅 Normal (Daily)"
    alt_label = "📆 Alternate Day"
    if current_mode == "normal":
        normal_label = f"✓ {normal_label}"
    elif current_mode == "alternate":
        alt_label = f"✓ {alt_label}"
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(normal_label, callback_data="manual_mode:normal")],
            [InlineKeyboardButton(alt_label, callback_data="manual_mode:alternate")],
            [InlineKeyboardButton("⬅️ Change Window", callback_data="manual:choose_window")],
            [InlineKeyboardButton("❌ Cancel", callback_data="manual:cancel")],
        ]
    )


def build_manual_final_confirm_keyboard() -> InlineKeyboardMarkup:
    """Final confirmation buttons for a manually entered reminder."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✅ Confirm & Activate",
                    callback_data="manual:confirm",
                )
            ],
            [
                InlineKeyboardButton(
                    "⚙️ Change Window / Mode",
                    callback_data="manual:choose_window",
                )
            ],
            [InlineKeyboardButton("❌ Cancel", callback_data="manual:cancel")],
        ]
    )
