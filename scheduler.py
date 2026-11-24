"""APScheduler-based automated reminder dispatcher, channel formatter, and admin alert notifier."""

from __future__ import annotations

import asyncio
import html
import logging
import time as monotonic_time
from datetime import date, datetime, time, timedelta
from typing import Optional, Union

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import BadRequest, RetryAfter

from database import Database, Reminder, resolve_timezone

logger = logging.getLogger(__name__)

_MAX_ADMIN_ALERT_LEN = 3900


def _truncate_text(text: str, limit: int) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: max(1, limit - 15)].rstrip()}... [truncated]"


class AdminNotifier:
    """Rate-limited, loop-safe instant private error & status alerts to ADMIN_USER_ID."""

    def __init__(
        self,
        *,
        bot: Bot,
        admin_user_id: int,
        cooldown_seconds: float = 30.0,
        max_per_minute: int = 15,
    ) -> None:
        self.bot = bot
        self.admin_user_id = admin_user_id
        self.cooldown_seconds = cooldown_seconds
        self.max_per_minute = max_per_minute
        self._recent_keys: dict[str, float] = {}
        self._send_timestamps: list[float] = []
        self._sending = False

    def _allow_send(self, dedupe_key: str) -> bool:
        now = monotonic_time.monotonic()
        # Evict timestamps older than 60s
        self._send_timestamps = [ts for ts in self._send_timestamps if now - ts < 60.0]
        if len(self._send_timestamps) >= self.max_per_minute:
            logger.warning(
                "AdminNotifier rate limit reached (%d/min); suppressing alert '%s'",
                self.max_per_minute,
                dedupe_key,
            )
            return False

        # Evict stale dedupe entries
        stale_cutoff = now - max(self.cooldown_seconds * 4.0, 120.0)
        self._recent_keys = {k: ts for k, ts in self._recent_keys.items() if ts >= stale_cutoff}

        last_sent = self._recent_keys.get(dedupe_key)
        if last_sent is not None and (now - last_sent) < self.cooldown_seconds:
            logger.debug("Suppressing duplicate admin alert within cooldown: %s", dedupe_key)
            return False

        self._recent_keys[dedupe_key] = now
        self._send_timestamps.append(now)
        return True

    async def _send_safe(self, html_text: str, plain_fallback: str) -> bool:
        if self._sending:
            logger.warning("Re-entrant AdminNotifier call suppressed to prevent infinite loop")
            return False

        self._sending = True
        try:
            try:
                await self.bot.send_message(
                    chat_id=self.admin_user_id,
                    text=html_text[:4096],
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )
                return True
            except BadRequest:
                # Fallback to plain text if HTML formatting ever fails
                await self.bot.send_message(
                    chat_id=self.admin_user_id,
                    text=plain_fallback[:4096],
                    disable_web_page_preview=True,
                )
                return True
        except Exception:
            logger.exception("AdminNotifier failed to deliver alert to admin %d", self.admin_user_id)
            return False
        finally:
            self._sending = False

    async def notify_error(
        self,
        operation: str,
        error: Union[BaseException, str],
        *,
        context_info: Optional[str] = None,
        dedupe_key: Optional[str] = None,
    ) -> bool:
        """Immediately send an HTML-escaped failure alert to ADMIN_USER_ID."""
        if isinstance(error, BaseException):
            raw_reason = f"{type(error).__name__}: {error}"
        else:
            raw_reason = str(error)

        reason_short = _truncate_text(raw_reason, 1800)
        key = dedupe_key or f"err:{operation}:{reason_short[:120]}"
        if not self._allow_send(key):
            return False

        safe_op = html.escape(_truncate_text(operation, 120))
        safe_reason = html.escape(reason_short)
        ctx_block = ""
        plain_ctx = ""
        if context_info:
            trimmed_ctx = _truncate_text(context_info, 500)
            ctx_block = f"📍 <b>Context:</b> {html.escape(trimmed_ctx)}\n"
            plain_ctx = f"Context: {trimmed_ctx}\n"

        html_msg = (
            "🚨 <b>Bot Error Alert</b>\n\n"
            f"⚙️ <b>Operation:</b> {safe_op}\n"
            f"{ctx_block}"
            f"❗ <b>Reason:</b>\n<code>{safe_reason}</code>"
        )
        if len(html_msg) > _MAX_ADMIN_ALERT_LEN:
            trimmed_reason = html.escape(_truncate_text(raw_reason, 800))
            html_msg = (
                "🚨 <b>Bot Error Alert</b>\n\n"
                f"⚙️ <b>Operation:</b> {safe_op}\n"
                f"{ctx_block}"
                f"❗ <b>Reason:</b>\n<code>{trimmed_reason}</code>"
            )

        plain_msg = (
            "Bot Error Alert\n\n"
            f"Operation: {operation}\n"
            f"{plain_ctx}"
            f"Reason: {reason_short}"
        )
        return await self._send_safe(html_msg, plain_msg)

    async def notify_notice(
        self,
        title: str,
        body: str,
        *,
        post_link: Optional[str] = None,
        dedupe_key: Optional[str] = None,
    ) -> bool:
        """Send a brief, HTML-escaped operational status notice to ADMIN_USER_ID."""
        key = dedupe_key or f"notice:{title}:{body[:80]}"
        if not self._allow_send(key):
            return False

        safe_title = html.escape(_truncate_text(title, 160))
        safe_body = html.escape(_truncate_text(body, 2200))
        link_block = ""
        plain_link = ""
        if post_link:
            safe_link = html.escape(post_link.strip(), quote=True)
            link_block = f'\n🔗 <b>Post Link:</b> <a href="{safe_link}">Open Post</a>'
            plain_link = f"\nPost Link: {post_link.strip()}"

        html_msg = f"ℹ️ <b>{safe_title}</b>\n\n{safe_body}{link_block}"
        plain_msg = f"{title}\n\n{body}{plain_link}"
        return await self._send_safe(html_msg, plain_msg)


def format_channel_reminder(title: str, end_date: str, post_link: str) -> str:
    """Format the exact HTML message posted to the Telegram channel."""
    safe_title = html.escape(title.strip())
    safe_end_date = html.escape(end_date.strip())
    safe_link = html.escape(post_link.strip(), quote=True)
    return (
        f"{safe_title}\n"
        f"Event ending {safe_end_date}\n\n"
        f'Event link:- <a href="{safe_link}">Open</a>'
    )


def should_send_scheduled_reminder(reminder: Reminder, today: date) -> bool:
    """Evaluate whether an active reminder is eligible to fire on `today`."""
    if reminder.status != "active":
        return False
    if (
        reminder.window_days is None
        or reminder.window_days <= 0
        or reminder.repeat_mode not in {"normal", "alternate"}
    ):
        return False

    today_iso = today.isoformat()
    if reminder.last_reminded_on == today_iso:
        return False

    days_left = reminder.days_left(today)
    if days_left < 0:
        return False
    if not (1 <= days_left <= reminder.window_days):
        return False

    if reminder.repeat_mode == "normal":
        return True

    # Alternate day: remind every 2 days starting from the start of the selected window
    # e.g., window_days=10 -> days_left in {10, 8, 6, 4, 2}
    return (reminder.window_days - days_left) % 2 == 0


async def _send_channel_html_with_retry(
    *,
    bot: Bot,
    channel_id: Union[int, str],
    text: str,
) -> None:
    """Send HTML message to the channel, retrying once if Telegram requests RetryAfter."""
    try:
        await bot.send_message(
            chat_id=channel_id,
            text=text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
    except RetryAfter as exc:
        wait_seconds = float(exc.retry_after) + 0.5
        logger.warning(
            "Telegram flood control hit for channel %s; waiting %.1fs before retry",
            channel_id,
            wait_seconds,
        )
        await asyncio.sleep(wait_seconds)
        await bot.send_message(
            chat_id=channel_id,
            text=text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )


async def post_reminder_to_channel(
    *,
    bot: Bot,
    channel_id: Union[int, str],
    reminder: Reminder,
    db: Database,
    today: date,
) -> None:
    """Send an instant reminder to the target channel and update last_reminded_on."""
    message_text = format_channel_reminder(
        title=reminder.title,
        end_date=reminder.end_date,
        post_link=reminder.post_link,
    )
    await _send_channel_html_with_retry(
        bot=bot,
        channel_id=channel_id,
        text=message_text,
    )
    await db.update_last_reminded_on(reminder.id, today.isoformat())


class ReminderScheduler:
    """Manages periodic deadline checks and automated channel reminder posting."""

    def __init__(
        self,
        *,
        bot: Bot,
        db: Database,
        target_channel_id: Union[int, str],
        timezone: str = "Asia/Dhaka",
        daily_hour: int = 10,
        daily_minute: int = 0,
        notifier: Optional[AdminNotifier] = None,
    ) -> None:
        self.bot = bot
        self.db = db
        self.target_channel_id = target_channel_id
        self.timezone_name = timezone
        self.tz = resolve_timezone(timezone)
        self.daily_hour = daily_hour
        self.daily_minute = daily_minute
        self.notifier = notifier
        self._lock = asyncio.Lock()
        self.scheduler = AsyncIOScheduler(timezone=self.tz)

    def start(self) -> None:
        """Register scheduled jobs and start AsyncIOScheduler."""
        self.scheduler.add_job(
            self.run_reminder_check,
            trigger=CronTrigger(
                hour=self.daily_hour,
                minute=self.daily_minute,
                timezone=self.tz,
            ),
            kwargs={"respect_schedule_time": False},
            id="daily_reminder_dispatch",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
            misfire_grace_time=3600,
        )

        self.scheduler.add_job(
            self.run_reminder_check,
            trigger=CronTrigger(minute=5, timezone=self.tz),
            kwargs={"respect_schedule_time": True},
            id="hourly_reminder_safety_check",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
            misfire_grace_time=1800,
        )

        startup_run_time = datetime.now(self.tz) + timedelta(seconds=5)
        self.scheduler.add_job(
            self.run_reminder_check,
            trigger=DateTrigger(run_date=startup_run_time, timezone=self.tz),
            kwargs={"respect_schedule_time": True},
            id="startup_reminder_catchup",
            replace_existing=True,
            max_instances=1,
        )

        self.scheduler.start()
        logger.info(
            "ReminderScheduler started (daily=%02d:%02d %s, plus hourly deduplicated check)",
            self.daily_hour,
            self.daily_minute,
            self.timezone_name,
        )

    def shutdown(self) -> None:
        """Stop the scheduler cleanly."""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("ReminderScheduler stopped")

    async def _alert_error(
        self,
        operation: str,
        exc: BaseException,
        *,
        context_info: str,
        dedupe_key: Optional[str] = None,
    ) -> None:
        if self.notifier is not None:
            await self.notifier.notify_error(
                operation,
                exc,
                context_info=context_info,
                dedupe_key=dedupe_key,
            )

    async def run_reminder_check(self, *, respect_schedule_time: bool = True) -> dict[str, int]:
        """Expire overdue reminders and atomically dispatch due reminders for today."""
        async with self._lock:
            now = datetime.now(self.tz)
            today = now.date()
            today_iso = today.isoformat()

            expired_count = 0
            sent_count = 0
            failed_count = 0

            try:
                active_reminders = await self.db.list_active_reminders()
            except Exception as exc:
                logger.exception("Failed to load active reminders from database")
                await self._alert_error(
                    "Database Error",
                    exc,
                    context_info="Scheduled reminder check could not load active reminders from Supabase",
                    dedupe_key="sched_load_active_err",
                )
                return {"expired": 0, "sent": 0, "failed": 1}

            scheduled_cutoff = time(hour=self.daily_hour, minute=self.daily_minute)
            can_dispatch_now = (not respect_schedule_time) or (now.time() >= scheduled_cutoff)

            for reminder in active_reminders:
                try:
                    days_left = reminder.days_left(today)
                except ValueError as exc:
                    logger.error(
                        "Skipping reminder #%d with invalid end_date=%r",
                        reminder.id,
                        reminder.end_date,
                    )
                    await self._alert_error(
                        "Scheduled Reminder",
                        exc,
                        context_info=f"Reminder #{reminder.id} ({reminder.title}) has invalid end_date={reminder.end_date!r}",
                        dedupe_key=f"sched_bad_date:{reminder.id}",
                    )
                    continue

                if days_left < 0:
                    try:
                        await self.db.update_reminder_status(reminder.id, "expired")
                        expired_count += 1
                        logger.info(
                            "Marked reminder #%d (%s) as expired (end_date=%s, today=%s)",
                            reminder.id,
                            reminder.title,
                            reminder.end_date,
                            today_iso,
                        )
                    except Exception as exc:
                        failed_count += 1
                        logger.exception("Failed to mark reminder #%d as expired", reminder.id)
                        await self._alert_error(
                            "Database Error",
                            exc,
                            context_info=f"Could not mark expired reminder #{reminder.id} ({reminder.title})",
                            dedupe_key=f"sched_expire_err:{reminder.id}",
                        )
                    continue

                if not can_dispatch_now:
                    continue

                if not should_send_scheduled_reminder(reminder, today):
                    continue

                try:
                    claimed, previous_reminded_on = await self.db.claim_reminder_for_today(
                        reminder.id,
                        today_iso,
                    )
                except Exception as exc:
                    failed_count += 1
                    logger.exception("Failed to claim reminder #%d for dispatch", reminder.id)
                    await self._alert_error(
                        "Database Error",
                        exc,
                        context_info=f"Could not claim scheduled reminder #{reminder.id} ({reminder.title})",
                        dedupe_key=f"sched_claim_err:{reminder.id}",
                    )
                    continue

                if not claimed:
                    logger.debug(
                        "Reminder #%d already claimed/reminded for %s; skipping",
                        reminder.id,
                        today_iso,
                    )
                    continue

                message_text = format_channel_reminder(
                    title=reminder.title,
                    end_date=reminder.end_date,
                    post_link=reminder.post_link,
                )

                try:
                    await _send_channel_html_with_retry(
                        bot=self.bot,
                        channel_id=self.target_channel_id,
                        text=message_text,
                    )
                except Exception as exc:
                    failed_count += 1
                    logger.exception(
                        "Failed to send scheduled reminder #%d (%s) to channel %s; reverting claim",
                        reminder.id,
                        reminder.title,
                        self.target_channel_id,
                    )
                    await self._alert_error(
                        "Scheduled Reminder",
                        exc,
                        context_info=(
                            f"Failed posting reminder #{reminder.id} ({reminder.title}) "
                            f"to channel {self.target_channel_id}"
                        ),
                        dedupe_key=f"sched_send_err:{reminder.id}:{today_iso}",
                    )
                    try:
                        await self.db.update_last_reminded_on(
                            reminder.id,
                            previous_reminded_on,
                        )
                    except Exception as revert_exc:
                        logger.exception(
                            "Failed to revert last_reminded_on for reminder #%d",
                            reminder.id,
                        )
                        await self._alert_error(
                            "Database Error",
                            revert_exc,
                            context_info=f"Failed reverting claim for reminder #{reminder.id}",
                            dedupe_key=f"sched_revert_err:{reminder.id}",
                        )
                    continue

                sent_count += 1
                logger.info(
                    "Dispatched scheduled reminder #%d (%s, days_left=%d, mode=%s)",
                    reminder.id,
                    reminder.title,
                    days_left,
                    reminder.repeat_mode,
                )
                await asyncio.sleep(0.5)

            logger.info(
                "Reminder check complete (today=%s, expired=%d, sent=%d, failed=%d)",
                today_iso,
                expired_count,
                sent_count,
                failed_count,
            )
            return {
                "expired": expired_count,
                "sent": sent_count,
                "failed": failed_count,
            }
