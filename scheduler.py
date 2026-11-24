"""APScheduler-based automated reminder dispatcher and channel message formatter."""

from __future__ import annotations

import asyncio
import html
import logging
from datetime import date, datetime, time, timedelta
from typing import Union

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import RetryAfter

from database import Database, Reminder, resolve_timezone

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.bot = bot
        self.db = db
        self.target_channel_id = target_channel_id
        self.timezone_name = timezone
        self.tz = resolve_timezone(timezone)
        self.daily_hour = daily_hour
        self.daily_minute = daily_minute
        self._lock = asyncio.Lock()
        self.scheduler = AsyncIOScheduler(timezone=self.tz)

    def start(self) -> None:
        """Register scheduled jobs and start AsyncIOScheduler."""
        # 1. Fixed daily reminder dispatch (default: 10:00 AM Asia/Dhaka)
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

        # 2. Hourly safety & expiration check with atomic last_reminded_on deduplication
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

        # 3. One-shot startup catch-up check (5 seconds after worker startup)
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
            except Exception:
                logger.exception("Failed to load active reminders from database")
                return {"expired": 0, "sent": 0, "failed": 1}

            scheduled_cutoff = time(hour=self.daily_hour, minute=self.daily_minute)
            can_dispatch_now = (not respect_schedule_time) or (now.time() >= scheduled_cutoff)

            for reminder in active_reminders:
                try:
                    days_left = reminder.days_left(today)
                except ValueError:
                    logger.error(
                        "Skipping reminder #%d with invalid end_date=%r",
                        reminder.id,
                        reminder.end_date,
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
                    except Exception:
                        failed_count += 1
                        logger.exception("Failed to mark reminder #%d as expired", reminder.id)
                    continue

                if not can_dispatch_now:
                    continue

                if not should_send_scheduled_reminder(reminder, today):
                    continue

                # Atomically claim today's dispatch slot in PostgreSQL to prevent duplicates
                try:
                    claimed, previous_reminded_on = await self.db.claim_reminder_for_today(
                        reminder.id,
                        today_iso,
                    )
                except Exception:
                    failed_count += 1
                    logger.exception("Failed to claim reminder #%d for dispatch", reminder.id)
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
                except Exception:
                    failed_count += 1
                    logger.exception(
                        "Failed to send scheduled reminder #%d (%s) to channel %s; reverting claim",
                        reminder.id,
                        reminder.title,
                        self.target_channel_id,
                    )
                    try:
                        await self.db.update_last_reminded_on(
                            reminder.id,
                            previous_reminded_on,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to revert last_reminded_on for reminder #%d",
                            reminder.id,
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
