"""Worker entrypoint for the Telegram Event Deadline Reminder Bot (Railway-ready)."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Union
from zoneinfo import ZoneInfoNotFoundError

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, ApplicationBuilder

from database import Database, prepare_postgres_dsn, resolve_timezone
from gemini_parser import DEFAULT_GEMINI_MODEL, GeminiDeadlineParser
from handlers import register_handlers
from scheduler import ReminderScheduler


@dataclass(frozen=True, slots=True)
class BotConfig:
    """Validated runtime configuration loaded from environment variables."""

    bot_token: str
    gemini_api_key: str
    admin_user_id: int
    channel_id: Union[int, str]
    database_url: str
    timezone: str
    gemini_model: str
    daily_reminder_hour: int
    daily_reminder_minute: int


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _parse_channel_id(raw: str) -> Union[int, str]:
    cleaned = raw.strip()
    if cleaned.lstrip("-").isdigit():
        return int(cleaned)
    return cleaned


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer (got {raw!r})") from exc


def load_config() -> BotConfig:
    """Load and validate configuration from .env (local) and environment (Railway)."""
    load_dotenv()

    bot_token = _require_env("BOT_TOKEN")
    gemini_api_key = _require_env("GEMINI_API_KEY")

    raw_admin_id = _require_env("ADMIN_USER_ID")
    try:
        admin_user_id = int(raw_admin_id)
    except ValueError as exc:
        raise ValueError("ADMIN_USER_ID must be a numeric Telegram user ID") from exc
    if admin_user_id <= 0:
        raise ValueError("ADMIN_USER_ID must be a positive Telegram user ID")

    channel_id = _parse_channel_id(_require_env("CHANNEL_ID"))

    database_url = _require_env("DATABASE_URL")
    # Validate DSN shape early before starting Telegram polling
    prepare_postgres_dsn(database_url)

    timezone = os.getenv("TIMEZONE", "Asia/Dhaka").strip() or "Asia/Dhaka"
    try:
        resolve_timezone(timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(
            f"Unknown TIMEZONE '{timezone}'. Install 'tzdata' and use a valid IANA timezone."
        ) from exc

    gemini_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL

    daily_hour = _parse_int_env("DAILY_REMINDER_HOUR", 10)
    daily_minute = _parse_int_env("DAILY_REMINDER_MINUTE", 0)
    if not (0 <= daily_hour <= 23 and 0 <= daily_minute <= 59):
        raise ValueError("DAILY_REMINDER_HOUR must be 0-23 and DAILY_REMINDER_MINUTE must be 0-59")

    return BotConfig(
        bot_token=bot_token,
        gemini_api_key=gemini_api_key,
        admin_user_id=admin_user_id,
        channel_id=channel_id,
        database_url=database_url,
        timezone=timezone,
        gemini_model=gemini_model,
        daily_reminder_hour=daily_hour,
        daily_reminder_minute=daily_minute,
    )


def configure_logging() -> None:
    """Configure unbuffered stdout logging for Railway and local terminals."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        stream=sys.stdout,
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("apscheduler.executors.default").setLevel(logging.WARNING)


async def _on_post_init(application: Application) -> None:
    """Connect to Supabase PostgreSQL and start APScheduler once the event loop is ready."""
    db: Database = application.bot_data["db"]
    await db.connect()

    scheduler = ReminderScheduler(
        bot=application.bot,
        db=db,
        target_channel_id=application.bot_data["channel_id"],
        timezone=application.bot_data["timezone"],
        daily_hour=application.bot_data["daily_hour"],
        daily_minute=application.bot_data["daily_minute"],
    )
    scheduler.start()
    application.bot_data["scheduler"] = scheduler


async def _on_post_shutdown(application: Application) -> None:
    """Gracefully stop APScheduler and close the Supabase connection pool on SIGTERM/SIGINT."""
    scheduler: ReminderScheduler | None = application.bot_data.get("scheduler")
    if scheduler is not None:
        scheduler.shutdown()

    db: Database | None = application.bot_data.get("db")
    if db is not None:
        await db.close()


def main() -> None:
    """Build and run the Telegram bot worker."""
    configure_logging()
    logger = logging.getLogger(__name__)

    try:
        config = load_config()
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    db = Database(database_url=config.database_url, timezone=config.timezone)
    parser = GeminiDeadlineParser(
        api_key=config.gemini_api_key,
        model=config.gemini_model,
        timezone_name=config.timezone,
    )

    application = (
        ApplicationBuilder()
        .token(config.bot_token)
        .post_init(_on_post_init)
        .post_shutdown(_on_post_shutdown)
        .build()
    )

    application.bot_data["db"] = db
    application.bot_data["parser"] = parser
    application.bot_data["admin_user_id"] = config.admin_user_id
    application.bot_data["channel_id"] = config.channel_id
    application.bot_data["timezone"] = config.timezone
    application.bot_data["daily_hour"] = config.daily_reminder_hour
    application.bot_data["daily_minute"] = config.daily_reminder_minute

    register_handlers(application)

    logger.info(
        "Starting Telegram Deadline Reminder Worker (channel=%s, admin=%d, tz=%s, model=%s)",
        config.channel_id,
        config.admin_user_id,
        config.timezone,
        config.gemini_model,
    )

    application.run_polling(
        allowed_updates=[
            Update.CHANNEL_POST,
            Update.EDITED_CHANNEL_POST,
            Update.MESSAGE,
            Update.CALLBACK_QUERY,
        ],
        drop_pending_updates=False,
    )


if __name__ == "__main__":
    main()
