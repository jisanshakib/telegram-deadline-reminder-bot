"""Worker entrypoint for the Telegram Event Deadline Reminder Bot (Railway-ready)."""

from __future__ import annotations

import logging
import os
import re
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
from scheduler import AdminNotifier, ReminderScheduler

_CHANNEL_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?(?:t\.me|telegram\.me)/@?([A-Za-z][A-Za-z0-9_]{3,})/?$",
    re.IGNORECASE,
)


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
    value = os.getenv(name, "").strip().strip("\"'")
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _parse_channel_id(raw: str) -> Union[int, str]:
    """Normalize CHANNEL_ID from -100..., bare internal digits, @username, or t.me/username."""
    cleaned = raw.strip().strip("\"'")
    if cleaned.startswith("-100") and cleaned[4:].isdigit():
        return int(cleaned)
    if cleaned.startswith("-") and cleaned[1:].isdigit():
        return int(cleaned)
    if cleaned.isdigit():
        # Users frequently copy the internal ID from https://t.me/c/<internal_id>/123 without -100
        return int(f"-100{cleaned}")
    url_match = _CHANNEL_URL_RE.match(cleaned)
    if url_match:
        return f"@{url_match.group(1)}"
    if cleaned.startswith("@"):
        return cleaned
    return f"@{cleaned}"


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip().strip("\"'")
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
    prepare_postgres_dsn(database_url)

    timezone = os.getenv("TIMEZONE", "Asia/Dhaka").strip().strip("\"'") or "Asia/Dhaka"
    try:
        resolve_timezone(timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(
            f"Unknown TIMEZONE '{timezone}'. Install 'tzdata' and use a valid IANA timezone."
        ) from exc

    gemini_model = (
        os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip().strip("\"'") or DEFAULT_GEMINI_MODEL
    )

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
    """Connect to Supabase, verify channel access, and start APScheduler on worker boot."""
    logger = logging.getLogger(__name__)
    admin_id: int = application.bot_data["admin_user_id"]
    notifier = AdminNotifier(bot=application.bot, admin_user_id=admin_id)
    application.bot_data["notifier"] = notifier

    db: Database = application.bot_data["db"]
    try:
        await db.connect()
    except Exception as exc:
        logger.exception("Database connection failed on startup")
        await notifier.notify_error(
            "Database Error",
            exc,
            context_info="Worker startup could not connect to Supabase PostgreSQL",
            dedupe_key="startup_db_err",
        )
        raise

    # Resolve canonical numeric channel ID & verify bot access on startup
    configured_channel = application.bot_data["channel_id"]
    effective_target_channel = configured_channel
    try:
        chat = await application.bot.get_chat(configured_channel)
        application.bot_data["resolved_channel_id"] = chat.id
        if chat.username:
            application.bot_data["resolved_channel_username"] = chat.username.lower()
        effective_target_channel = chat.id
        logger.info(
            "Verified target channel access: id=%s, title=%r, username=%s",
            chat.id,
            getattr(chat, "title", None),
            chat.username,
        )
        try:
            me = await application.bot.get_me()
            member = await application.bot.get_chat_member(chat.id, me.id)
            if member.status not in {"administrator", "creator"}:
                await notifier.notify_error(
                    "Channel Setup Check",
                    (
                        f"Bot is currently '{member.status}' in channel {chat.id} "
                        "(not an administrator). Telegram only delivers channel_post updates "
                        "when the bot is an Admin with Post Messages permission."
                    ),
                    context_info=f"Configured CHANNEL_ID={configured_channel}",
                    dedupe_key="startup_not_admin",
                )
        except Exception as member_exc:
            logger.warning("Could not verify bot admin role in channel %s: %s", chat.id, member_exc)
    except Exception as chat_exc:
        logger.exception("Could not access configured CHANNEL_ID=%s on startup", configured_channel)
        await notifier.notify_error(
            "Channel Setup Check",
            chat_exc,
            context_info=(
                f"Configured CHANNEL_ID={configured_channel}. Ensure the bot is added as "
                "an Administrator to the channel and CHANNEL_ID is correct."
            ),
            dedupe_key="startup_channel_err",
        )

    scheduler = ReminderScheduler(
        bot=application.bot,
        db=db,
        target_channel_id=effective_target_channel,
        timezone=application.bot_data["timezone"],
        daily_hour=application.bot_data["daily_hour"],
        daily_minute=application.bot_data["daily_minute"],
        notifier=notifier,
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

    # Explicitly accept all update types so channel_post & edited_channel_post are always delivered
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=False,
    )


if __name__ == "__main__":
    main()
