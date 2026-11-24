"""Async Supabase PostgreSQL persistence layer using asyncpg."""

from __future__ import annotations

import logging
import ssl
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, tzinfo
from typing import Any, Mapping, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import asyncpg

logger = logging.getLogger(__name__)

# Query parameters often copied from Prisma/Supabase connection snippets that asyncpg rejects.
_UNSUPPORTED_ASYNCPG_QUERY_PARAMS = frozenset({"pgbouncer", "connection_limit", "pool_timeout"})


def resolve_timezone(tz_name: str) -> tzinfo:
    """Resolve an IANA timezone name via ZoneInfo, with a safe Asia/Dhaka fallback."""
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        normalized = tz_name.strip().lower()
        if normalized == "asia/dhaka":
            logger.warning(
                "tzdata package not installed; falling back to fixed UTC+06:00 for Asia/Dhaka. "
                "Install 'tzdata' for full IANA timezone support."
            )
            return timezone(timedelta(hours=6), name="Asia/Dhaka")
        if normalized in {"utc", "etc/utc"}:
            return timezone.utc
        raise


def prepare_postgres_dsn(database_url: str) -> tuple[str, dict[str, Any]]:
    """Normalize a Supabase/PostgreSQL DATABASE_URL and derive asyncpg pool kwargs.

    Handles:
    - `postgres://` -> `postgresql://` scheme normalization.
    - Stripping non-libpq query params such as `pgbouncer=true`.
    - Enforcing TLS (`ssl="require"`) for remote hosts (including Supabase) unless explicitly disabled.
    - Disabling prepared statement caching (`statement_cache_size=0`) for Supabase Supavisor
      transaction pooler compatibility (port 6543) while remaining fully compatible with
      direct/session connections (port 5432).
    """
    raw = database_url.strip()
    if not raw:
        raise ValueError("DATABASE_URL cannot be empty")

    if raw.startswith("postgres://"):
        raw = "postgresql://" + raw[len("postgres://") :]

    parsed = urlparse(raw)
    if parsed.scheme not in {"postgresql", "postgres"}:
        raise ValueError(
            "DATABASE_URL must be a valid PostgreSQL connection string "
            "(starting with postgresql:// or postgres://)"
        )
    if not parsed.hostname:
        raise ValueError("DATABASE_URL is missing a database host")

    query_items = parse_qsl(parsed.query, keep_blank_values=True)
    sslmode_val: Optional[str] = None
    filtered_items: list[tuple[str, str]] = []

    for key, val in query_items:
        lower_key = key.lower()
        if lower_key in _UNSUPPORTED_ASYNCPG_QUERY_PARAMS:
            continue
        if lower_key == "sslmode":
            sslmode_val = val.lower()
            continue
        filtered_items.append((key, val))

    clean_query = urlencode(filtered_items)
    clean_dsn = urlunparse(parsed._replace(query=clean_query))

    host = (parsed.hostname or "").lower()
    is_local = host in {"localhost", "127.0.0.1", "::1"}

    ssl_arg: Any
    if sslmode_val == "disable":
        ssl_arg = False
    elif sslmode_val in {"verify-ca", "verify-full"}:
        ssl_arg = ssl.create_default_context()
    elif sslmode_val in {"require", "prefer", "allow"}:
        ssl_arg = "require"
    elif is_local:
        ssl_arg = None
    else:
        # Supabase cloud connections require TLS
        ssl_arg = "require"

    pool_kwargs: dict[str, Any] = {
        "statement_cache_size": 0,  # Required for Supabase Transaction Pooler (Supavisor)
        "min_size": 1,
        "max_size": 5,
        "command_timeout": 30,
        "max_inactive_connection_lifetime": 300,
    }
    if ssl_arg is not None:
        pool_kwargs["ssl"] = ssl_arg

    return clean_dsn, pool_kwargs


@dataclass(slots=True)
class Reminder:
    """Represents a tracked event deadline reminder."""

    id: int
    channel_id: str
    message_id: int
    post_link: str
    title: str
    post_text: str
    start_date: Optional[str]  # YYYY-MM-DD or None
    end_date: str  # YYYY-MM-DD
    window_days: Optional[int]  # 1, 2, 3, 5, 7, or 10
    repeat_mode: Optional[str]  # "normal" or "alternate"
    status: str  # "pending", "active", "declined", "expired"
    last_reminded_on: Optional[str]  # YYYY-MM-DD or None
    created_at: str
    updated_at: str

    def days_left(self, today: date) -> int:
        """Return number of calendar days remaining until end_date."""
        return (date.fromisoformat(self.end_date) - today).days

    def total_duration_days(self) -> Optional[int]:
        """Return total span in days if start_date is available."""
        if not self.start_date:
            return None
        return (
            date.fromisoformat(self.end_date) - date.fromisoformat(self.start_date)
        ).days

    @property
    def repeat_mode_label(self) -> str:
        """Human-friendly label for repeat_mode."""
        if self.repeat_mode == "normal":
            return "Normal (Daily)"
        if self.repeat_mode == "alternate":
            return "Alternate Day"
        return "Not set"

    @property
    def window_label(self) -> str:
        """Human-friendly label for window_days."""
        if self.window_days is None:
            return "Not set"
        suffix = "Day" if self.window_days == 1 else "Days"
        return f"Last {self.window_days} {suffix}"


@dataclass(slots=True)
class CachedPost:
    """Represents a cached channel post for manual link lookup."""

    channel_id: str
    channel_username: Optional[str]
    internal_channel_id: Optional[str]
    message_id: int
    post_text: str
    post_link: str
    posted_at: str


def extract_internal_channel_id(channel_id: str | int) -> Optional[str]:
    """Extract the internal numeric ID used in private https://t.me/c/<id>/<msg> links."""
    raw = str(channel_id).strip()
    if raw.startswith("-100") and raw[4:].isdigit():
        return raw[4:]
    if raw.startswith("-") and raw[1:].isdigit():
        return raw[1:]
    if raw.isdigit():
        return raw
    return None


def _ts_to_str(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    return str(value)


class Database:
    """Async PostgreSQL repository designed for Supabase via asyncpg."""

    def __init__(self, database_url: str, timezone: str = "Asia/Dhaka") -> None:
        self.dsn, self.pool_kwargs = prepare_postgres_dsn(database_url)
        self.tz = resolve_timezone(timezone)
        self._pool: Optional[asyncpg.Pool] = None

    def _now_dt(self) -> datetime:
        return datetime.now(self.tz)

    def today(self) -> date:
        """Return current calendar date in configured timezone."""
        return datetime.now(self.tz).date()

    async def connect(self) -> None:
        """Create asyncpg connection pool and ensure tables/indexes exist."""
        self._pool = await asyncpg.create_pool(dsn=self.dsn, **self.pool_kwargs)
        await self._init_schema()
        logger.info("Connected to Supabase PostgreSQL database pool")

    async def close(self) -> None:
        """Gracefully close all pooled PostgreSQL connections."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("Closed Supabase PostgreSQL database pool")

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Database is not connected. Call await db.connect() first.")
        return self._pool

    async def _init_schema(self) -> None:
        ddl_statements = (
            """
            CREATE TABLE IF NOT EXISTS reminders (
                id BIGSERIAL PRIMARY KEY,
                channel_id TEXT NOT NULL,
                message_id BIGINT NOT NULL,
                post_link TEXT NOT NULL,
                title TEXT NOT NULL,
                post_text TEXT NOT NULL,
                start_date TEXT,
                end_date TEXT NOT NULL,
                window_days INTEGER CHECK (window_days IS NULL OR window_days > 0),
                repeat_mode TEXT CHECK (repeat_mode IS NULL OR repeat_mode IN ('normal', 'alternate')),
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'active', 'declined', 'expired')),
                last_reminded_on TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_reminders_status_end_date
                ON reminders (status, end_date)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_reminders_channel_msg
                ON reminders (channel_id, message_id)
            """,
            """
            CREATE TABLE IF NOT EXISTS channel_posts_cache (
                channel_id TEXT NOT NULL,
                channel_username TEXT,
                internal_channel_id TEXT,
                message_id BIGINT NOT NULL,
                post_text TEXT NOT NULL,
                post_link TEXT NOT NULL,
                posted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (channel_id, message_id)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_cache_username_msg
                ON channel_posts_cache (channel_username, message_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_cache_internal_msg
                ON channel_posts_cache (internal_channel_id, message_id)
            """,
        )
        async with self.pool.acquire() as conn:
            for stmt in ddl_statements:
                await conn.execute(stmt)

    @staticmethod
    def _row_to_reminder(row: Mapping[str, Any]) -> Reminder:
        return Reminder(
            id=int(row["id"]),
            channel_id=str(row["channel_id"]),
            message_id=int(row["message_id"]),
            post_link=str(row["post_link"]),
            title=str(row["title"]),
            post_text=str(row["post_text"]),
            start_date=row["start_date"],
            end_date=str(row["end_date"]),
            window_days=row["window_days"],
            repeat_mode=row["repeat_mode"],
            status=str(row["status"]),
            last_reminded_on=row["last_reminded_on"],
            created_at=_ts_to_str(row["created_at"]),
            updated_at=_ts_to_str(row["updated_at"]),
        )

    @staticmethod
    def _row_to_cached_post(row: Mapping[str, Any]) -> CachedPost:
        return CachedPost(
            channel_id=str(row["channel_id"]),
            channel_username=row["channel_username"],
            internal_channel_id=row["internal_channel_id"],
            message_id=int(row["message_id"]),
            post_text=str(row["post_text"]),
            post_link=str(row["post_link"]),
            posted_at=_ts_to_str(row["posted_at"]),
        )

    # ------------------------------------------------------------------
    # Channel Post Cache CRUD
    # ------------------------------------------------------------------

    async def cache_channel_post(
        self,
        *,
        channel_id: str | int,
        message_id: int,
        post_text: str,
        post_link: str,
        channel_username: Optional[str] = None,
    ) -> None:
        """Insert or update a channel post in the cache table."""
        norm_channel_id = str(channel_id).strip()
        norm_username = channel_username.lstrip("@").lower() if channel_username else None
        internal_id = extract_internal_channel_id(norm_channel_id)
        now = self._now_dt()

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO channel_posts_cache (
                    channel_id,
                    channel_username,
                    internal_channel_id,
                    message_id,
                    post_text,
                    post_link,
                    posted_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (channel_id, message_id) DO UPDATE SET
                    channel_username = COALESCE(EXCLUDED.channel_username, channel_posts_cache.channel_username),
                    internal_channel_id = COALESCE(EXCLUDED.internal_channel_id, channel_posts_cache.internal_channel_id),
                    post_text = EXCLUDED.post_text,
                    post_link = EXCLUDED.post_link,
                    posted_at = EXCLUDED.posted_at
                """,
                norm_channel_id,
                norm_username,
                internal_id,
                int(message_id),
                post_text,
                post_link,
                now,
            )

    async def get_cached_post(
        self,
        *,
        message_id: int,
        channel_id: Optional[str | int] = None,
        channel_username: Optional[str] = None,
        internal_channel_id: Optional[str] = None,
    ) -> Optional[CachedPost]:
        """Look up a cached channel post by channel_id, username, or internal private ID."""
        msg_id = int(message_id)
        async with self.pool.acquire() as conn:
            if channel_id is not None:
                row = await conn.fetchrow(
                    """
                    SELECT * FROM channel_posts_cache
                    WHERE channel_id = $1 AND message_id = $2
                    LIMIT 1
                    """,
                    str(channel_id).strip(),
                    msg_id,
                )
                if row:
                    return self._row_to_cached_post(row)

            if channel_username:
                row = await conn.fetchrow(
                    """
                    SELECT * FROM channel_posts_cache
                    WHERE channel_username = $1 AND message_id = $2
                    LIMIT 1
                    """,
                    channel_username.lstrip("@").lower(),
                    msg_id,
                )
                if row:
                    return self._row_to_cached_post(row)

            if internal_channel_id:
                row = await conn.fetchrow(
                    """
                    SELECT * FROM channel_posts_cache
                    WHERE internal_channel_id = $1 AND message_id = $2
                    LIMIT 1
                    """,
                    str(internal_channel_id).strip(),
                    msg_id,
                )
                if row:
                    return self._row_to_cached_post(row)

        return None

    # ------------------------------------------------------------------
    # Reminders CRUD
    # ------------------------------------------------------------------

    async def create_reminder(
        self,
        *,
        channel_id: str | int,
        message_id: int,
        post_link: str,
        title: str,
        post_text: str,
        start_date: Optional[str],
        end_date: str,
        window_days: Optional[int] = None,
        repeat_mode: Optional[str] = None,
        status: str = "pending",
    ) -> Reminder:
        """Create a new reminder row and return the persisted Reminder object."""
        now = self._now_dt()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO reminders (
                    channel_id,
                    message_id,
                    post_link,
                    title,
                    post_text,
                    start_date,
                    end_date,
                    window_days,
                    repeat_mode,
                    status,
                    last_reminded_on,
                    created_at,
                    updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NULL, $11, $12)
                RETURNING *
                """,
                str(channel_id).strip(),
                int(message_id),
                post_link,
                title.strip(),
                post_text,
                start_date,
                end_date,
                window_days,
                repeat_mode,
                status,
                now,
                now,
            )
        assert row is not None
        return self._row_to_reminder(row)

    async def get_reminder(self, reminder_id: int) -> Optional[Reminder]:
        """Fetch a single reminder by primary key."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM reminders WHERE id = $1",
                int(reminder_id),
            )
        return self._row_to_reminder(row) if row else None

    async def has_open_reminder_for_post(
        self,
        channel_id: str | int,
        message_id: int,
        *,
        post_link: Optional[str] = None,
    ) -> bool:
        """Return True if a pending or active reminder already exists for this post."""
        async with self.pool.acquire() as conn:
            if post_link:
                row = await conn.fetchrow(
                    """
                    SELECT 1 FROM reminders
                    WHERE status IN ('pending', 'active')
                      AND ((channel_id = $1 AND message_id = $2) OR post_link = $3)
                    LIMIT 1
                    """,
                    str(channel_id).strip(),
                    int(message_id),
                    post_link.strip(),
                )
            else:
                row = await conn.fetchrow(
                    """
                    SELECT 1 FROM reminders
                    WHERE channel_id = $1 AND message_id = $2
                      AND status IN ('pending', 'active')
                    LIMIT 1
                    """,
                    str(channel_id).strip(),
                    int(message_id),
                )
        return row is not None

    async def list_active_reminders(self) -> list[Reminder]:
        """Return all active reminders sorted by soonest end_date first."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM reminders
                WHERE status = 'active'
                ORDER BY end_date ASC, id ASC
                """
            )
        return [self._row_to_reminder(r) for r in rows]

    async def update_reminder_status(
        self,
        reminder_id: int,
        status: str,
    ) -> Optional[Reminder]:
        """Update reminder status (pending, active, declined, expired)."""
        now = self._now_dt()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE reminders
                SET status = $1, updated_at = $2
                WHERE id = $3
                RETURNING *
                """,
                status,
                now,
                int(reminder_id),
            )
        return self._row_to_reminder(row) if row else None

    async def update_reminder_settings(
        self,
        reminder_id: int,
        *,
        window_days: int,
        repeat_mode: str,
        activate: bool = False,
    ) -> Optional[Reminder]:
        """Update window_days and repeat_mode, optionally activating the reminder."""
        now = self._now_dt()
        async with self.pool.acquire() as conn:
            if activate:
                row = await conn.fetchrow(
                    """
                    UPDATE reminders
                    SET window_days = $1, repeat_mode = $2, status = 'active', updated_at = $3
                    WHERE id = $4
                    RETURNING *
                    """,
                    int(window_days),
                    repeat_mode,
                    now,
                    int(reminder_id),
                )
            else:
                row = await conn.fetchrow(
                    """
                    UPDATE reminders
                    SET window_days = $1, repeat_mode = $2, updated_at = $3
                    WHERE id = $4
                    RETURNING *
                    """,
                    int(window_days),
                    repeat_mode,
                    now,
                    int(reminder_id),
                )
        return self._row_to_reminder(row) if row else None

    async def update_last_reminded_on(
        self,
        reminder_id: int,
        reminded_date: Optional[str],
    ) -> None:
        """Record (or restore) the YYYY-MM-DD date when a reminder was last posted."""
        now = self._now_dt()
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE reminders
                SET last_reminded_on = $1, updated_at = $2
                WHERE id = $3
                """,
                reminded_date,
                now,
                int(reminder_id),
            )

    async def claim_reminder_for_today(
        self,
        reminder_id: int,
        today_iso: str,
    ) -> tuple[bool, Optional[str]]:
        """Atomically claim a reminder for today's scheduled dispatch.

        Returns `(claimed, previous_last_reminded_on)`.
        If Telegram delivery subsequently fails, pass `previous_last_reminded_on`
        to `update_last_reminded_on` so the next hourly run will retry cleanly.
        """
        now = self._now_dt()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                WITH prev AS (
                    SELECT id, last_reminded_on
                    FROM reminders
                    WHERE id = $1
                      AND status = 'active'
                      AND (last_reminded_on IS NULL OR last_reminded_on <> $2)
                    FOR UPDATE
                )
                UPDATE reminders r
                SET last_reminded_on = $2, updated_at = $3
                FROM prev
                WHERE r.id = prev.id
                RETURNING prev.last_reminded_on AS previous_reminded_on
                """,
                int(reminder_id),
                today_iso,
                now,
            )
        if row is None:
            return False, None
        return True, row["previous_reminded_on"]

    async def delete_reminder(self, reminder_id: int) -> bool:
        """Permanently delete a reminder from PostgreSQL."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "DELETE FROM reminders WHERE id = $1 RETURNING id",
                int(reminder_id),
            )
        return row is not None

    async def expire_past_reminders(self, today_iso: Optional[str] = None) -> int:
        """Mark active reminders whose end_date is strictly before today as expired."""
        cutoff = today_iso or self.today().isoformat()
        now = self._now_dt()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                UPDATE reminders
                SET status = 'expired', updated_at = $1
                WHERE status = 'active' AND end_date < $2
                RETURNING id
                """,
                now,
                cutoff,
            )
        return len(rows)
