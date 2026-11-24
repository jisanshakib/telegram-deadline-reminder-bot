"""Gemini API integration for time-period & deadline decision extraction."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import date
from typing import Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-4-hydrogen"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 3

# Non-retryable HTTP status codes from Gemini API (fail fast so admin is alerted immediately)
_NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404}

_SYSTEM_INSTRUCTION_TEMPLATE = """You inspect Telegram channel posts and decide whether each post contains a time period, date range, duration, or event deadline that qualifies to be tracked as a reminder.

Rules:
1. Evaluate EVERY post against the provided current date ({timezone_name}).
2. Decide `has_deadline = true` whenever the post announces an event, campaign, contest, giveaway, airdrop, registration, sale, scholarship, submission, offer, snapshot, or task with an explicit or calculable end date, date range, time period, or duration (e.g. "Nov 24 to Dec 5", "ends in 7 days", "valid for 48 hours", "ends tomorrow", "last date 15 Dec", "closes tonight", "until this Friday").
3. Decide `has_deadline = false` if the post is general news, commentary, a reminder/repost with no actionable event window, or mentions no time period / deadline at all.
4. Infer a concise, clear `title` capturing the main event or topic (no hashtags, emojis, or trailing punctuation; max ~80 characters).
5. Normalize `start_date` and `end_date` to strict ISO format `YYYY-MM-DD`, anchored to the provided current date ({timezone_name}).
   - For durations like "valid for 7 days" or "ends in 5 days" without an explicit start date, set `start_date` to the current date and compute `end_date` accordingly.
   - For a single event/deadline date (e.g. "Deadline: 15 December"), set `end_date` to that date and `start_date` to null unless a start date is stated.
   - If a month and day are given without a year, ALWAYS anchor the year to the provided current date (current year, or next year if that month/day already passed) — never fall back to an older year.
6. Provide a brief one-sentence `decision_reason` explaining why this post qualifies or does not qualify as a reminder.
7. Never invent dates that are not supported by the post text. If `start_date` or `end_date` cannot be reliably determined, return null for that field."""


class DeadlineExtraction(BaseModel):
    """Structured output schema returned by the Gemini API."""

    has_deadline: bool = Field(
        description=(
            "True if the post contains a qualifying time period, date range, duration, "
            "or deadline/end date to track as a reminder; otherwise False."
        )
    )
    title: Optional[str] = Field(
        default=None,
        description="Concise main event or topic title inferred from the post.",
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Event start date in YYYY-MM-DD format, or null if not stated.",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="Event end date / deadline in YYYY-MM-DD format, or null if not stated.",
    )
    decision_reason: Optional[str] = Field(
        default=None,
        description="Brief one-sentence explanation of Gemini's decision.",
    )


def _normalize_iso_date(value: Optional[str]) -> Optional[str]:
    """Validate and normalize an ISO date string (YYYY-MM-DD), returning None if invalid."""
    if not value or not isinstance(value, str):
        return None
    cleaned = value.strip()
    if cleaned.lower() in {"null", "none", "n/a", ""}:
        return None
    match = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:[T\s].*)?$", cleaned)
    if not match:
        return None
    try:
        parsed = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        return parsed.isoformat()
    except ValueError:
        return None


def _clean_text(value: Optional[str], max_len: int) -> Optional[str]:
    """Sanitize extracted string fields."""
    if not value or not isinstance(value, str):
        return None
    cleaned = " ".join(value.strip().split())
    if not cleaned or cleaned.lower() in {"null", "none", "n/a"}:
        return None
    return cleaned[:max_len]


def _clean_title(value: Optional[str]) -> Optional[str]:
    """Sanitize extracted title text."""
    return _clean_text(value, 120)


def _strip_markdown_fences(raw_text: str) -> str:
    """Remove optional ```json ... ``` fences if a model wraps JSON text."""
    cleaned = raw_text.strip()
    fence_match = re.match(
        r"^```(?:json)?\s*(.*?)\s*```$",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fence_match:
        return fence_match.group(1).strip()
    return cleaned


def _parse_response_payload(response: object) -> Optional[DeadlineExtraction]:
    """Safely extract DeadlineExtraction from response.parsed or response.text."""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, DeadlineExtraction):
        return parsed
    if isinstance(parsed, dict):
        try:
            return DeadlineExtraction.model_validate(parsed)
        except ValidationError:
            logger.warning("Gemini response.parsed dict failed schema validation")

    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        cleaned = _strip_markdown_fences(text)
        try:
            return DeadlineExtraction.model_validate_json(cleaned)
        except ValidationError:
            try:
                data = json.loads(cleaned)
                if isinstance(data, dict):
                    return DeadlineExtraction.model_validate(data)
            except (json.JSONDecodeError, ValidationError):
                logger.warning("Failed to parse Gemini JSON text output: %.200s", cleaned)

    return None


def _is_non_retryable_error(exc: BaseException) -> bool:
    """Detect client/configuration errors (400/401/403/404) that should fail fast."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int) and code in _NON_RETRYABLE_STATUS_CODES:
        return True
    msg = str(exc).upper()
    return any(
        marker in msg
        for marker in (
            "404 NOT_FOUND",
            "403 PERMISSION_DENIED",
            "401 UNAUTHENTICATED",
            "API_KEY_INVALID",
        )
    )


class GeminiDeadlineParser:
    """Async Gemini client wrapper using google-genai structured output."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GEMINI_MODEL,
        timezone_name: str = "Asia/Dhaka",
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.timezone_name = timezone_name
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(1, max_retries)
        self._system_instruction = _SYSTEM_INSTRUCTION_TEMPLATE.format(
            timezone_name=timezone_name
        )

    async def extract_deadline(
        self,
        post_text: str,
        current_date: date,
    ) -> DeadlineExtraction:
        """Ask Gemini to decide if the post has a qualifying time period/deadline."""
        trimmed_text = post_text.strip()
        if not trimmed_text:
            return DeadlineExtraction(
                has_deadline=False,
                title=None,
                start_date=None,
                end_date=None,
                decision_reason="Post had no text or caption content.",
            )

        prompt = (
            f"Current Date ({self.timezone_name}): {current_date.isoformat()} "
            f"({current_date.strftime('%A, %d %B %Y')}, Year={current_date.year})\n\n"
            f"Telegram Channel Post:\n{trimmed_text[:6000]}"
        )

        last_error: Optional[Exception] = None
        response: object = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    self.client.aio.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=self._system_instruction,
                            response_mime_type="application/json",
                            response_schema=DeadlineExtraction,
                            temperature=0.1,
                        ),
                    ),
                    timeout=self.timeout_seconds,
                )
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if _is_non_retryable_error(exc) or attempt >= self.max_retries:
                    logger.exception(
                        "Gemini API failed (model=%s, attempt=%d/%d): %s",
                        self.model,
                        attempt,
                        self.max_retries,
                        exc,
                    )
                    break
                backoff = min(2.0 ** (attempt - 1), 6.0)
                logger.warning(
                    "Gemini API attempt %d/%d failed (model=%s, err=%s); retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    self.model,
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)

        if last_error is not None:
            raise RuntimeError(
                f"Gemini model '{self.model}' error: {type(last_error).__name__}: {last_error}"
            ) from last_error

        raw_parsed = _parse_response_payload(response)
        if raw_parsed is None:
            raw_preview = str(getattr(response, "text", "") or "")[:200]
            raise ValueError(
                f"Gemini model '{self.model}' returned an empty or unparseable JSON response"
                + (f": {raw_preview}" if raw_preview else "")
            )

        start_date = _normalize_iso_date(raw_parsed.start_date)
        end_date = _normalize_iso_date(raw_parsed.end_date)
        title = _clean_title(raw_parsed.title)
        decision_reason = _clean_text(getattr(raw_parsed, "decision_reason", None), 240)

        if start_date and end_date and date.fromisoformat(start_date) > date.fromisoformat(end_date):
            logger.warning(
                "Gemini extracted start_date (%s) > end_date (%s); clearing invalid start_date",
                start_date,
                end_date,
            )
            start_date = None

        has_deadline = bool(raw_parsed.has_deadline and end_date is not None)

        return DeadlineExtraction(
            has_deadline=has_deadline,
            title=title,
            start_date=start_date,
            end_date=end_date,
            decision_reason=decision_reason,
        )
