"""Gemini API integration for structured event deadline extraction."""

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

_SYSTEM_INSTRUCTION_TEMPLATE = """You extract event deadlines from Telegram channel posts.
Rules:
1. Analyze the post carefully and determine whether it mentions an event, campaign, contest, giveaway, airdrop, registration, sale, scholarship, submission, or task that has a deadline or end date.
2. Set `has_deadline` to true ONLY if an explicit or clearly calculable end date / deadline exists in the post.
3. Infer a concise, clear `title` capturing the main event or topic (no hashtags, emojis, or trailing punctuation; max ~80 characters).
4. Normalize `start_date` and `end_date` to strict ISO format `YYYY-MM-DD`, using the provided current date ({timezone_name}) to resolve relative expressions (e.g. "ends in 5 days", "ends tomorrow", "until this Friday", "ends May 20").
5. If a month and day are given without a year, ALWAYS anchor the year to the provided current date (choosing the current year or next year if that month/day already passed) — never fall back to an older year.
6. If `start_date` or `end_date` is not present or cannot be reliably determined, return null for that field.
7. Never invent dates that are not supported by the post text."""


class DeadlineExtraction(BaseModel):
    """Structured output schema returned by the Gemini API."""

    has_deadline: bool = Field(
        description="Whether the post contains a clear event deadline or end date."
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


def _clean_title(value: Optional[str]) -> Optional[str]:
    """Sanitize extracted title text."""
    if not value or not isinstance(value, str):
        return None
    cleaned = " ".join(value.strip().split())
    if not cleaned or cleaned.lower() in {"null", "none", "n/a"}:
        return None
    return cleaned[:120]


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
        for marker in ("404 NOT_FOUND", "403 PERMISSION_DENIED", "401 UNAUTHENTICATED", "API_KEY_INVALID")
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
        """Extract structured deadline fields from a Telegram post with retry and validation."""
        trimmed_text = post_text.strip()
        if not trimmed_text:
            return DeadlineExtraction(
                has_deadline=False,
                title=None,
                start_date=None,
                end_date=None,
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
        )
