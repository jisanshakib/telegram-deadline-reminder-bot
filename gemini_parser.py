"""Gemini API integration with 503 retry/backoff and automatic model fallback."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from datetime import date
from typing import Optional, Sequence

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# Current fast, reliable Gemini models (configurable via GEMINI_MODEL & GEMINI_FALLBACK_MODELS)
DEFAULT_GEMINI_MODEL = "gemini-4-hydrogen"
DEFAULT_FALLBACK_MODELS: tuple[str, ...] = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-4-argon",
)
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES_PER_MODEL = 3

# Auth / key errors that should fail fast without trying fallback models
_AUTH_ERROR_CODES = {401, 403}
# Model-level errors where we should skip retries on this model and try the next fallback model
_MODEL_UNAVAILABLE_CODES = {400, 404}
# Transient server / capacity errors where we retry with backoff before falling back
_TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}

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


def _extract_status_code(exc: BaseException) -> Optional[int]:
    """Extract numeric HTTP status code from google-genai APIError/ServerError or message."""
    for attr in ("code", "status_code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    match = re.search(r"\b(400|401|403|404|408|429|500|502|503|504)\b", str(exc))
    if match:
        return int(match.group(1))
    return None


def _is_auth_error(exc: BaseException) -> bool:
    """Return True for invalid API key / auth errors where fallback models won't help."""
    code = _extract_status_code(exc)
    if code in _AUTH_ERROR_CODES:
        return True
    msg = str(exc).upper()
    return any(
        marker in msg
        for marker in ("PERMISSION_DENIED", "UNAUTHENTICATED", "API_KEY_INVALID", "INVALID API KEY")
    )


def _is_model_not_found_error(exc: BaseException) -> bool:
    """Return True when a model name is unrecognized/unavailable so we switch fallback immediately."""
    code = _extract_status_code(exc)
    if code in _MODEL_UNAVAILABLE_CODES:
        return True
    msg = str(exc).upper()
    return any(
        marker in msg
        for marker in (
            "NOT_FOUND",
            "MODEL NOT FOUND",
            "IS NOT FOUND",
            "NOT SUPPORTED FOR GENERATECONTENT",
            "UNRECOGNIZED MODEL",
        )
    )


def _is_transient_server_error(exc: BaseException) -> bool:
    """Return True for 503 UNAVAILABLE, high-demand, rate-limit, timeout, or 5xx errors."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    code = _extract_status_code(exc)
    if code in _TRANSIENT_STATUS_CODES:
        return True
    cls_name = type(exc).__name__.upper()
    if "SERVERERROR" in cls_name:
        return True
    msg = str(exc).upper()
    return any(
        marker in msg
        for marker in (
            "503",
            "UNAVAILABLE",
            "OVERLOADED",
            "HIGH DEMAND",
            "RESOURCE_EXHAUSTED",
            "TRY AGAIN",
            "DEADLINE_EXCEEDED",
            "INTERNAL",
            "SERVICE UNAVAILABLE",
        )
    )


def _is_non_retryable_error(exc: BaseException) -> bool:
    """Return True if retrying the same model immediately is pointless (auth or model 404)."""
    return _is_auth_error(exc) or _is_model_not_found_error(exc)


def build_model_chain(
    primary_model: str,
    fallback_models: Optional[Sequence[str]] = None,
) -> tuple[str, ...]:
    """Build a deduplicated, ordered list of Gemini models to try."""
    raw_list = [primary_model]
    if fallback_models is not None:
        raw_list.extend(fallback_models)
    else:
        raw_list.extend(DEFAULT_FALLBACK_MODELS)

    seen: set[str] = set()
    ordered: list[str] = []
    for item in raw_list:
        cleaned = (item or "").strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            ordered.append(cleaned)
    return tuple(ordered) or (DEFAULT_GEMINI_MODEL,)


class GeminiDeadlineParser:
    """Async Gemini client wrapper with structured output, 503 backoff, and model fallback."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GEMINI_MODEL,
        timezone_name: str = "Asia/Dhaka",
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES_PER_MODEL,
        fallback_models: Optional[Sequence[str]] = None,
    ) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model.strip() or DEFAULT_GEMINI_MODEL
        self.model_chain = build_model_chain(self.model, fallback_models)
        self.last_used_model: Optional[str] = None
        self.timezone_name = timezone_name
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(1, max_retries)
        self._system_instruction = _SYSTEM_INSTRUCTION_TEMPLATE.format(
            timezone_name=timezone_name
        )

    async def _call_single_model(
        self,
        model_name: str,
        prompt: str,
    ) -> object:
        """Invoke a single Gemini model with exponential backoff on 503/transient errors."""
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                return await asyncio.wait_for(
                    self.client.aio.models.generate_content(
                        model=model_name,
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
            except Exception as exc:
                last_error = exc
                if _is_auth_error(exc) or _is_model_not_found_error(exc):
                    logger.warning(
                        "Gemini model '%s' non-retryable failure (%s: %s)",
                        model_name,
                        type(exc).__name__,
                        exc,
                    )
                    raise

                if attempt >= self.max_retries or not _is_transient_server_error(exc):
                    logger.warning(
                        "Gemini model '%s' failed after %d/%d attempt(s) (%s: %s)",
                        model_name,
                        attempt,
                        self.max_retries,
                        type(exc).__name__,
                        exc,
                    )
                    raise

                # Exponential backoff with light jitter for 503 UNAVAILABLE / high-demand spikes
                backoff = min(1.5 * (2.0 ** (attempt - 1)), 8.0) + random.uniform(0.1, 0.5)
                logger.warning(
                    "Gemini model '%s' transient error (attempt %d/%d: %s); retrying in %.1fs",
                    model_name,
                    attempt,
                    self.max_retries,
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)

        assert last_error is not None
        raise last_error

    async def extract_deadline(
        self,
        post_text: str,
        current_date: date,
    ) -> DeadlineExtraction:
        """Ask Gemini to decide if the post has a qualifying time period/deadline.

        Retries 503 UNAVAILABLE / transient server errors with backoff, then automatically
        falls back across `self.model_chain` before surfacing a clear error to the admin.
        """
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

        attempt_errors: list[str] = []
        raw_parsed: Optional[DeadlineExtraction] = None

        for idx, candidate_model in enumerate(self.model_chain):
            try:
                response = await self._call_single_model(candidate_model, prompt)
            except Exception as exc:
                err_summary = f"{candidate_model}: {type(exc).__name__}: {exc}"
                attempt_errors.append(err_summary)
                if _is_auth_error(exc):
                    raise RuntimeError(
                        f"Gemini authentication failed ({err_summary})"
                    ) from exc

                if idx + 1 < len(self.model_chain):
                    logger.warning(
                        "Falling back from Gemini model '%s' to '%s' after error: %s",
                        candidate_model,
                        self.model_chain[idx + 1],
                        exc,
                    )
                continue

            parsed = _parse_response_payload(response)
            if parsed is not None:
                raw_parsed = parsed
                self.last_used_model = candidate_model
                if candidate_model != self.model:
                    logger.info(
                        "Gemini fallback succeeded using model '%s' (primary was '%s')",
                        candidate_model,
                        self.model,
                    )
                break

            raw_preview = str(getattr(response, "text", "") or "")[:160]
            attempt_errors.append(
                f"{candidate_model}: empty or unparseable JSON"
                + (f" ({raw_preview})" if raw_preview else "")
            )
            if idx + 1 < len(self.model_chain):
                logger.warning(
                    "Model '%s' returned unparseable JSON; trying fallback '%s'",
                    candidate_model,
                    self.model_chain[idx + 1],
                )

        if raw_parsed is None:
            joined = " | ".join(attempt_errors) if attempt_errors else "unknown error"
            raise RuntimeError(
                f"All Gemini models failed ({', '.join(self.model_chain)}): {joined}"
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
