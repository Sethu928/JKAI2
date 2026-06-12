"""
Single entry-point for every LLM call in the project.
No other module may instantiate an OpenAI client or call a completion API directly.
"""

import logging
import re
import threading
from typing import Optional

import requests
from openai import OpenAI

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regexes for <think> / <thinking> stripping
# ---------------------------------------------------------------------------
_THINK_CLOSED = re.compile(
    r"<(think|thinking)>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_THINK_OPEN = re.compile(r"<think(?:ing)?\s*>", re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove model chain-of-thought blocks from *text*.

    Closed tags (``<think>…</think>``) are deleted in full.
    An unclosed opening tag causes everything from that point onward to be
    truncated — the model didn't finish hiding its reasoning, so we discard
    the tail rather than leaking it.
    """
    text = _THINK_CLOSED.sub("", text)
    m = _THINK_OPEN.search(text)
    if m:
        text = text[: m.start()]
    return text.strip()


# ---------------------------------------------------------------------------
# Singleton clients
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_local_client: Optional[OpenAI] = None
_cloud_client: Optional[OpenAI] = None


def _get_local_client() -> OpenAI:
    global _local_client
    if _local_client is None:
        with _lock:
            if _local_client is None:
                _local_client = OpenAI(
                    base_url=config.LM_STUDIO_URL,
                    api_key="local",
                )
    return _local_client


def _get_cloud_client() -> Optional[OpenAI]:
    global _cloud_client
    if not config.OPENAI_AVAILABLE:
        return None
    if _cloud_client is None:
        with _lock:
            if _cloud_client is None:
                _cloud_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _cloud_client


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def is_alive(timeout: int = 3) -> bool:
    """Return True if the LM Studio server responds on the /models endpoint."""
    try:
        url = config.LM_STUDIO_URL.rstrip("/") + "/models"
        resp = requests.get(url, timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Core call
# ---------------------------------------------------------------------------

def llm_call(
    messages: list[dict],
    use_cloud: bool = False,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    strip: bool = True,
) -> str:
    """Send *messages* to a LLM and return the assistant reply as a string.

    A token cap is **always** applied (``max_tokens`` or
    ``config.LLM_MAX_TOKENS``).  The call is retried once on network failure.
    On total failure the function returns ``""`` and logs the error — it never
    raises, so callers don't need try/except around every invocation.

    Args:
        messages:    OpenAI-style message list, e.g.
                     ``[{"role": "user", "content": "Hello"}]``.
        use_cloud:   Use the OpenAI cloud client when True *and* available;
                     falls back to the local LM Studio client silently.
        max_tokens:  Hard cap for this call; defaults to
                     ``config.LLM_MAX_TOKENS``.
        temperature: Sampling temperature; defaults to
                     ``config.LLM_TEMPERATURE``.
        strip:       Apply :func:`strip_think` to the raw output when True.

    Returns:
        The assistant message content, or ``""`` on unrecoverable error.
    """
    client: Optional[OpenAI] = None
    model: str = ""

    if use_cloud:
        client = _get_cloud_client()
        model = "gpt-4o-mini"

    if client is None:
        client = _get_local_client()
        model = config.LM_STUDIO_MODEL

    cap = max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS
    temp = temperature if temperature is not None else config.LLM_TEMPERATURE

    last_exc: Optional[Exception] = None
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=cap,
                temperature=temp,
                timeout=config.LLM_TIMEOUT,
            )
            text: str = response.choices[0].message.content or ""
            return strip_think(text) if strip else text
        except Exception as exc:
            last_exc = exc
            if attempt == 0:
                logger.warning("llm_call attempt 1 failed (%s), retrying…", exc)

    logger.error("llm_call failed after 2 attempts: %s", last_exc)
    return ""
