"""Utility helpers shared across backend modules."""

from __future__ import annotations

from typing import Any, Dict


def safe_get_text(
    data: Dict[str, Any],
    primary_key: str,
    fallback_key: str = "",
    default: str = "",
) -> str:
    """Safely extract text values from dictionaries, handling None gracefully."""
    value = data.get(primary_key)
    if value is None and fallback_key:
        value = data.get(fallback_key)
    if value is None:
        return default
    return str(value).strip()


def safe_truncate_text(text: Any, max_length: int) -> str:
    """Truncate long text values without raising on None."""
    if text is None:
        return ""
    text = str(text).strip()
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

