"""Persistent interaction logging for patient sessions."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


def _timestamp() -> float:
    return time.time()


@dataclass
class InteractionEvent:
    """Single interaction event within a session."""

    event_type: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=_timestamp)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }


@dataclass
class SessionInteractionLog:
    """Aggregated interaction history for one session."""

    session_id: str
    patient_id: Any
    user: str | None = None
    created_at: float = field(default_factory=_timestamp)
    events: List[InteractionEvent] = field(default_factory=list)

    def set_user(self, user: str | None) -> None:
        self.user = user

    def add_event(self, event_type: str, payload: Dict[str, Any]) -> InteractionEvent:
        event = InteractionEvent(event_type=event_type, payload=payload)
        self.events.append(event)
        return event

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "patient_id": self.patient_id,
             "user": self.user,
            "created_at": self.created_at,
            "events": [event.to_dict() for event in self.events],
        }

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"interactions_{self.session_id}.json"
        with path.open("w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, ensure_ascii=False, indent=2)
        return path
