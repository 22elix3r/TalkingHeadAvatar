"""
Meeting transcript management for rolling context windows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class TranscriptEntry:
    speaker: str
    text: str
    timestamp: float


class MeetingTranscript:
    """Rolling transcript with oldest-first truncation by token budget."""

    def __init__(self, max_tokens: int = 120000):
        self.max_tokens = max_tokens
        self.entries: list[TranscriptEntry] = []

    def add(self, speaker: str, text: str, timestamp: float | None = None):
        self.entries.append(
            TranscriptEntry(
                speaker=speaker,
                text=text.strip(),
                timestamp=time.time() if timestamp is None else timestamp,
            )
        )

    def extend(self, entries: Iterable[TranscriptEntry]):
        self.entries.extend(entries)

    def to_context(self, tokenizer) -> str:
        while self.entries:
            formatted = self._format_entries(self.entries)
            tokens = tokenizer.encode(formatted, add_special_tokens=False)
            if len(tokens) <= self.max_tokens:
                return formatted
            self.entries.pop(0)
        return ""

    def truncate_to_last_n_minutes(self, minutes: float, now: float | None = None):
        """
        Drop entries older than ``minutes`` from the current time.
        """
        if minutes <= 0:
            self.entries.clear()
            return
        cutoff = (time.time() if now is None else now) - (minutes * 60.0)
        self.entries = [entry for entry in self.entries if entry.timestamp >= cutoff]

    @staticmethod
    def _format_entries(entries: list[TranscriptEntry]) -> str:
        return "\n".join(f"[{entry.speaker}]: {entry.text}" for entry in entries)
