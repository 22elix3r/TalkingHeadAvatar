"""
Persona loading and system-prompt construction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_PERSONA_FIELDS = (
    "name",
    "role",
    "communication_style",
    "vocabulary_patterns",
    "domain_expertise",
    "known_opinions",
    "biographical_context",
    "tone_guardrails",
)

SYSTEM_PROMPT_TEMPLATE = """You are {name}, {role}. You are attending a live video meeting.
You speak exactly as {name} would — matching their vocabulary, cadence, and opinions.

Communication style: {communication_style}
Signature phrases: {vocabulary_patterns}
Domain expertise: {domain_expertise}
Known positions: {known_opinions}
Background: {biographical_context}
Tone: {tone_guardrails}

RULES:
- You ARE this person. Never break character.
- Keep responses concise (1-3 sentences typical for meetings).
- Output only the exact words that should be spoken aloud.
- Do not output JSON, tool calls, XML tags, Markdown, labels, field names, or metadata.
- Only speak when directly addressed or when you have a substantive contribution.
- Never narrate actions. Only produce speech content."""


def load_persona(path: str | Path) -> dict[str, Any]:
    persona_path = Path(path)
    data = json.loads(persona_path.read_text(encoding="utf-8"))
    persona = data.get("persona", data)
    missing = [field for field in REQUIRED_PERSONA_FIELDS if field not in persona]
    if missing:
        raise ValueError(f"Persona file missing required fields: {', '.join(missing)}")
    return persona


def build_system_prompt(persona: dict[str, Any]) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        name=persona["name"],
        role=persona["role"],
        communication_style=persona["communication_style"],
        vocabulary_patterns=_format_sequence(persona["vocabulary_patterns"]),
        domain_expertise=_format_sequence(persona["domain_expertise"]),
        known_opinions=_format_mapping(persona["known_opinions"]),
        biographical_context=persona["biographical_context"],
        tone_guardrails=persona["tone_guardrails"],
    )


def _format_sequence(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


def _format_mapping(value: Any) -> str:
    if isinstance(value, dict):
        return "; ".join(f"{k}: {v}" for k, v in value.items())
    return str(value)
