"""
Gemma model loading utilities — GGUF / llama-cpp-python backend.

Drop-in replacement for the previous BitsAndBytes loader.
Requires:
  - llama-cpp-python compiled with CUDA support  (see scripts/install_llama_cpp_python.sh)
  - A Gemma-4-E4B-IT GGUF file  (see scripts/download_gemma_gguf.sh)

Public API is unchanged:
  load_gemma()  ->  (model, processor)   # processor=None for GGUF path
  process_gemma_output(...)
  stream_gemma_with_early_tts(...)
  build_audio_generation_inputs(...)     # stub — audio handled upstream via whisper
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from queue import Queue
from collections.abc import Iterable
from typing import Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Path to the downloaded GGUF repo cache.
# Override via env var GEMMA_GGUF_PATH if needed. The HF cache revision changes
# by download, so discover the actual .gguf instead of hard-coding a snapshot id.
HF_CACHE_REPO_PATH = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--unsloth--gemma-4-E4B-it-GGUF"
)
_LEGACY_GGUF_PATH = (
    HF_CACHE_REPO_PATH
    / "snapshots/ce152932ac27bc40bc9c727386760424d50bb456"
    / "gemma-4-E4B-it-Q4_K_M.gguf"
)

_PREFERRED_QUANT_MARKERS = (
    "Q4_K_M",
    "UD-Q4_K_XL",
    "Q4_K_XL",
    "Q5_K_M",
    "Q6_K",
    "Q8_0",
    "Q3_K_M",
)


def _iter_gguf_files(root: Path) -> Iterable[Path]:
    if root.exists():
        yield from sorted(root.rglob("*.gguf"))


def _select_preferred_gguf(candidates: Iterable[Path]) -> Path | None:
    existing = [path for path in candidates if path.exists() and path.is_file()]
    if not existing:
        return None

    def score(path: Path) -> tuple[int, str]:
        name = path.name
        for index, marker in enumerate(_PREFERRED_QUANT_MARKERS):
            if marker in name:
                return index, name
        return len(_PREFERRED_QUANT_MARKERS), name

    return min(existing, key=score)


def resolve_gemma_gguf_path(gguf_path: str | Path | None = None) -> Path:
    """
    Resolve a Gemma GGUF file from an explicit file/dir, env var, or HF cache.

    Hugging Face stores downloads under revision-specific snapshot directories.
    This keeps the runtime usable after a fresh `huggingface-cli download` without
    requiring the user to copy the resolved snapshot file path by hand.
    """
    explicit_path = gguf_path or os.environ.get("GEMMA_GGUF_PATH")
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if path.is_dir():
            selected = _select_preferred_gguf(_iter_gguf_files(path))
            return selected if selected is not None else path
        return path

    candidates: list[Path] = []
    refs_main = HF_CACHE_REPO_PATH / "refs/main"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        if revision:
            candidates.extend(_iter_gguf_files(HF_CACHE_REPO_PATH / "snapshots" / revision))
    candidates.extend(_iter_gguf_files(HF_CACHE_REPO_PATH / "snapshots"))

    selected = _select_preferred_gguf(candidates)
    return selected if selected is not None else _LEGACY_GGUF_PATH


DEFAULT_GGUF_PATH = resolve_gemma_gguf_path()
GEMMA_GGUF_PATH: Path = Path(os.environ.get("GEMMA_GGUF_PATH", DEFAULT_GGUF_PATH)).expanduser()

# HF model id kept for reference / fallback docs only
GEMMA_MODEL_ID = "google/gemma-4-e4b-it"

# GPU layers to offload.  -1 = all layers on GPU.
# If VRAM is tight during training, set GEMMA_N_GPU_LAYERS=20 (partial offload).
N_GPU_LAYERS: int = int(os.environ.get("GEMMA_N_GPU_LAYERS", "-1"))

# Context window (tokens).  4096 is comfortable for chat turns.
N_CTX: int = int(os.environ.get("GEMMA_N_CTX", "4096"))

AVATAR_TOOLS = [
    {
        "name": "avatar_speak",
        "description": "Signal that the avatar should respond with the given text",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The full reply text to synthesize and render",
                },
                "emotion_mode": {
                    "type": "string",
                    "enum": ["neutral", "engaged", "emphatic", "concerned"],
                    "description": "Upper-face expression bias for the avatar",
                },
            },
            "required": ["text"],
        },
    }
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_gemma(
    gguf_path: str | Path | None = None,
    n_gpu_layers: int | None = None,
    n_ctx: int | None = None,
    verbose: bool = False,
    # Legacy kwargs accepted but ignored — kept for call-site compatibility
    device_map=None,
    model_id=None,
    torch_dtype=None,
    attn_implementation=None,
):
    """
    Load Gemma 4-E4B-IT from a GGUF file using llama-cpp-python.

    Returns:
        (llm, None)   —  second element is None (no separate processor needed).
                          Callers that previously unpacked (model, processor) keep working
                          as long as they pass processor only to build_audio_generation_inputs,
                          which is a stub anyway.

    Environment overrides:
        GEMMA_GGUF_PATH      — path to the .gguf file
        GEMMA_N_GPU_LAYERS   — number of transformer layers to offload to GPU (-1 = all)
        GEMMA_N_CTX          — context window size in tokens
    """
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python is not installed.\n"
            "Run  scripts/install_llama_cpp_python.sh  after training completes."
        ) from exc

    path = resolve_gemma_gguf_path(gguf_path or GEMMA_GGUF_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"GGUF model not found at {path}\n"
            f"Expected a .gguf under {HF_CACHE_REPO_PATH}/snapshots/...\n"
            f"Or set GEMMA_GGUF_PATH to a GGUF file or directory."
        )
    if path.is_dir():
        raise FileNotFoundError(
            f"No .gguf model found under directory: {path}\n"
            f"Set GEMMA_GGUF_PATH to the exact GGUF file."
        )

    layers = n_gpu_layers if n_gpu_layers is not None else N_GPU_LAYERS
    ctx = n_ctx if n_ctx is not None else N_CTX

    print(f"[load_gemma] Loading GGUF from: {path}")
    print(f"[load_gemma] GPU layers: {layers}  |  Context: {ctx}")

    llm = Llama(
        model_path=str(path),
        n_gpu_layers=layers,
        n_ctx=ctx,
        verbose=verbose,
        # Keep one thread per physical core for CPU-fallback layers
        n_threads=os.cpu_count() or 4,
    )

    print("[load_gemma] Model loaded successfully.")
    return llm, None  # (model, processor=None)


# ---------------------------------------------------------------------------
# Output parsing  (unchanged from original)
# ---------------------------------------------------------------------------


def split_first_sentence(text: str) -> tuple[str | None, str]:
    """Detect the first sentence boundary and split text into (sentence, remainder)."""
    match = re.search(r"[.!?]\s", text)
    if match:
        idx = match.end()
        return text[:idx].strip(), text[idx:]

    words = text.split()
    if len(words) >= 12:
        comma_match = re.search(r",\s", text)
        if comma_match and len(text[: comma_match.start()].split()) >= 8:
            idx = comma_match.end()
            return text[:idx].strip(), text[idx:]

    return None, text


def process_gemma_output(
    output_text: str,
    default_emotion: str = "neutral",
) -> dict[str, str] | None:
    """Parse Gemma output and extract avatar_speak tool call payload."""
    call = _extract_json_tool_call(output_text)
    if call is None:
        call = _extract_native_tool_call(output_text)
    if call is None:
        return None

    params = call.get("parameters", {})
    text = clean_avatar_speech_text(params.get("text", ""))
    if not text:
        return None
    emotion = params.get("emotion_mode", default_emotion)
    return {"text": text, "emotion": emotion}


def clean_avatar_speech_text(text: str) -> str:
    """
    Strip tool-call wrappers from model output before anything reaches TTS.

    This is intentionally conservative: normal prose is returned unchanged, but
    common JSON/Gemma/native wrapper formats are reduced to their text payload.
    """
    raw = str(text or "").strip()
    if not raw:
        return ""

    payload = _extract_json_tool_call(raw) or _extract_native_tool_call(raw)
    if payload is not None:
        return clean_avatar_speech_text(payload.get("parameters", {}).get("text", ""))

    cleaned = raw.strip().strip("`")
    cleaned = re.sub(r"<\|/?(?:tool_call|tool_response|turn|channel)[^>]*\|?>", " ", cleaned)
    cleaned = re.sub(r"</?tool_call>|</?tool_response>", " ", cleaned)

    extracted = _extract_text_field(cleaned)
    if extracted:
        cleaned = extracted

    cleaned = re.sub(
        r"^\s*(?:\{?\s*)?(?:\"?name\"?\s*[:=]?\s*)?avatar[_\s-]*speak\b"
        r"(?:\s*,?\s*\"?parameters\"?\s*[:=]?)?\s*(?:\{)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(r"^\s*\"?parameters\"?\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*\"?text\"?\s*[:=]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\s*,?\s*\"?emotion_mode\"?\s*[:=]\s*\"?(neutral|engaged|emphatic|concerned)\"?\s*\}?\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" \t\r\n\"'{}[]")


def _extract_json_tool_call(output_text: str) -> dict[str, Any] | None:
    if "avatar_speak" not in output_text:
        return None
    for candidate in _json_object_candidates(output_text):
        try:
            call = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if call.get("name") == "avatar_speak":
            return call
        function = call.get("function")
        if isinstance(function, dict) and function.get("name") == "avatar_speak":
            args = function.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            return {"name": "avatar_speak", "parameters": args}
    return None


def _json_object_candidates(text: str) -> list[str]:
    candidates = []
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : index + 1])
                    break
        start = text.find("{", start + 1)
    return candidates


def _extract_native_tool_call(output_text: str) -> dict[str, Any] | None:
    if "avatar_speak" not in output_text:
        return None
    body_match = re.search(
        r"(?:<\|tool_call\>\s*)?call:avatar_speak\s*\{(?P<body>.*?)\}\s*(?:<tool_call\|>)?",
        output_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    body = body_match.group("body") if body_match else output_text
    text = _extract_text_field(body)
    if not text:
        return None
    emotion = _extract_emotion_field(body)
    return {
        "name": "avatar_speak",
        "parameters": {
            "text": text,
            "emotion_mode": emotion or "neutral",
        },
    }


def _extract_text_field(text: str) -> str | None:
    patterns = (
        r"\"text\"\s*:\s*\"(?P<value>(?:\\.|[^\"])*)\"",
        r"'text'\s*:\s*'(?P<value>(?:\\.|[^'])*)'",
        r"text\s*:\s*<\|\"\|>(?P<value>.*?)<\|\"\|>",
        r"\btext\b\s+[\"'](?P<value>.*?)[\"'](?:\s+(?:\"?emotion_mode\"?|emotion)\b|$)",
        r"\btext\b\s*[:=]\s*(?P<value>.+?)(?:\s*,\s*(?:\"?emotion_mode\"?|emotion)\b|\s*\}\s*$|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            value = match.group("value").strip()
            try:
                return json.loads(f'"{value}"')
            except json.JSONDecodeError:
                return value
    return None


def _extract_emotion_field(text: str) -> str | None:
    match = re.search(
        r"(?:\"?emotion_mode\"?|emotion)\s*[:=]\s*(?:<\|\"\|>)?\"?"
        r"(?P<value>neutral|engaged|emphatic|concerned)\"?(?:<\|\"\|>)?",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return match.group("value").lower()


# ---------------------------------------------------------------------------
# Streaming generation
# ---------------------------------------------------------------------------


def stream_gemma_with_early_tts(
    model,  # Llama instance from load_gemma()
    tokenizer,  # ignored for GGUF path — kept for API compatibility
    generation_kwargs: dict[str, Any],
    tts_queue: Queue,
    default_emotion: str = "neutral",
) -> str:
    """
    Stream token-by-token generation from llama-cpp-python and push completed
    sentences to the TTS queue as soon as they are ready.

    generation_kwargs expected keys:
        messages  — list of {"role": ..., "content": ...} dicts  (preferred)
        prompt    — raw string prompt (fallback if messages not present)
        max_tokens — int (default 512)
        temperature — float (default 0.7)
        top_p      — float (default 0.9)
    """
    messages = generation_kwargs.get("messages")
    max_tokens = generation_kwargs.get("max_tokens", 512)
    temperature = generation_kwargs.get("temperature", 0.7)
    top_p = generation_kwargs.get("top_p", 0.9)

    accumulated = ""
    full_output = ""

    if messages is not None:
        # Chat-completion streaming
        stream = model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        )
        for chunk in stream:
            delta = chunk["choices"][0]["delta"].get("content", "") or ""
            accumulated += delta
            full_output += delta
            sentence, remainder = split_first_sentence(accumulated)
            if sentence:
                tts_queue.put({"text": sentence, "emotion": default_emotion})
                accumulated = remainder
    else:
        # Raw completion streaming
        prompt = generation_kwargs.get("prompt", "")
        stream = model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        )
        for chunk in stream:
            token_text = chunk["choices"][0]["text"] or ""
            accumulated += token_text
            full_output += token_text
            sentence, remainder = split_first_sentence(accumulated)
            if sentence:
                tts_queue.put({"text": sentence, "emotion": default_emotion})
                accumulated = remainder

    if accumulated.strip():
        tts_queue.put({"text": accumulated.strip(), "emotion": default_emotion})

    return full_output


# ---------------------------------------------------------------------------
# Audio input stub  (audio handled by Whisper upstream; kept for API compat)
# ---------------------------------------------------------------------------


def build_audio_generation_inputs(
    processor,  # None for GGUF path
    prompt: str,
    audio_chunk,
    sample_rate: int = 16000,
    device=None,
) -> dict:
    """
    Stub for API compatibility.  With the GGUF backend, audio is transcribed
    upstream (Whisper) and passed in as text via the messages list.
    Returns a simple dict with the prompt so callers don't need to branch.
    """
    return {"prompt": prompt, "audio_chunk": audio_chunk, "sample_rate": sample_rate}
