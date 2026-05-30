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
from typing import Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Path to the downloaded GGUF — unsloth quant in HF cache.
# Override via env var GEMMA_GGUF_PATH if needed.
_HF_CACHE_GGUF = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--unsloth--gemma-4-E4B-it-GGUF"
    / "snapshots/ce152932ac27bc40bc9c727386760424d50bb456"
    / "gemma-4-E4B-it-Q4_K_M.gguf"
)
DEFAULT_GGUF_PATH = _HF_CACHE_GGUF
GEMMA_GGUF_PATH: Path = Path(os.environ.get("GEMMA_GGUF_PATH", DEFAULT_GGUF_PATH))

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

    path = Path(gguf_path or GEMMA_GGUF_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"GGUF model not found at {path}\n"
            f"Expected: ~/.cache/huggingface/hub/models--unsloth--gemma-4-E4B-it-GGUF/.../gemma-4-E4B-it-Q4_K_M.gguf\n"
            f"Or set the GEMMA_GGUF_PATH environment variable to an alternative path."
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
    call_payload = None
    if "<tool_call>" in output_text:
        start = output_text.find("<tool_call>") + len("<tool_call>")
        end = output_text.find("</tool_call>", start)
        if end != -1:
            call_payload = output_text[start:end].strip()
    elif '"name"' in output_text and "avatar_speak" in output_text:
        brace_start = output_text.find("{")
        brace_end = output_text.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            call_payload = output_text[brace_start : brace_end + 1]

    if call_payload is None:
        return None

    call = json.loads(call_payload)
    if call.get("name") != "avatar_speak":
        return None
    params = call.get("parameters", {})
    text = params.get("text", "").strip()
    if not text:
        return None
    emotion = params.get("emotion_mode", default_emotion)
    return {"text": text, "emotion": emotion}


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
