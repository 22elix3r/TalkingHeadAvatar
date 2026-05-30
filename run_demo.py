#!/usr/bin/env python3
"""
run_demo.py — End-to-end local demo scaffold.

Current behavior:
1. Loads persona + optional Gemma.
2. Captures meeting audio and transcribes participant speech to text.
3. Accepts typed participant text from stdin as optional fallback.
4. Streams reply through TTS -> audio driver.
5. Renders one fixed-FPS avatar stream to the virtual camera.
"""

from __future__ import annotations

import argparse
from importlib import metadata as importlib_metadata
import json
import threading
from avatar_renderer import AvatarRenderer
import time
from pathlib import Path
from queue import Empty, Full, Queue

import numpy as np
import torch

from orchestrator import (
    MeetingAudioListener,
    MeetingSpeechRecognizer,
    MeetingTranscript,
    build_system_prompt,
    clean_avatar_speech_text,
    load_gemma,
    load_persona,
    process_gemma_output,
    split_first_sentence,
)
from runtime_animation import RuntimeAnimationConfig, RuntimeAnimationController
from virtual_camera import VirtualCameraOutput


def parse_args():
    parser = argparse.ArgumentParser(description="Live Avatar Meeting Demo")
    parser.add_argument(
        "--avatar_ckpt",
        required=True,
        help="Path to extracted avatar checkpoint directory",
    )
    parser.add_argument(
        "--audio_driver_ckpt",
        default=None,
        help="Path to trained MotionTranslator checkpoint (optional)",
    )
    parser.add_argument("--persona_file", required=True, help="Path to persona JSON")
    parser.add_argument(
        "--voice_ref",
        default=None,
        help="Path to voice reference WAV (optional; falls back to default TTS voice)",
    )
    parser.add_argument("--camera_device", default="/dev/video10")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--camera_width",
        type=int,
        default=None,
        help="Virtual camera output width. Defaults to --resolution.",
    )
    parser.add_argument(
        "--camera_height",
        type=int,
        default=None,
        help="Virtual camera output height. Defaults to --resolution.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--emotion_mode",
        default="neutral",
        choices=["neutral", "engaged", "emphatic", "concerned"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_gemma", action="store_true")
    parser.add_argument(
        "--gemma_gguf_path",
        default=None,
        help="Optional GGUF file or directory. Defaults to GEMMA_GGUF_PATH or the HF cache.",
    )
    parser.add_argument(
        "--gemma_n_gpu_layers",
        type=int,
        default=None,
        help="llama.cpp GPU layer count for Gemma (-1 = all layers).",
    )
    parser.add_argument(
        "--gemma_n_ctx",
        type=int,
        default=None,
        help="Gemma context window for llama.cpp.",
    )
    parser.add_argument(
        "--gemma_max_tokens",
        type=int,
        default=180,
        help="Maximum new tokens per avatar reply.",
    )
    parser.add_argument(
        "--gemma_temperature",
        type=float,
        default=0.8,
        help="Gemma sampling temperature.",
    )
    parser.add_argument(
        "--gemma_top_p",
        type=float,
        default=0.9,
        help="Gemma nucleus sampling value.",
    )
    parser.add_argument("--gemma_verbose", action="store_true")
    parser.add_argument("--skip_audio_listener", action="store_true")
    parser.add_argument(
        "--audio_input_device",
        default=None,
        help="Optional sounddevice input device index/name for the meeting microphone.",
    )
    parser.add_argument("--disable_stt", action="store_true")
    parser.add_argument(
        "--stt_model",
        default="openai/whisper-tiny.en",
        help="HF model id for ASR (default: openai/whisper-tiny.en)",
    )
    parser.add_argument(
        "--stt_language",
        default="en",
        help="Language code hint for ASR transcription (default: en)",
    )
    parser.add_argument(
        "--stt_min_audio_rms",
        type=float,
        default=0.003,
        help="Silence threshold for ASR chunk filtering",
    )
    parser.add_argument("--enable_stdin_fallback", action="store_true")
    parser.add_argument(
        "--video_only",
        action="store_true",
        help="Stream only the virtual camera feed (skip audio driver and TTS imports/init)",
    )
    parser.add_argument(
        "--play_audio",
        action="store_true",
        help="Play generated TTS audio to the default system output while also driving motion",
    )
    parser.add_argument(
        "--audio_output_device",
        default=None,
        help="Optional sounddevice output device index/name for --play_audio",
    )
    parser.add_argument(
        "--motion_chunk_ms",
        type=float,
        default=80.0,
        help="Audio chunk duration for HuBERT-driven motion updates (default: 80 ms)",
    )
    parser.add_argument(
        "--disable_motion_realtime_pacing",
        action="store_true",
        help="Consume TTS audio as fast as possible instead of pacing motion to audio time",
    )
    parser.add_argument(
        "--motion_expr_scale",
        type=float,
        default=1.0,
        help="Multiplier applied to predicted expression coefficients before rendering",
    )
    parser.add_argument(
        "--motion_jaw_scale",
        type=float,
        default=2.0,
        help="Multiplier applied to predicted jaw pose before rendering (default: 2.0)",
    )
    parser.add_argument(
        "--skip_audio_driver_warmup",
        action="store_true",
        help="Do not run a silent HuBERT warmup at startup",
    )
    parser.add_argument(
        "--disable_runtime_animation",
        action="store_true",
        help="Disable procedural head, gaze, blink, and micro-expression animation",
    )
    parser.add_argument(
        "--head_motion_scale",
        type=float,
        default=1.0,
        help="Runtime head/neck/translation motion multiplier",
    )
    parser.add_argument(
        "--eye_motion_scale",
        type=float,
        default=1.0,
        help="Runtime gaze and blink motion multiplier",
    )
    parser.add_argument(
        "--expression_runtime_scale",
        type=float,
        default=0.35,
        help="Runtime micro-expression multiplier",
    )
    parser.add_argument(
        "--idle_motion_scale",
        type=float,
        default=1.0,
        help="Runtime idle breathing/drift multiplier",
    )
    parser.add_argument(
        "--blink_rate",
        type=float,
        default=12.0,
        help="Approximate runtime blinks per minute",
    )
    parser.add_argument(
        "--preview_frame_path",
        default=None,
        help="Optional JPEG path for web/runtime preview frames",
    )
    parser.add_argument(
        "--preview_fps",
        type=float,
        default=10.0,
        help="Maximum preview JPEG update rate when --preview_frame_path is set",
    )
    parser.add_argument(
        "--control_state_path",
        default=None,
        help="Optional JSON control-state path for runtime controls such as mic mute.",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


class AvatarMotionState:
    """
    Thread-safe latest FLAME motion state shared by audio and render loops.

    Audio/TTS threads update this object. The render loop is the only code path
    that reads it and submits frames to the virtual camera.
    """

    def __init__(self, n_expr: int = 100, idle_hold_s: float = 0.25, idle_decay_s: float = 0.35):
        self._lock = threading.Lock()
        self._n_expr = int(n_expr)
        self._expr = np.zeros(self._n_expr, dtype=np.float32)
        self._jaw = np.zeros(3, dtype=np.float32)
        self._has_motion = False
        self._updated_at = 0.0
        self._idle_hold_s = float(idle_hold_s)
        self._idle_decay_s = float(idle_decay_s)
        self._emotion = "neutral"

    @staticmethod
    def _coerce_expr(expr: np.ndarray, n_expr: int) -> np.ndarray:
        arr = np.asarray(expr, dtype=np.float32).reshape(-1)
        if arr.shape[0] == n_expr:
            return arr.copy()
        out = np.zeros(n_expr, dtype=np.float32)
        n = min(n_expr, arr.shape[0])
        out[:n] = arr[:n]
        return out

    @staticmethod
    def _coerce_jaw(jaw: np.ndarray) -> np.ndarray:
        arr = np.asarray(jaw, dtype=np.float32).reshape(-1)
        out = np.zeros(3, dtype=np.float32)
        n = min(3, arr.shape[0])
        out[:n] = arr[:n]
        return out

    def configure_expression_dim(self, n_expr: int) -> None:
        n_expr = int(n_expr)
        with self._lock:
            if n_expr == self._n_expr:
                return
            self._expr = self._coerce_expr(self._expr, n_expr)
            self._n_expr = n_expr

    def update(self, expr: np.ndarray, jaw: np.ndarray) -> None:
        with self._lock:
            self._expr = self._coerce_expr(expr, self._n_expr)
            self._jaw = self._coerce_jaw(jaw)
            self._has_motion = True
            self._updated_at = time.monotonic()

    def set_emotion(self, emotion: str) -> None:
        with self._lock:
            self._emotion = str(emotion or "neutral")

    def emotion(self) -> str:
        with self._lock:
            return self._emotion

    def snapshot(self, n_expr: int | None = None) -> tuple[np.ndarray, np.ndarray, bool, float]:
        with self._lock:
            target_n_expr = self._n_expr if n_expr is None else int(n_expr)
            expr = self._coerce_expr(self._expr, target_n_expr)
            jaw = self._jaw.copy()
            has_motion = self._has_motion
            updated_at = self._updated_at

        age = time.monotonic() - updated_at if updated_at else float("inf")
        if has_motion and age > self._idle_hold_s:
            fade = max(0.0, 1.0 - ((age - self._idle_hold_s) / max(self._idle_decay_s, 1e-6)))
            expr *= fade
            jaw *= fade
            if fade <= 0.001:
                has_motion = False
        return expr, jaw, has_motion, age


class AudioPlaybackOutput:
    """Background 16 kHz mono float32 playback for generated TTS chunks."""

    def __init__(
        self,
        sample_rate: int = 16000,
        device: str | int | None = None,
        queue_maxsize: int = 50,
    ):
        self.sample_rate = sample_rate
        self.device = device
        self.audio_queue: Queue[np.ndarray] = Queue(maxsize=queue_maxsize)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self) -> None:
        import sounddevice as sd

        with sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device,
        ) as stream:
            print(
                f"[AudioPlaybackOutput] Opened output device={self.device or 'default'} "
                f"{self.sample_rate}Hz mono"
            )
            while not self._stop_event.is_set():
                try:
                    chunk = self.audio_queue.get(timeout=0.2)
                except Empty:
                    continue
                chunk = np.asarray(chunk, dtype=np.float32).reshape(-1, 1)
                stream.write(chunk)


class AudioFanoutQueue:
    """Queue-like sink that copies each generated TTS chunk to multiple queues."""

    def __init__(self, queues: list[Queue[np.ndarray]]):
        self.queues = queues

    def put_nowait(self, chunk: np.ndarray) -> None:
        for queue in self.queues:
            item = np.asarray(chunk, dtype=np.float32).copy()
            try:
                queue.put_nowait(item)
            except Full:
                try:
                    queue.get_nowait()
                except Empty:
                    pass
                queue.put_nowait(item)


class PreviewFrameWriter:
    """Writes throttled JPEG preview frames for the runtime web app."""

    def __init__(self, path: str | Path, width: int, height: int, fps: float = 10.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = int(width)
        self.height = int(height)
        self.min_interval = 1.0 / max(float(fps), 0.1)
        self._last_write = 0.0

    def write(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        if now - self._last_write < self.min_interval:
            return
        self._last_write = now

        from PIL import Image

        frame = np.asarray(frame, dtype=np.uint8)
        if frame.shape[:2] != (self.height, self.width):
            frame = self._letterbox(frame)

        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        Image.fromarray(frame, mode="RGB").save(tmp_path, format="JPEG", quality=85)
        tmp_path.replace(self.path)

    def _letterbox(self, frame: np.ndarray) -> np.ndarray:
        from PIL import Image

        src_h, src_w = frame.shape[:2]
        scale = min(self.width / src_w, self.height / src_h)
        out_w = max(1, int(round(src_w * scale)))
        out_h = max(1, int(round(src_h * scale)))
        image = Image.fromarray(frame, mode="RGB").resize((out_w, out_h), Image.Resampling.BILINEAR)

        canvas = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        x0 = (self.width - out_w) // 2
        y0 = (self.height - out_h) // 2
        canvas[y0 : y0 + out_h, x0 : x0 + out_w] = np.asarray(image, dtype=np.uint8)
        return canvas


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


def _decode_generated_text(tokenizer, output_ids: torch.Tensor, prompt_len: int) -> str:
    return tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()


class _TranscriptTokenizerAdapter:
    """
    Adapter to provide MeetingTranscript token counting for both HF and GGUF backends.
    """

    def __init__(self, hf_tokenizer, gguf_model):
        self.hf_tokenizer = hf_tokenizer
        self.gguf_model = gguf_model

    def encode(self, text: str, add_special_tokens: bool = False):
        if self.hf_tokenizer is not None:
            return self.hf_tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return self.gguf_model.tokenize(text.encode("utf-8"), add_bos=add_special_tokens)


def _build_orchestrator_prompt(system_prompt: str, context: str) -> str:
    return (
        f"{system_prompt}\n\n"
        f"Meeting transcript:\n{context}\n\n"
        "Reply only with the exact words the avatar should say aloud. "
        "Do not include avatar_speak, JSON, field names, code fences, labels, or metadata."
    )


def _generate_gemma_output(
    gemma_model,
    gemma_tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """
    Generate response text from either llama.cpp GGUF backend or HF transformers backend.
    """
    if hasattr(gemma_model, "create_chat_completion"):
        response = gemma_model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        message = response["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            function_call = tool_calls[0].get("function", {})
            arguments = function_call.get("arguments", "{}")
            parameters = json.loads(arguments) if isinstance(arguments, str) else arguments
            return json.dumps({"name": function_call.get("name"), "parameters": parameters})
        return (message.get("content") or "").strip()

    if gemma_tokenizer is None:
        raise RuntimeError("Gemma tokenizer is required for transformers backend.")
    device = _model_device(gemma_model)
    inputs = gemma_tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output_ids = gemma_model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=top_p,
    )
    return _decode_generated_text(
        tokenizer=gemma_tokenizer,
        output_ids=output_ids,
        prompt_len=inputs["input_ids"].shape[1],
    )


def _validate_avatar_checkpoint(path: str) -> Path:
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Avatar checkpoint path does not exist: {checkpoint}")
    if checkpoint.is_file():
        raise ValueError(
            "Avatar checkpoint must be an extracted directory. "
            f"Received file path: {checkpoint}"
        )
    point_cloud_dir = checkpoint / "point_cloud"
    if not point_cloud_dir.exists():
        raise ValueError(
            "Avatar checkpoint directory is missing 'point_cloud/'. "
            f"Expected under: {checkpoint}"
        )
    return checkpoint


def _audio_stack_load_error(exc: Exception) -> RuntimeError:
    torch_version = torch.__version__.split("+", maxsplit=1)[0]
    try:
        torchaudio_version = importlib_metadata.version("torchaudio")
    except importlib_metadata.PackageNotFoundError:
        torchaudio_version = "not-installed"

    return RuntimeError(
        "Failed to initialize audio stack (audio_driver/tts_engine).\n"
        f"Detected torch={torch_version}, torchaudio={torchaudio_version}.\n"
        "These packages must use matching major/minor versions.\n"
        f"Fix example: pip install --upgrade \"torch=={torch_version}\" \"torchaudio=={torch_version}\"\n"
        "Or run with --video_only to validate camera/OBS wiring without TTS."
    )


def _read_runtime_control_state(path: str | Path | None) -> dict:
    if not path:
        return {}
    try:
        state_path = Path(path).expanduser()
        if not state_path.is_absolute():
            state_path = Path.cwd() / state_path
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _runtime_mic_muted(path: str | Path | None) -> bool:
    return bool(_read_runtime_control_state(path).get("mic_muted", False))


def main():
    args = parse_args()
    StreamingAudioDriver = None
    ChatterboxEngine = None
    TTSAudioBridge = None
    if not args.video_only:
        try:
            from audio_driver.inference import StreamingAudioDriver as _StreamingAudioDriver
            from tts_engine import ChatterboxEngine as _ChatterboxEngine
            from tts_engine import TTSAudioBridge as _TTSAudioBridge
        except Exception as exc:
            raise _audio_stack_load_error(exc) from exc
        StreamingAudioDriver = _StreamingAudioDriver
        ChatterboxEngine = _ChatterboxEngine
        TTSAudioBridge = _TTSAudioBridge

    avatar_ckpt = _validate_avatar_checkpoint(args.avatar_ckpt)
    persona = load_persona(args.persona_file)
    system_prompt = build_system_prompt(persona)
    print(f"[0/6] Using avatar checkpoint: {avatar_ckpt}")

    print("[1/6] Loading Gemma...")
    gemma_model = None
    gemma_processor = None
    gemma_tokenizer = None
    transcript_tokenizer = None
    if not args.skip_gemma:
        gemma_model, gemma_processor = load_gemma(
            gguf_path=args.gemma_gguf_path,
            n_gpu_layers=args.gemma_n_gpu_layers,
            n_ctx=args.gemma_n_ctx,
            verbose=args.gemma_verbose,
            device_map="auto",
        )
        gemma_tokenizer = getattr(gemma_processor, "tokenizer", gemma_processor)
        transcript_tokenizer = _TranscriptTokenizerAdapter(
            hf_tokenizer=gemma_tokenizer,
            gguf_model=gemma_model,
        )

    if args.video_only:
        print("[2/6] Skipping audio driver (--video_only).")
        driver = None
    else:
        print("[2/6] Initializing audio driver...")
        driver = StreamingAudioDriver(
            motion_translator_ckpt=args.audio_driver_ckpt,
            device=args.device,
            fp16=args.device.startswith("cuda"),
        )
        if not args.skip_audio_driver_warmup:
            print("[2/6] Warming up audio driver...")
            driver.step_from_numpy(np.zeros(16000, dtype=np.float32))
            driver.reset()
            print("[2/6] Audio driver warmup complete.")
    # Initialize Renderer
    renderer = None
    if args.avatar_ckpt:
        try:
            renderer = AvatarRenderer(
                args.avatar_ckpt,
                resolution=args.resolution,
                device=args.device,
            )
        except Exception as e:
            print(f"Error initializing AvatarRenderer: {e}")
            print("Falling back to no-op rendering.")
    motion_state = AvatarMotionState(
        n_expr=getattr(renderer, "n_expr", 100) if renderer is not None else 100
    )
    motion_state.set_emotion(args.emotion_mode)
    animation_controller = None
    if renderer is not None and not args.disable_runtime_animation:
        animation_controller = RuntimeAnimationController(
            n_expr=renderer.n_expr,
            config=RuntimeAnimationConfig(
                enabled=True,
                fps=float(args.fps),
                head_motion_scale=args.head_motion_scale,
                eye_motion_scale=args.eye_motion_scale,
                expression_motion_scale=args.expression_runtime_scale,
                idle_motion_scale=args.idle_motion_scale,
                blink_rate_per_minute=args.blink_rate,
            ),
            expression_basis=renderer.expression_motion_basis(n_components=4),
        )
        print(
            "[runtime_animation] enabled "
            f"head={args.head_motion_scale:.2f} "
            f"eyes={args.eye_motion_scale:.2f} "
            f"expr={args.expression_runtime_scale:.2f} "
            f"idle={args.idle_motion_scale:.2f} "
            f"blink_rate={args.blink_rate:.1f}/min"
        )
    elif args.disable_runtime_animation:
        print("[runtime_animation] disabled")

    if args.video_only:
        print("[3/6] Skipping TTS engine (--video_only).")
        tts = None
    elif args.voice_ref:
        print("[3/6] Initializing TTS engine (voice reference)...")
        tts = ChatterboxEngine(voice_ref_path=args.voice_ref, device=args.device)
    else:
        print("[3/6] Initializing TTS engine (default voice)...")
        tts = ChatterboxEngine(voice_ref_path=args.voice_ref, device=args.device)

    print("[4/6] Initializing virtual camera...")
    if not args.dry_run:
        if not Path(args.camera_device).exists():
            raise FileNotFoundError(
                f"Camera device not found: {args.camera_device}\n"
                "Set up v4l2loopback first (see scripts/setup_v4l2loopback.sh).\n"
                "Recommended avatar sink: /dev/video10 (separate from OBS Virtual Camera /dev/video0)."
            )
    camera_width = args.camera_width or args.resolution
    camera_height = args.camera_height or args.resolution
    vcam = VirtualCameraOutput(
        width=camera_width,
        height=camera_height,
        fps=args.fps,
        device=args.camera_device,
    )
    preview_writer = (
        PreviewFrameWriter(
            path=args.preview_frame_path,
            width=camera_width,
            height=camera_height,
            fps=args.preview_fps,
        )
        if args.preview_frame_path
        else None
    )

    print("[5/6] Initializing audio listener...")
    listener = None
    if not args.skip_audio_listener:
        listener = MeetingAudioListener(input_device=args.audio_input_device)
        listener.start()

    speech_recognizer = None
    if listener is not None and gemma_model is not None and not args.disable_stt:
        print("[5/6] Initializing speech-to-text...")
        speech_recognizer = MeetingSpeechRecognizer(
            model_id=args.stt_model,
            device=args.device,
            language=args.stt_language,
            min_audio_rms=args.stt_min_audio_rms,
        )
        speech_recognizer.warmup()
    elif args.disable_stt:
        print("[5/6] STT disabled (--disable_stt).")

    text_queue: Queue[dict[str, str]] = Queue(maxsize=10)
    stop_event = threading.Event()

    motion_chunk_count = 0

    def consume_audio_chunk(chunk_16k: np.ndarray):
        nonlocal motion_chunk_count
        if driver is None:
            return
        expr, jaw = driver.step_from_numpy(chunk_16k)
        expr = np.clip(expr * args.motion_expr_scale, -5.0, 5.0)
        jaw = np.clip(jaw * args.motion_jaw_scale, -0.45, 0.45)
        motion_state.update(expr, jaw)
        motion_chunk_count += 1
        if motion_chunk_count == 1 or motion_chunk_count % 25 == 0:
            expr_rms = float(np.sqrt(np.mean(np.square(expr))))
            jaw_norm = float(np.linalg.norm(jaw))
            print(
                "[motion] "
                f"chunks={motion_chunk_count} "
                f"expr_rms={expr_rms:.5f} "
                f"jaw_norm={jaw_norm:.5f} "
                f"latency_ms={getattr(driver, 'latency_ms', 0.0):.1f}"
            )

    motion_chunk_size = max(320, int(round(16000 * args.motion_chunk_ms / 1000.0)))
    bridge = (
        TTSAudioBridge(
            audio_consumer=consume_audio_chunk,
            hubert_chunk_size=motion_chunk_size,
            sample_rate=16000,
            realtime=not args.disable_motion_realtime_pacing,
        )
        if TTSAudioBridge is not None
        else None
    )
    if bridge is not None:
        print(
            "[motion] "
            f"chunk_size={motion_chunk_size} samples "
            f"({motion_chunk_size / 16000 * 1000:.1f} ms), "
            f"realtime_pacing={not args.disable_motion_realtime_pacing}"
        )
    audio_playback = None
    if args.play_audio and not args.video_only:
        audio_playback = AudioPlaybackOutput(device=args.audio_output_device)

    tts_audio_sink = None
    if bridge is not None and audio_playback is not None:
        tts_audio_sink = AudioFanoutQueue([bridge.tts_audio_queue, audio_playback.audio_queue])
    elif bridge is not None:
        tts_audio_sink = bridge.tts_audio_queue

    transcript = MeetingTranscript()

    def handle_participant_text(participant_text: str):
        line = participant_text.strip()
        if not line:
            return

        if gemma_model is None or transcript_tokenizer is None:
            text_queue.put({"text": line, "emotion": args.emotion_mode})
            return

        transcript.add("Participant", line)
        transcript.truncate_to_last_n_minutes(30)
        context = transcript.to_context(transcript_tokenizer)
        prompt = _build_orchestrator_prompt(system_prompt, context)
        output_text = _generate_gemma_output(
            gemma_model=gemma_model,
            gemma_tokenizer=gemma_tokenizer,
            prompt=prompt,
            max_tokens=args.gemma_max_tokens,
            temperature=args.gemma_temperature,
            top_p=args.gemma_top_p,
        )
        enqueue_avatar_output(output_text)

    def enqueue_avatar_output(output_text: str):
        payload = process_gemma_output(output_text, default_emotion=args.emotion_mode)
        if payload is not None:
            payload["text"] = clean_avatar_speech_text(payload["text"])
            if not payload["text"]:
                return
            text_queue.put(payload)
            transcript.add("Avatar", payload["text"])
            return

        remaining = clean_avatar_speech_text(output_text)
        if not remaining:
            return
        transcript.add("Avatar", remaining)
        while remaining:
            sentence, remainder = split_first_sentence(remaining)
            if sentence:
                text_queue.put({"text": sentence, "emotion": args.emotion_mode})
                remaining = remainder.strip()
            else:
                text_queue.put({"text": remaining, "emotion": args.emotion_mode})
                break

    def tts_loop():
        while not stop_event.is_set():
            try:
                item = text_queue.get(timeout=0.2)
            except Empty:
                continue
            if tts is None or tts_audio_sink is None:
                continue
            spoken_text = clean_avatar_speech_text(item["text"])
            if not spoken_text:
                continue
            motion_state.set_emotion(item.get("emotion", args.emotion_mode))
            tts.synthesize_streaming(
                spoken_text,
                tts_audio_sink,
                emotion_mode=item.get("emotion", args.emotion_mode),
            )

    def orchestrator_loop():
        if listener is None or gemma_model is None or transcript_tokenizer is None:
            return
        if speech_recognizer is None:
            print(
                "[orchestrator] Speech-to-text is disabled or unavailable. "
                "Use --enable_stdin_fallback for manual prompt input."
            )
            return

        mic_was_muted = False
        while not stop_event.is_set():
            mic_is_muted = _runtime_mic_muted(args.control_state_path)
            if mic_is_muted:
                if not mic_was_muted:
                    print("[mic] muted - microphone input ignored.")
                mic_was_muted = True
                while True:
                    try:
                        listener.audio_queue.get_nowait()
                    except Empty:
                        break
                time.sleep(0.2)
                continue
            if mic_was_muted:
                print("[mic] unmuted - microphone input active.")
                mic_was_muted = False

            try:
                audio_chunk = listener.audio_queue.get(timeout=0.2)
            except Empty:
                continue

            participant_text = speech_recognizer.transcribe_chunk(
                audio_chunk=audio_chunk,
                sample_rate=listener.sr,
            )
            if not participant_text:
                continue
            print(f"[participant] {participant_text}")
            handle_participant_text(participant_text)

    def stdin_loop():
        while not stop_event.is_set():
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                continue

            handle_participant_text(line)

    tts_thread = threading.Thread(target=tts_loop, daemon=True)
    orchestrator_thread = threading.Thread(target=orchestrator_loop, daemon=True)
    stdin_thread = (
        threading.Thread(target=stdin_loop, daemon=True)
        if args.enable_stdin_fallback
        else None
    )

    if not args.dry_run:
        vcam.start()
    if bridge is not None:
        bridge.start()
    if audio_playback is not None:
        audio_playback.start()
    tts_thread.start()
    orchestrator_thread.start()
    if stdin_thread is not None:
        stdin_thread.start()

    print("[6/6] Ready.")
    if args.video_only:
        print("video_only enabled. Streaming neutral avatar frames to virtual camera.")
    if args.enable_stdin_fallback:
        if speech_recognizer is not None:
            print("Live STT enabled. stdin fallback also enabled (Ctrl+C to exit).")
        else:
            print("stdin fallback enabled. Type text and press Enter (Ctrl+C to exit).")
    else:
        if speech_recognizer is not None:
            print("Audio listener + STT loop active (Ctrl+C to exit).")
        else:
            print("Audio listener loop active without STT (Ctrl+C to exit).")
    try:
        print("Rendering fixed-FPS avatar loop...")
        frame_count = 0
        frame_interval = 1.0 / max(args.fps, 1)
        while not stop_event.is_set():
            frame_start = time.perf_counter()
            if renderer:
                expr, jaw, has_motion, _age = motion_state.snapshot(renderer.n_expr)
                if animation_controller is not None:
                    anim = animation_controller.step(
                        expression=expr,
                        jaw=jaw,
                        has_motion=has_motion,
                        emotion_mode=motion_state.emotion(),
                    )
                    frame = renderer.render(
                        anim.expression,
                        anim.jaw,
                        rotation_delta=anim.rotation_delta,
                        neck_delta=anim.neck_delta,
                        eyes_delta=anim.eyes_delta,
                        translation_delta=anim.translation_delta,
                    )
                elif has_motion:
                    frame = renderer.render(expr, jaw)
                else:
                    frame = renderer.render_idle()
                if preview_writer is not None:
                    preview_writer.write(frame)
                if not args.dry_run:
                    vcam.submit_frame(frame)
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"[orchestrator] Rendered {frame_count} avatar frames.")
            elapsed = time.perf_counter() - frame_start
            time.sleep(max(0.0, frame_interval - elapsed))
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_event.set()
        if bridge is not None:
            bridge.stop()
        if audio_playback is not None:
            audio_playback.stop()
        if listener is not None:
            listener.stop()
        if not args.dry_run:
            vcam.stop()
        time.sleep(0.2)


if __name__ == "__main__":
    main()
