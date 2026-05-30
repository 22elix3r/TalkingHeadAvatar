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
    AVATAR_TOOLS,
    MeetingAudioListener,
    MeetingSpeechRecognizer,
    MeetingTranscript,
    build_system_prompt,
    load_gemma,
    load_persona,
    process_gemma_output,
    split_first_sentence,
)
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
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--emotion_mode",
        default="neutral",
        choices=["neutral", "engaged", "emphatic", "concerned"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_gemma", action="store_true")
    parser.add_argument("--skip_audio_listener", action="store_true")
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
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


class AvatarMotionState:
    """
    Thread-safe latest FLAME motion state shared by audio and render loops.

    Audio/TTS threads update this object. The render loop is the only code path
    that reads it and submits frames to the virtual camera.
    """

    def __init__(self, n_expr: int = 100):
        self._lock = threading.Lock()
        self._n_expr = int(n_expr)
        self._expr = np.zeros(self._n_expr, dtype=np.float32)
        self._jaw = np.zeros(3, dtype=np.float32)
        self._has_motion = False
        self._updated_at = 0.0

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

    def snapshot(self, n_expr: int | None = None) -> tuple[np.ndarray, np.ndarray, bool, float]:
        with self._lock:
            target_n_expr = self._n_expr if n_expr is None else int(n_expr)
            expr = self._coerce_expr(self._expr, target_n_expr)
            jaw = self._jaw.copy()
            has_motion = self._has_motion
            updated_at = self._updated_at

        age = time.monotonic() - updated_at if updated_at else float("inf")
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
        f"When appropriate, call avatar_speak with JSON matching this tool schema:\n{AVATAR_TOOLS[0]}"
    )


def _generate_gemma_output(gemma_model, gemma_tokenizer, prompt: str) -> str:
    """
    Generate response text from either llama.cpp GGUF backend or HF transformers backend.
    """
    if hasattr(gemma_model, "create_chat_completion"):
        response = gemma_model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=160,
            temperature=0.7,
            top_p=0.9,
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
        max_new_tokens=160,
        temperature=0.7,
        do_sample=True,
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
        gemma_model, gemma_processor = load_gemma(device_map="auto")
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
    vcam = VirtualCameraOutput(
        width=args.resolution,
        height=args.resolution,
        fps=args.fps,
        device=args.camera_device,
    )

    print("[5/6] Initializing audio listener...")
    listener = None
    if not args.skip_audio_listener:
        listener = MeetingAudioListener()
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

    def consume_audio_chunk(chunk_16k: np.ndarray):
        if driver is None:
            return
        expr, jaw = driver.step_from_numpy(chunk_16k)
        motion_state.update(expr, jaw)

    bridge = (
        TTSAudioBridge(audio_consumer=consume_audio_chunk, hubert_chunk_size=3200)
        if TTSAudioBridge is not None
        else None
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
        )
        enqueue_avatar_output(output_text)

    def enqueue_avatar_output(output_text: str):
        payload = process_gemma_output(output_text, default_emotion=args.emotion_mode)
        if payload is not None:
            text_queue.put(payload)
            transcript.add("Avatar", payload["text"])
            return

        remaining = output_text.strip()
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
            tts.synthesize_streaming(
                item["text"],
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

        while not stop_event.is_set():
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
                if has_motion:
                    frame = renderer.render(expr, jaw)
                else:
                    frame = renderer.render_idle()
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
