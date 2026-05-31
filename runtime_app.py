#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


PROJECT_ROOT = Path(__file__).resolve().parent
HF_GEMMA_CACHE = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--unsloth--gemma-4-E4B-it-GGUF"
)
GEMMA_QUANT_PREFERENCE = (
    "Q4_K_M",
    "UD-Q4_K_XL",
    "Q4_K_XL",
    "Q5_K_M",
    "Q6_K",
    "Q8_0",
    "Q3_K_M",
)


def _default_gemma_gguf_path() -> str:
    explicit = os.environ.get("GEMMA_GGUF_PATH")
    if explicit:
        return explicit

    candidates: list[Path] = []
    refs_main = HF_GEMMA_CACHE / "refs/main"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        if revision:
            snapshot = HF_GEMMA_CACHE / "snapshots" / revision
            if snapshot.exists():
                candidates.extend(sorted(snapshot.rglob("*.gguf")))

    snapshots = HF_GEMMA_CACHE / "snapshots"
    if snapshots.exists():
        candidates.extend(sorted(snapshots.rglob("*.gguf")))

    existing = [path for path in candidates if path.exists() and path.is_file()]
    if not existing:
        return ""

    def score(path: Path) -> tuple[int, str]:
        for index, marker in enumerate(GEMMA_QUANT_PREFERENCE):
            if marker in path.name:
                return index, path.name
        return len(GEMMA_QUANT_PREFERENCE), path.name

    return str(min(existing, key=score))


COMMON_DEFAULT_CONFIG = {
    "profile": "new_speaker_pilot",
    "session_name": "new_speaker_pilot",
    "gpu": "0",
    "camera_device": "/dev/video10",
    "resolution": 512,
    "camera_width": 640,
    "camera_height": 480,
    "fps": 30,
    "device": "cuda",
    "emotion_mode": "neutral",
    "skip_gemma": False,
    "gemma_gguf_path": _default_gemma_gguf_path(),
    "gemma_n_gpu_layers": -1,
    "gemma_n_ctx": 4096,
    "gemma_max_tokens": 180,
    "gemma_temperature": 0.8,
    "gemma_top_p": 0.9,
    "gemma_verbose": False,
    "skip_audio_listener": False,
    "audio_input_device": "",
    "audio_output_device": "",
    "audio_playback_delay_ms": 0.0,
    "disable_stt": False,
    "stt_model": "openai/whisper-tiny.en",
    "stt_language": "en",
    "stt_min_audio_rms": 0.003,
    "enable_stdin_fallback": True,
    "play_audio": True,
    "motion_chunk_ms": 80,
    "audio_driver_ema": 0.3,
    "audio_context_seconds": 1.0,
    "motion_delay_ms": 0.0,
    "motion_expr_scale": 1.0,
    "motion_jaw_scale": 2.0,
    "learned_pose_scale": 1.0,
    "learned_neck_scale": 1.0,
    "learned_eye_scale": 1.0,
    "learned_translation_scale": 1.0,
    "head_motion_scale": 1.35,
    "eye_motion_scale": 1.0,
    "expression_runtime_scale": 0.45,
    "idle_motion_scale": 1.0,
    "blink_rate": 12,
    "disable_runtime_animation": False,
    "preview_fps": 10,
}

RUNTIME_PROFILES = {
    "new_speaker_pilot": {
        "label": "New Speaker Pilot",
        "session_name": "new_speaker_pilot",
        "avatar_ckpt": "output/new_speaker/ga_pilot",
        "audio_driver_ckpt": "audio_driver/checkpoints/new_speaker/best_model.pt",
        "persona_file": "data/new_speaker/persona.json",
        "voice_ref": "data/new_speaker/voice_reference.wav",
        "preview_frame_path": "output/new_speaker/runtime_app/preview.jpg",
        "control_state_path": "output/new_speaker/runtime_app/control_state.json",
        "log_path": "output/new_speaker/logs/runtime_pilot.log",
        "motion_chunk_ms": 40,
        "audio_driver_ema": 0.15,
        "audio_context_seconds": 0.8,
        "motion_delay_ms": 80,
        "motion_jaw_scale": 2.0,
        "motion_expr_scale": 1.0,
        "learned_pose_scale": 1.0,
        "learned_neck_scale": 1.0,
        "learned_eye_scale": 1.0,
        "learned_translation_scale": 1.0,
        "head_motion_scale": 1.25,
        "eye_motion_scale": 1.0,
        "expression_runtime_scale": 0.4,
        "idle_motion_scale": 1.0,
        "blink_rate": 12,
    },
    "sam_altman": {
        "label": "Sam Altman",
        "session_name": "e2e_demo",
        "avatar_ckpt": "output/sam_altman/ga_phase4_recovered_300k",
        "audio_driver_ckpt": "audio_driver/checkpoints/sam_altman_phase5/best_model.pt",
        "persona_file": "data/sam_altman/persona.json",
        "voice_ref": "data/sam_altman/voice_reference.wav",
        "preview_frame_path": "output/sam_altman/runtime_app/preview.jpg",
        "control_state_path": "output/sam_altman/runtime_app/control_state.json",
        "log_path": "output/sam_altman/logs/e2e_demo.log",
        "motion_jaw_scale": 2.0,
        "motion_expr_scale": 1.0,
        "learned_pose_scale": 1.0,
        "learned_neck_scale": 1.0,
        "learned_eye_scale": 1.0,
        "learned_translation_scale": 1.0,
        "head_motion_scale": 1.35,
        "eye_motion_scale": 1.0,
        "expression_runtime_scale": 0.45,
        "idle_motion_scale": 1.0,
        "blink_rate": 12,
    },
}


def _default_config_for_profile(profile: str = "new_speaker_pilot") -> dict:
    config = COMMON_DEFAULT_CONFIG.copy()
    selected = RUNTIME_PROFILES.get(profile) or RUNTIME_PROFILES["new_speaker_pilot"]
    config.update({key: value for key, value in selected.items() if key != "label"})
    config["profile"] = profile if profile in RUNTIME_PROFILES else "new_speaker_pilot"
    return config


DEFAULT_CONFIG = _default_config_for_profile("new_speaker_pilot")

DEFAULT_CONTROL_STATE = {
    "mic_muted": False,
}


BOOL_FIELDS = {
    "skip_gemma",
    "gemma_verbose",
    "skip_audio_listener",
    "disable_stt",
    "enable_stdin_fallback",
    "play_audio",
    "disable_runtime_animation",
}

INT_FIELDS = {
    "resolution",
    "camera_width",
    "camera_height",
    "fps",
    "gemma_n_gpu_layers",
    "gemma_n_ctx",
    "gemma_max_tokens",
}
FLOAT_FIELDS = {
    "gemma_temperature",
    "gemma_top_p",
    "stt_min_audio_rms",
    "audio_playback_delay_ms",
    "motion_chunk_ms",
    "audio_driver_ema",
    "audio_context_seconds",
    "motion_delay_ms",
    "motion_expr_scale",
    "motion_jaw_scale",
    "learned_pose_scale",
    "learned_neck_scale",
    "learned_eye_scale",
    "learned_translation_scale",
    "head_motion_scale",
    "eye_motion_scale",
    "expression_runtime_scale",
    "idle_motion_scale",
    "blink_rate",
    "preview_fps",
}


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _json_response(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _text_response(
    handler: BaseHTTPRequestHandler,
    body: str,
    content_type: str = "text/html; charset=utf-8",
    status: int = 200,
) -> None:
    data = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _read_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0") or 0)
    raw = handler.rfile.read(length) if length else b"{}"
    if not raw:
        return {}
    content_type = handler.headers.get("Content-Type", "")
    if "application/json" in content_type:
        return json.loads(raw.decode("utf-8"))
    return {k: v[-1] for k, v in parse_qs(raw.decode("utf-8")).items()}


def _tail_file(path: Path, max_bytes: int = 24000) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > max_bytes:
            f.seek(-max_bytes, os.SEEK_END)
        data = f.read()
    return data.decode("utf-8", errors="replace")


def _read_json_file(path: Path, default: dict) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default.copy()


def _write_json_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


class RuntimeDemoManager:
    def __init__(self):
        self.last_config = DEFAULT_CONFIG.copy()

    def tmux_exists(self, session_name: str) -> bool:
        result = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0

    def start(self, config: dict) -> dict:
        config = self._coerce_config(config)
        session = config["session_name"]
        if self.tmux_exists(session):
            return {"ok": False, "message": f"tmux session already running: {session}"}

        log_path = _project_path(config["log_path"])
        preview_path = _project_path(config["preview_frame_path"])
        control_path = _project_path(config["control_state_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        control_path.parent.mkdir(parents=True, exist_ok=True)
        if not control_path.exists():
            _write_json_file(control_path, DEFAULT_CONTROL_STATE)

        cmd = self._build_demo_command(config)
        result = subprocess.run(
            ["tmux", "new-session", "-d", "-s", session, cmd],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return {"ok": False, "message": result.stderr.strip() or result.stdout.strip()}

        self.last_config = config
        return {"ok": True, "message": f"started {session}", "command": cmd}

    def stop(self, session_name: str | None = None) -> dict:
        session = session_name or self.last_config["session_name"]
        if not self.tmux_exists(session):
            return {"ok": True, "message": f"not running: {session}"}
        result = subprocess.run(
            ["tmux", "kill-session", "-t", session],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return {"ok": False, "message": result.stderr.strip() or result.stdout.strip()}
        return {"ok": True, "message": f"stopped {session}"}

    def restart(self, config: dict) -> dict:
        config = self._coerce_config(config)
        stopped = self.stop(config["session_name"])
        if not stopped["ok"]:
            return stopped
        time.sleep(0.4)
        return self.start(config)

    def send_prompt(self, text: str, session_name: str | None = None) -> dict:
        session = session_name or self.last_config["session_name"]
        text = str(text or "").strip()
        if not text:
            return {"ok": False, "message": "empty prompt"}
        if not self.tmux_exists(session):
            return {"ok": False, "message": f"tmux session is not running: {session}"}
        result = subprocess.run(
            ["tmux", "send-keys", "-t", session, text, "C-m"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return {"ok": False, "message": result.stderr.strip() or result.stdout.strip()}
        return {"ok": True, "message": "prompt sent"}

    def control_state(self, config: dict | None = None) -> dict:
        current_config = self._coerce_config(config or self.last_config)
        control_path = _project_path(current_config["control_state_path"])
        state = DEFAULT_CONTROL_STATE.copy()
        state.update(_read_json_file(control_path, DEFAULT_CONTROL_STATE))
        return state

    def set_mic_muted(self, muted: bool, config: dict | None = None) -> dict:
        current_config = self._coerce_config(config or self.last_config)
        control_path = _project_path(current_config["control_state_path"])
        state = self.control_state(current_config)
        state["mic_muted"] = bool(muted)
        state["updated_at"] = time.time()
        _write_json_file(control_path, state)
        self.last_config = current_config
        status = "muted" if state["mic_muted"] else "unmuted"
        return {"ok": True, "message": f"microphone {status}", "control": state}

    def status(self) -> dict:
        config = self.last_config
        session = config["session_name"]
        log_path = _project_path(config["log_path"])
        preview_path = _project_path(config["preview_frame_path"])
        control_path = _project_path(config["control_state_path"])
        preview_mtime = preview_path.stat().st_mtime if preview_path.exists() else None
        return {
            "running": self.tmux_exists(session),
            "session_name": session,
            "log_path": str(log_path),
            "preview_path": str(preview_path),
            "preview_mtime": preview_mtime,
            "control_path": str(control_path),
            "control": self.control_state(config),
            "config": config,
            "logs": _tail_file(log_path),
        }

    def _coerce_config(self, config: dict) -> dict:
        requested = config or {}
        profile = str(requested.get("profile") or DEFAULT_CONFIG["profile"])
        merged = _default_config_for_profile(profile)
        merged.update(config or {})
        if merged.get("profile") not in RUNTIME_PROFILES:
            merged["profile"] = DEFAULT_CONFIG["profile"]
        for key in BOOL_FIELDS:
            value = merged.get(key)
            if isinstance(value, str):
                merged[key] = value.lower() in {"1", "true", "yes", "on"}
            else:
                merged[key] = bool(value)
        for key in INT_FIELDS:
            merged[key] = int(merged[key])
        for key in FLOAT_FIELDS:
            merged[key] = float(merged[key])
        return merged

    def _build_demo_command(self, config: dict) -> str:
        log_path = _project_path(config["log_path"])
        argv = [
            "python",
            "run_demo.py",
            "--avatar_ckpt",
            config["avatar_ckpt"],
            "--audio_driver_ckpt",
            config["audio_driver_ckpt"],
            "--persona_file",
            config["persona_file"],
            "--voice_ref",
            config["voice_ref"],
            "--camera_device",
            config["camera_device"],
            "--resolution",
            str(config["resolution"]),
            "--camera_width",
            str(config["camera_width"]),
            "--camera_height",
            str(config["camera_height"]),
            "--fps",
            str(config["fps"]),
            "--device",
            config["device"],
            "--emotion_mode",
            config["emotion_mode"],
            "--gemma_gguf_path",
            config["gemma_gguf_path"],
            "--gemma_n_gpu_layers",
            str(config["gemma_n_gpu_layers"]),
            "--gemma_n_ctx",
            str(config["gemma_n_ctx"]),
            "--gemma_max_tokens",
            str(config["gemma_max_tokens"]),
            "--gemma_temperature",
            str(config["gemma_temperature"]),
            "--gemma_top_p",
            str(config["gemma_top_p"]),
            "--stt_model",
            config["stt_model"],
            "--stt_language",
            config["stt_language"],
            "--stt_min_audio_rms",
            str(config["stt_min_audio_rms"]),
            "--audio_playback_delay_ms",
            str(config["audio_playback_delay_ms"]),
            "--motion_chunk_ms",
            str(config["motion_chunk_ms"]),
            "--audio_driver_ema",
            str(config["audio_driver_ema"]),
            "--audio_context_seconds",
            str(config["audio_context_seconds"]),
            "--motion_delay_ms",
            str(config["motion_delay_ms"]),
            "--motion_expr_scale",
            str(config["motion_expr_scale"]),
            "--motion_jaw_scale",
            str(config["motion_jaw_scale"]),
            "--learned_pose_scale",
            str(config["learned_pose_scale"]),
            "--learned_neck_scale",
            str(config["learned_neck_scale"]),
            "--learned_eye_scale",
            str(config["learned_eye_scale"]),
            "--learned_translation_scale",
            str(config["learned_translation_scale"]),
            "--head_motion_scale",
            str(config["head_motion_scale"]),
            "--eye_motion_scale",
            str(config["eye_motion_scale"]),
            "--expression_runtime_scale",
            str(config["expression_runtime_scale"]),
            "--idle_motion_scale",
            str(config["idle_motion_scale"]),
            "--blink_rate",
            str(config["blink_rate"]),
            "--preview_frame_path",
            config["preview_frame_path"],
            "--preview_fps",
            str(config["preview_fps"]),
            "--control_state_path",
            config["control_state_path"],
        ]
        if config["skip_gemma"]:
            argv.append("--skip_gemma")
        if config["gemma_verbose"]:
            argv.append("--gemma_verbose")
        if config["skip_audio_listener"]:
            argv.append("--skip_audio_listener")
        if config["audio_input_device"]:
            argv.extend(["--audio_input_device", config["audio_input_device"]])
        if config["disable_stt"]:
            argv.append("--disable_stt")
        if config["enable_stdin_fallback"]:
            argv.append("--enable_stdin_fallback")
        if config["play_audio"]:
            argv.append("--play_audio")
        if config["audio_output_device"]:
            argv.extend(["--audio_output_device", config["audio_output_device"]])
        if config["disable_runtime_animation"]:
            argv.append("--disable_runtime_animation")

        python_cmd = " ".join(shlex.quote(part) for part in argv)
        script = (
            f"cd {shlex.quote(str(PROJECT_ROOT))} && "
            "source activate_env.sh && "
            f"CUDA_VISIBLE_DEVICES={shlex.quote(str(config['gpu']))} "
            f"PYTHONUNBUFFERED=1 {python_cmd} "
            f"2>&1 | tee -a {shlex.quote(str(log_path))}"
        )
        return f"bash -lc {shlex.quote(script)}"


MANAGER = RuntimeDemoManager()


class RuntimeAppHandler(BaseHTTPRequestHandler):
    server_version = "TalkingHeadRuntimeApp/1.0"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[runtime_app] {self.address_string()} - {fmt % args}")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            _text_response(self, self._html())
            return
        if path == "/api/status":
            _json_response(self, MANAGER.status())
            return
        if path == "/api/config":
            _json_response(self, {"default": DEFAULT_CONFIG, "profiles": RUNTIME_PROFILES})
            return
        if path == "/preview.jpg":
            self._serve_preview_jpeg()
            return
        if path == "/preview.mjpg":
            self._serve_preview_mjpeg()
            return
        _json_response(self, {"ok": False, "message": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = _read_body(self)
            if path == "/api/start":
                _json_response(self, MANAGER.start(payload))
                return
            if path == "/api/stop":
                _json_response(self, MANAGER.stop(payload.get("session_name")))
                return
            if path == "/api/restart":
                _json_response(self, MANAGER.restart(payload))
                return
            if path == "/api/prompt":
                _json_response(self, MANAGER.send_prompt(payload.get("text", ""), payload.get("session_name")))
                return
            if path == "/api/mic":
                muted = bool(payload.get("muted", False))
                _json_response(self, MANAGER.set_mic_muted(muted, payload.get("config")))
                return
        except Exception as exc:
            _json_response(self, {"ok": False, "message": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        _json_response(self, {"ok": False, "message": "not found"}, HTTPStatus.NOT_FOUND)

    def _serve_preview_jpeg(self) -> None:
        preview_path = _project_path(MANAGER.last_config["preview_frame_path"])
        if not preview_path.exists():
            _text_response(self, "preview not available yet", "text/plain", HTTPStatus.NOT_FOUND)
            return
        data = preview_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_preview_mjpeg(self) -> None:
        preview_path = _project_path(MANAGER.last_config["preview_frame_path"])
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last_mtime = None
        try:
            while True:
                if preview_path.exists():
                    mtime = preview_path.stat().st_mtime
                    if mtime != last_mtime:
                        last_mtime = mtime
                        data = preview_path.read_bytes()
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(data)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _html(self) -> str:
        config_json = json.dumps(DEFAULT_CONFIG)
        profiles_json = json.dumps(RUNTIME_PROFILES)
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Talking Head Runtime</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0f1215;
      --surface: #151a1f;
      --surface-2: #1d242a;
      --surface-3: #242c33;
      --line: #303a43;
      --line-2: #3c4852;
      --text: #e8edf2;
      --muted: #9da8b3;
      --subtle: #73808c;
      --accent: #2f9fe8;
      --accent-2: #1c6ea6;
      --ok: #39bd78;
      --bad: #ef6868;
      --warn: #e5b448;
      --radius: 8px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      height: 58px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px 0 20px;
      border-bottom: 1px solid var(--line);
      background: #11161a;
    }}
    h1 {{ margin: 0; font-size: 18px; font-weight: 700; }}
    .top-status {{ display: flex; gap: 10px; align-items: center; }}
    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
      min-height: 28px;
      padding: 4px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--surface);
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }}
    .chip.live {{ color: #dff9ec; border-color: rgba(57, 189, 120, 0.55); }}
    .chip.muted {{ color: #ffe2e2; border-color: rgba(239, 104, 104, 0.55); }}
    .dot {{ width: 8px; height: 8px; border-radius: 50%; background: var(--bad); }}
    .dot.running {{ background: var(--ok); }}
    main {{
      display: grid;
      grid-template-columns: 360px minmax(560px, 1fr) 430px;
      gap: 14px;
      padding: 16px;
      height: calc(100vh - 58px);
      min-height: 0;
    }}
    section {{
      min-width: 0;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      overflow: hidden;
    }}
    .left, .center, .right {{
      display: flex;
      flex-direction: column;
      gap: 14px;
      min-height: 0;
    }}
    .panel-title {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      min-height: 38px;
      padding: 9px 12px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .panel-body {{ padding: 12px; }}
    .stack {{ display: flex; flex-direction: column; gap: 10px; }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px 12px;
    }}
    .span-2 {{ grid-column: span 2; }}
    label {{ display: flex; flex-direction: column; gap: 5px; color: var(--muted); font-size: 12px; min-width: 0; }}
    input, select, textarea {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #0b0f12;
      color: var(--text);
      padding: 8px 9px;
      font: inherit;
      min-height: 36px;
    }}
    textarea {{ resize: vertical; min-height: 96px; }}
    input[type="checkbox"] {{ width: auto; }}
    .checkrow {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }}
    .checkrow label {{ flex-direction: row; align-items: center; color: var(--text); }}
    button {{
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--surface-2);
      color: var(--text);
      padding: 9px 12px;
      font-weight: 650;
      cursor: pointer;
      min-height: 38px;
    }}
    button.primary {{ background: var(--accent-2); border-color: var(--accent); }}
    button.danger {{ background: #743333; border-color: #9c4141; }}
    button.good {{ background: #1d6944; border-color: #2e9f68; }}
    button.warn {{ background: #7c4d19; border-color: #b87321; }}
    button.full {{ width: 100%; }}
    button:hover {{ filter: brightness(1.12); }}
    .actions {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #11161a;
      min-height: 58px;
    }}
    .metric .k {{
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      font-weight: 700;
    }}
    .metric .v {{ margin-top: 4px; font-size: 13px; color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .video-wrap {{
      display: flex;
      align-items: center;
      justify-content: center;
      background: #050607;
      height: min(64vh, calc(100vh - 300px));
      min-height: 420px;
    }}
    .video-wrap img {{
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      background: #0b0d0f;
    }}
    .preview-meta {{ color: var(--subtle); font-size: 12px; text-transform: none; letter-spacing: 0; font-weight: 600; }}
    .prompt-row {{ display: grid; grid-template-columns: 1fr; gap: 8px; }}
    pre {{
      margin: 0;
      padding: 12px;
      flex: 1;
      min-height: 190px;
      overflow: auto;
      background: #080a0c;
      color: #c9d3dd;
      white-space: pre-wrap;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    details {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #11161a;
      overflow: hidden;
    }}
    summary {{
      cursor: pointer;
      padding: 10px 12px;
      color: var(--text);
      font-weight: 700;
      border-bottom: 1px solid transparent;
    }}
    details[open] summary {{ border-bottom-color: var(--line); }}
    details .details-body {{ padding: 12px; }}
    .hint {{ color: var(--muted); font-size: 12px; min-height: 18px; }}
    .log-section {{ display: flex; flex-direction: column; min-height: 0; flex: 1; }}
    .config-scroll {{ overflow: auto; min-height: 0; }}
    .path-input {{ font-size: 12px; }}
    @media (max-width: 1260px) {{
      main {{ grid-template-columns: 340px minmax(500px, 1fr); }}
      .right {{ grid-column: 1 / -1; }}
    }}
    @media (max-width: 860px) {{
      main {{ grid-template-columns: 1fr; height: auto; }}
      .video-wrap {{ min-height: 320px; height: 46vh; }}
      .actions {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Talking Head Runtime</h1>
    <div class="top-status">
      <div id="micChip" class="chip"><span class="dot"></span><span id="micChipText">Mic unknown</span></div>
      <div id="runChip" class="chip"><span id="statusDot" class="dot"></span><span id="statusText">Checking</span></div>
    </div>
  </header>
  <main>
    <div class="left">
      <section>
        <div class="panel-title">Session</div>
        <div class="panel-body">
          <div class="actions">
            <button class="primary" onclick="startRun()">Start</button>
            <button onclick="restartRun()">Restart</button>
            <button class="danger" onclick="stopRun()">Stop</button>
          </div>
          <div class="metric-grid" style="margin-top: 10px;">
            <div class="metric"><div class="k">Session</div><div id="sessionMetric" class="v">-</div></div>
            <div class="metric"><div class="k">Device</div><div id="deviceMetric" class="v">-</div></div>
            <div class="metric"><div class="k">Camera</div><div id="cameraMetric" class="v">-</div></div>
            <div class="metric"><div class="k">Preview</div><div id="previewMetric" class="v">-</div></div>
          </div>
          <div class="hint" id="message"></div>
        </div>
      </section>
      <section>
        <div class="panel-title">Live Input</div>
        <div class="panel-body">
          <div class="stack">
            <button id="micToggle" class="good full" onclick="toggleMic()">Mic On</button>
            <div class="prompt-row">
              <textarea id="prompt" rows="4" placeholder="Typed turn for the avatar"></textarea>
              <button onclick="sendPrompt()">Send Text</button>
            </div>
          </div>
        </div>
      </section>
      <section>
        <div class="panel-title">Fast Controls</div>
        <div class="panel-body">
          <div class="prompt-row">
            <label>Emotion
              <select id="emotion_mode">
                <option>neutral</option><option>engaged</option><option>emphatic</option><option>concerned</option>
              </select>
            </label>
            <div class="grid">
              <label>Jaw <input id="motion_jaw_scale" type="number" step="0.05"></label>
              <label>Expr <input id="motion_expr_scale" type="number" step="0.05"></label>
              <label>Learned Head <input id="learned_pose_scale" type="number" step="0.05"></label>
              <label>Learned Neck <input id="learned_neck_scale" type="number" step="0.05"></label>
              <label>Learned Eyes <input id="learned_eye_scale" type="number" step="0.05"></label>
              <label>Learned Move <input id="learned_translation_scale" type="number" step="0.05"></label>
              <label>Head <input id="head_motion_scale" type="number" step="0.05"></label>
              <label>Eyes <input id="eye_motion_scale" type="number" step="0.05"></label>
              <label>Runtime Expr <input id="expression_runtime_scale" type="number" step="0.05"></label>
              <label>Blinks/min <input id="blink_rate" type="number" step="1"></label>
              <label>Chunk ms <input id="motion_chunk_ms" type="number" step="1"></label>
              <label>Motion Delay ms <input id="motion_delay_ms" type="number" step="10"></label>
              <label>Audio Delay ms <input id="audio_playback_delay_ms" type="number" step="10"></label>
              <label>Driver EMA <input id="audio_driver_ema" type="number" step="0.05"></label>
              <label>Audio Context s <input id="audio_context_seconds" type="number" step="0.1"></label>
              <label>Idle <input id="idle_motion_scale" type="number" step="0.05"></label>
            </div>
          </div>
        </div>
      </section>
    </div>
    <div class="center">
      <section>
        <div class="panel-title"><span>Output Preview</span><span id="previewMeta" class="preview-meta">-</span></div>
        <div class="video-wrap">
          <img id="preview" src="/preview.mjpg" alt="Runtime preview">
        </div>
      </section>
      <section class="log-section">
        <div class="panel-title"><span>Logs</span><button onclick="refreshStatus()">Refresh</button></div>
        <pre id="logs"></pre>
      </section>
    </div>
    <div class="right">
      <section class="config-scroll">
        <div class="panel-title">Configuration</div>
        <div class="panel-body stack">
          <details open>
            <summary>Runtime</summary>
            <div class="details-body grid">
              <label class="span-2">Profile <select id="profile"></select></label>
              <label>Session <input id="session_name"></label>
              <label>GPU <input id="gpu"></label>
              <label>Device <select id="device"><option>cuda</option><option>cpu</option></select></label>
              <label>FPS <input id="fps" type="number"></label>
              <label>Resolution <input id="resolution" type="number"></label>
              <label>Preview FPS <input id="preview_fps" type="number" step="1"></label>
              <label>Camera <input id="camera_device"></label>
              <label>Camera Width <input id="camera_width" type="number"></label>
              <label>Camera Height <input id="camera_height" type="number"></label>
            </div>
          </details>
          <details>
            <summary>Conversation</summary>
            <div class="details-body grid">
              <label>Gemma GPU Layers <input id="gemma_n_gpu_layers" type="number"></label>
              <label>Gemma Context <input id="gemma_n_ctx" type="number"></label>
              <label>Gemma Max Tokens <input id="gemma_max_tokens" type="number"></label>
              <label>Gemma Temp <input id="gemma_temperature" type="number" step="0.05"></label>
              <label>Gemma Top P <input id="gemma_top_p" type="number" step="0.05"></label>
              <label>STT RMS <input id="stt_min_audio_rms" type="number" step="0.001"></label>
              <label>STT Model <input id="stt_model"></label>
              <label>STT Language <input id="stt_language"></label>
              <label class="span-2">Gemma GGUF <input id="gemma_gguf_path" class="path-input"></label>
            </div>
          </details>
          <details>
            <summary>Audio</summary>
            <div class="details-body grid">
              <label>Mic Input <input id="audio_input_device"></label>
              <label>Audio Output <input id="audio_output_device"></label>
              <label class="span-2">Voice Ref <input id="voice_ref" class="path-input"></label>
            </div>
          </details>
          <details>
            <summary>Files</summary>
            <div class="details-body grid">
              <label class="span-2">Avatar CKPT <input id="avatar_ckpt" class="path-input"></label>
              <label class="span-2">Audio Driver CKPT <input id="audio_driver_ckpt" class="path-input"></label>
              <label class="span-2">Persona <input id="persona_file" class="path-input"></label>
              <label class="span-2">Preview Path <input id="preview_frame_path" class="path-input"></label>
              <label class="span-2">Control State <input id="control_state_path" class="path-input"></label>
              <label class="span-2">Log Path <input id="log_path" class="path-input"></label>
            </div>
          </details>
          <details>
            <summary>Flags</summary>
            <div class="details-body checkrow">
              <label><input id="skip_gemma" type="checkbox"> Skip Gemma</label>
              <label><input id="gemma_verbose" type="checkbox"> Gemma Verbose</label>
              <label><input id="skip_audio_listener" type="checkbox"> Skip Listener</label>
              <label><input id="disable_stt" type="checkbox"> Disable STT</label>
              <label><input id="enable_stdin_fallback" type="checkbox"> Stdin Input</label>
              <label><input id="play_audio" type="checkbox"> Play Audio</label>
              <label><input id="disable_runtime_animation" type="checkbox"> Disable Runtime Animation</label>
            </div>
          </details>
        </div>
      </section>
    </div>
  </main>
  <script>
    const DEFAULT_CONFIG = {config_json};
    const RUNTIME_PROFILES = {profiles_json};
    const boolFields = new Set({json.dumps(sorted(BOOL_FIELDS))});
    const intFields = new Set({json.dumps(sorted(INT_FIELDS))});
    const floatFields = new Set({json.dumps(sorted(FLOAT_FIELDS))});
    const fieldIds = Object.keys(DEFAULT_CONFIG);
    let formDirty = false;
    let currentControl = {{mic_muted: false}};

    function initProfiles() {{
      const select = document.getElementById('profile');
      if (!select) return;
      select.innerHTML = '';
      for (const [key, profile] of Object.entries(RUNTIME_PROFILES)) {{
        const option = document.createElement('option');
        option.value = key;
        option.textContent = profile.label || key;
        select.appendChild(option);
      }}
    }}

    function setMessage(text) {{
      document.getElementById('message').textContent = text || '';
    }}

    function loadConfig(cfg) {{
      for (const key of fieldIds) {{
        const el = document.getElementById(key);
        if (!el) continue;
        if (el.type === 'checkbox') el.checked = !!cfg[key];
        else el.value = cfg[key] ?? '';
      }}
    }}

    function applyProfile(profileKey) {{
      const profile = RUNTIME_PROFILES[profileKey];
      if (!profile) return;
      const current = collectConfig();
      const next = Object.assign({{}}, current, profile, {{profile: profileKey}});
      delete next.label;
      loadConfig(next);
      formDirty = true;
      setMessage('Profile loaded: ' + (profile.label || profileKey));
    }}

    function collectConfig() {{
      const cfg = {{}};
      for (const key of fieldIds) {{
        const el = document.getElementById(key);
        if (!el) continue;
        if (el.type === 'checkbox') cfg[key] = el.checked;
        else if (intFields.has(key)) cfg[key] = parseInt(el.value || '0', 10);
        else if (floatFields.has(key)) cfg[key] = parseFloat(el.value || '0');
        else cfg[key] = el.value;
      }}
      return cfg;
    }}

    async function postJSON(path, payload) {{
      formDirty = false;
      const res = await fetch(path, {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(payload)
      }});
      const data = await res.json();
      setMessage(data.message || JSON.stringify(data));
      await refreshStatus();
      return data;
    }}

    function startRun() {{ postJSON('/api/start', collectConfig()); }}
    function stopRun() {{ postJSON('/api/stop', collectConfig()); }}
    function restartRun() {{ postJSON('/api/restart', collectConfig()); }}
    function toggleMic() {{
      postJSON('/api/mic', {{muted: !currentControl.mic_muted, config: collectConfig()}});
    }}
    function sendPrompt() {{
      const text = document.getElementById('prompt').value;
      postJSON('/api/prompt', {{text, session_name: document.getElementById('session_name').value}});
    }}

    function updateMic(control) {{
      currentControl = control || {{mic_muted: false}};
      const muted = !!currentControl.mic_muted;
      const button = document.getElementById('micToggle');
      const chip = document.getElementById('micChip');
      button.textContent = muted ? 'Mic Muted' : 'Mic On';
      button.classList.toggle('good', !muted);
      button.classList.toggle('warn', muted);
      chip.classList.toggle('muted', muted);
      chip.classList.toggle('live', !muted);
      chip.querySelector('.dot').classList.toggle('running', !muted);
      document.getElementById('micChipText').textContent = muted ? 'Mic muted' : 'Mic live';
    }}

    function updateMetrics(data) {{
      const cfg = data.config || DEFAULT_CONFIG;
      document.getElementById('sessionMetric').textContent = data.running ? data.session_name : 'stopped';
      document.getElementById('deviceMetric').textContent = cfg.device + ' / gpu ' + cfg.gpu;
      document.getElementById('cameraMetric').textContent = cfg.camera_device;
      document.getElementById('previewMetric').textContent = cfg.camera_width + 'x' + cfg.camera_height + ' @ ' + cfg.fps;
      document.getElementById('previewMeta').textContent = cfg.camera_width + 'x' + cfg.camera_height + ' / ' + cfg.preview_fps + ' fps preview';
    }}

    async function refreshStatus() {{
      const res = await fetch('/api/status');
      const data = await res.json();
      const dot = document.getElementById('statusDot');
      const runChip = document.getElementById('runChip');
      dot.classList.toggle('running', !!data.running);
      runChip.classList.toggle('live', !!data.running);
      document.getElementById('statusText').textContent = data.running ? 'Running: ' + data.session_name : 'Stopped';
      const logs = document.getElementById('logs');
      logs.textContent = data.logs || '';
      logs.scrollTop = logs.scrollHeight;
      updateMic(data.control);
      updateMetrics(data);
      if (data.config && !formDirty) loadConfig(data.config);
    }}

    initProfiles();
    loadConfig(DEFAULT_CONFIG);
    for (const key of fieldIds) {{
      const el = document.getElementById(key);
      if (el) {{
        el.addEventListener('input', () => {{ formDirty = true; }});
        el.addEventListener('change', () => {{ formDirty = true; }});
      }}
    }}
    const profileEl = document.getElementById('profile');
    if (profileEl) {{
      profileEl.addEventListener('change', () => applyProfile(profileEl.value));
    }}
    updateMic({{mic_muted: false}});
    refreshStatus();
    setInterval(refreshStatus, 1500);
  </script>
</body>
</html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TalkingHeadAvatar runtime control app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), RuntimeAppHandler)
    print(f"[runtime_app] Serving on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[runtime_app] shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
