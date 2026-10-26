"""Small, strict local config. Paths are resolved without creating directories."""

from dataclasses import dataclass, field
import os
from pathlib import Path
import tomllib
from typing import Literal


class ConfigError(ValueError):
    """An unsupported or invalid configuration."""


def _default_cache() -> Path:
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache"))
    return cache_home / "talkinghead-v2" / "artifacts"


@dataclass(frozen=True)
class LocalConfig:
    profile: Literal["local"] = "local"
    artifact_cache_dir: Path = field(default_factory=_default_cache)
    output_dir: Path = field(default_factory=lambda: Path.cwd() / ".tha-v2" / "output")
    min_free_bytes: int = 1_073_741_824

    def __post_init__(self) -> None:
        if self.profile != "local":
            raise ConfigError("profile must be 'local'; other profiles are not implemented")
        if type(self.min_free_bytes) is not int or self.min_free_bytes < 0:
            raise ConfigError("min_free_bytes must be a nonnegative integer")
        for name in ("artifact_cache_dir", "output_dir"):
            path = getattr(self, name)
            if not isinstance(path, Path):
                raise ConfigError(f"{name} must be a pathlib.Path")
            try:
                object.__setattr__(self, name, path.expanduser().resolve())
            except (OSError, ValueError, RuntimeError) as exc:
                raise ConfigError(f"cannot resolve {name}: {exc}") from exc


def load_config(path: Path | None = None, *, profile: str = "local") -> LocalConfig:
    """Read TOML; relative paths belong to the config file, not the caller's cwd."""
    if profile != "local":
        raise ConfigError("only the local profile is implemented")
    if path is None:
        return LocalConfig()
    try:
        config_path = path.expanduser().resolve()
        with config_path.open("rb") as stream:
            values = tomllib.load(stream)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    unknown = values.keys() - {"profile", "artifact_cache_dir", "output_dir", "min_free_bytes"}
    if unknown:
        raise ConfigError(f"unknown config keys: {', '.join(sorted(unknown))}")
    for name in ("artifact_cache_dir", "output_dir"):
        if name in values:
            raw = values[name]
            if not isinstance(raw, str) or not raw.strip():
                raise ConfigError(f"{name} must be a nonempty path string")
            try:
                expanded = Path(raw).expanduser()
            except (OSError, RuntimeError) as exc:
                raise ConfigError(f"cannot resolve {name}: {exc}") from exc
            values[name] = expanded if expanded.is_absolute() else config_path.parent / expanded
    return LocalConfig(**values)
