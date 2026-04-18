from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True)
class SourceConfig:
    kind: str = "webcam"
    value: str = "0"
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass(slots=True)
class ModelConfig:
    mac_sam_model_id: str = "mlx-community/sam3.1-bf16"
    linux_sam_model_id: str = "facebook/sam3.1"
    inference_max_side: int = 960
    inference_interval_ms: int = 250
    provider: str = "sam3.1"
    detr_model_id: str = "facebook/detr-resnet-50"
    rt_detr_model_id: str = "jadechoghari/RT-DETRv2"


@dataclass(slots=True)
class ThresholdConfig:
    person_confidence: float = 0.3
    inventory_confidence: float = 0.25
    overlap_iou: float = 0.5
    tracking_iou: float = 0.3
    min_area_ratio: float = 0.002


@dataclass(slots=True)
class TimingConfig:
    empty_scene_rescan_seconds: float = 5.0
    person_confirmation_frames: int = 3
    pre_roll_seconds: float = 3.0
    post_roll_seconds: float = 2.0


@dataclass(slots=True)
class InventoryConfig:
    labels: list[str] = field(default_factory=list)
    auto_rescan: bool = True


@dataclass(slots=True)
class OutputConfig:
    directory: str = "recordings"
    recording_mode: str = "raw"


@dataclass(slots=True)
class UploadConfig:
    enabled: bool = False
    provider: str = "gcp"
    bucket: str = ""
    prefix: str = "meta-watcher/"
    credentials_path: str = ""
    region: str = ""
    namespace: str = ""
    profile: str = ""
    upload_videos: bool = True
    upload_snapshots: bool = True
    upload_metadata: bool = True
    delete_after_upload: bool = False
    queue_size: int = 32


@dataclass(slots=True)
class TimestampConfig:
    enabled: bool = False
    ots_binary: str = "ots"
    calendar_urls: list[str] = field(default_factory=list)
    timeout_seconds: float = 30.0
    stamp_videos: bool = True
    stamp_snapshots: bool = True
    stamp_frames: bool = False
    stamp_metadata: bool = False


@dataclass(slots=True)
class AppConfig:
    source: SourceConfig
    models: ModelConfig
    thresholds: ThresholdConfig
    timings: TimingConfig
    inventory: InventoryConfig
    output: OutputConfig
    upload: UploadConfig
    timestamps: TimestampConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_config() -> AppConfig:
    return AppConfig(
        source=SourceConfig(),
        models=ModelConfig(),
        thresholds=ThresholdConfig(),
        timings=TimingConfig(),
        inventory=InventoryConfig(),
        output=OutputConfig(),
        upload=UploadConfig(),
        timestamps=TimestampConfig(),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load YAML config files.") from exc
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be an object in {path}")
    return data


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be an object in {path}")
    return data


def _merge_dataclass(cls: type[Any], data: dict[str, Any]) -> Any:
    base = cls()
    for key, value in data.items():
        if hasattr(base, key):
            setattr(base, key, value)
    return base


def build_config_from_dict(raw: dict[str, Any]) -> AppConfig:
    """Reconstruct an AppConfig from a plain dict, using dataclass defaults for
    anything missing. Unknown top-level or nested keys are silently dropped —
    the public PUT /api/config contract depends on that tolerance.
    """
    config = default_config()
    config.source = _merge_dataclass(SourceConfig, raw.get("source", {}) or {})
    config.models = _merge_dataclass(ModelConfig, raw.get("models", {}) or {})
    config.thresholds = _merge_dataclass(ThresholdConfig, raw.get("thresholds", {}) or {})
    config.timings = _merge_dataclass(TimingConfig, raw.get("timings", {}) or {})
    config.inventory = _merge_dataclass(InventoryConfig, raw.get("inventory", {}) or {})
    config.output = _merge_dataclass(OutputConfig, raw.get("output", {}) or {})
    config.upload = _merge_dataclass(UploadConfig, raw.get("upload", {}) or {})
    config.timestamps = _merge_dataclass(TimestampConfig, raw.get("timestamps", {}) or {})
    return config


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        return default_config()

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    raw = _load_yaml(cfg_path) if cfg_path.suffix.lower() in {".yaml", ".yml"} else _load_json(cfg_path)
    return build_config_from_dict(raw)


def repo_root() -> Path:
    """Directory holding the top-level `meta_watcher/` package.

    `Path(__file__).resolve().parents[1]` lands on the repo root during
    development (editable install) and on the site-packages parent when
    installed. Callers treat it as the default search dir and must tolerate
    a non-writable location.
    """
    return Path(__file__).resolve().parents[1]


def list_config_files(search_dirs: Iterable[Path]) -> list[Path]:
    """Enumerate JSON/YAML config candidates under `search_dirs`.

    Hidden files (starting with '.') are excluded; duplicates (same `resolve()`)
    collapsed; results sorted by name. Directories that don't exist are skipped.
    """
    seen: dict[Path, Path] = {}
    for directory in search_dirs:
        d = Path(directory)
        if not d.is_dir():
            continue
        for pattern in ("*.json", "*.yaml", "*.yml"):
            for entry in d.glob(pattern):
                if entry.name.startswith("."):
                    continue
                if not entry.is_file():
                    continue
                resolved = entry.resolve()
                seen.setdefault(resolved, entry)
    return sorted(seen.values(), key=lambda p: p.name.lower())


def save_config(path: str | Path, config: AppConfig) -> Path:
    """Atomically persist `config` to `path`.

    YAML targets are redirected to a JSON sibling (`name.json`) so we don't
    silently destroy comments/ordering. JSON is written with `indent=2` and a
    trailing newline. Returns the actual path written (which may differ from
    the input when a YAML path was redirected).
    """
    target = Path(path)
    if target.suffix.lower() in {".yaml", ".yml"}:
        target = target.with_suffix(".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(asdict(config), handle, indent=2)
            handle.write("\n")
        os.replace(tmp, target)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return target
