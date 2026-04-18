from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


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


@dataclass(slots=True)
class AppConfig:
    source: SourceConfig
    models: ModelConfig
    thresholds: ThresholdConfig
    timings: TimingConfig
    inventory: InventoryConfig
    output: OutputConfig

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


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        return default_config()

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    raw = _load_yaml(cfg_path) if cfg_path.suffix.lower() in {".yaml", ".yml"} else _load_json(cfg_path)
    config = default_config()
    config.source = _merge_dataclass(SourceConfig, raw.get("source", {}))
    config.models = _merge_dataclass(ModelConfig, raw.get("models", {}))
    config.thresholds = _merge_dataclass(ThresholdConfig, raw.get("thresholds", {}))
    config.timings = _merge_dataclass(TimingConfig, raw.get("timings", {}))
    config.inventory = _merge_dataclass(InventoryConfig, raw.get("inventory", {}))
    config.output = _merge_dataclass(OutputConfig, raw.get("output", {}))
    return config
