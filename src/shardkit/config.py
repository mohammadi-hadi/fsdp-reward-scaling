"""Configuration: frozen dataclasses, a YAML loader, and --key=value overrides.

No hydra. The whole surface is a few dozen fields, and every run writes its fully resolved
config next to its own numbers, so the settings that produced a result are committed with it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

#: How the FSDP1 names an interviewer will use map onto what this code does.
STRATEGIES = ("full_shard", "grad_op", "no_shard", "hsdp")


@dataclass(frozen=True)
class ModelConfig:
    name: str = "Qwen/Qwen3-1.7B-Base"
    tiny: bool = False
    tiny_layers: int = 4
    tiny_hidden: int = 64
    tiny_vocab: int = 256
    max_length: int = 2048


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "HuggingFaceH4/ultrafeedback_binarized"
    train_split: str = "train_prefs"
    eval_split: str = "test_prefs"
    synthetic_pairs: int = 0  # >0 uses generated pairs and never touches the network
    micro_batch_pairs: int = 8


@dataclass(frozen=True)
class ParallelConfig:
    strategy: str = "full_shard"
    replicate: int = 1  # only read when strategy is hsdp
    param_dtype: str = "bfloat16"
    reduce_dtype: str = "float32"
    activation_checkpointing: bool = True


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 50
    warmup_steps: int = 10
    lr: float = 1e-5
    weight_decay: float = 0.0
    grad_accum: int = 1
    seed: int = 7
    checkpoint_every: int = 0
    inject_fail_at_step: int = -1
    inject_fail_rank: int = 0


@dataclass(frozen=True)
class Config:
    run_name: str = "unnamed"
    device_type: str = "auto"  # auto | cuda | cpu
    out_dir: str = "runs"
    peak_flops_per_gpu: float = 362e12  # L40S bf16 dense; override per machine
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_SECTIONS = {
    "model": ModelConfig,
    "data": DataConfig,
    "parallel": ParallelConfig,
    "train": TrainConfig,
}


def _coerce(current: Any, raw: str) -> Any:
    if isinstance(current, bool):
        return raw.lower() in {"1", "true", "yes"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def load(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Read a YAML config, then apply ``section.field=value`` overrides from the command line."""
    raw: dict[str, Any] = {}
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text())
        raw = loaded or {}

    sections: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        sections[name] = cls(**(raw.get(name) or {}))
    top = {k: v for k, v in raw.items() if k not in _SECTIONS}
    config = Config(**top, **sections)

    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"override must be key=value, got {override!r}")
        key, value = override.split("=", 1)
        if "." in key:
            section, field_name = key.split(".", 1)
            if section not in _SECTIONS:
                raise ValueError(f"unknown config section {section!r}")
            current = getattr(getattr(config, section), field_name)
            new_section = dataclasses.replace(
                getattr(config, section), **{field_name: _coerce(current, value)}
            )
            config = dataclasses.replace(config, **{section: new_section})
        else:
            config = dataclasses.replace(config, **{key: _coerce(getattr(config, key), value)})

    if config.parallel.strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {config.parallel.strategy!r}")
    return config
