"""Worker configuration.

All fields and defaults are the authoritative mirror of DESIGN.md §5.7
(Worker env-var table). Values are read from environment variables (and an
optional `.env` file) via pydantic-settings.

Field names use SCREAMING_SNAKE_CASE to match the environment variable names
directly — pydantic-settings will read each field from the env var of the
same name when `env_prefix=""`.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Inference worker runtime configuration.

    See DESIGN.md §5.7 for the canonical description of every field.
    """

    # Redis
    REDIS_URL: str = "redis://redis:6379/0"

    # YOLO detector
    DETECT_CONF: float = Field(0.35, ge=0.0, le=1.0)
    DETECT_IOU: float = Field(0.5, ge=0.0, le=1.0)
    DETECT_IMGSZ: int = Field(416, ge=32)

    # MobileCLIP classifier
    CLIP_INPUT_SIZE: int = Field(224, ge=1)
    # `zero_shot` uses the prompts.md text features; `probe` uses a
    # linear head trained offline via scripts/train_probe.py.
    CLASSIFIER_MODE: Literal["zero_shot", "probe"] = "zero_shot"
    PROBE_PATH: Path = Path("/models/classifier-probe/probe.npz")

    # Rolling-window voting
    VOTE_WINDOW: int = Field(15, ge=1)
    VOTE_STABLE_MIN: int = Field(8, ge=1)
    VOTE_STABLE_CONF: float = Field(0.55, ge=0.0, le=1.0)

    # Filesystem paths
    MODELS_DIR: Path = Path("/models")
    PROMPTS_FILE: Path = Path("/config/prompts.yaml")

    # OpenVINO runtime
    OPENVINO_DEVICE: str = "CPU"
    OPENVINO_THREADS: int = Field(0, ge=0)  # 0 = auto

    model_config = SettingsConfigDict(env_file=".env", env_prefix="")
