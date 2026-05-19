"""Worker configuration.

All fields and defaults are the authoritative mirror of DESIGN.md §5.7
(Worker env-var table). Values are read from environment variables (and an
optional `.env` file) via pydantic-settings.

Field names use SCREAMING_SNAKE_CASE to match the environment variable names
directly — pydantic-settings will read each field from the env var of the
same name when `env_prefix=""`.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Inference worker runtime configuration.

    See DESIGN.md §5.7 for the canonical description of every field.
    """

    # Redis
    REDIS_URL: str = "redis://redis:6379/0"

    # YOLO detector
    DETECT_CONF: float = 0.35
    DETECT_IOU: float = 0.5
    DETECT_IMGSZ: int = 416

    # MobileCLIP classifier
    CLIP_INPUT_SIZE: int = 224

    # Rolling-window voting
    VOTE_WINDOW: int = 15
    VOTE_STABLE_MIN: int = 8
    VOTE_STABLE_CONF: float = 0.55

    # Filesystem paths
    MODELS_DIR: Path = Path("/models")
    PROMPTS_FILE: Path = Path("/config/prompts.yaml")

    # OpenVINO runtime
    OPENVINO_DEVICE: str = "CPU"
    OPENVINO_THREADS: int = 0  # 0 = auto

    model_config = SettingsConfigDict(env_file=".env", env_prefix="")
