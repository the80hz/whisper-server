import logging
import math
from typing import Any

from pydantic import ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


logger = logging.getLogger("whisper-api.config")


class Settings(BaseSettings):
    port: int = 3373
    whisper_model: str = "large-v3-turbo"
    log_level: str = "INFO"
    device: str = "auto"
    compute_type: str = "int8"
    log_file: str = "logs/whisper.log"
    queue_max_size: int = 8
    default_timeout_seconds: float = 180.0
    vad_filter: bool = True
    vad_threshold: float = 0.5
    vad_min_silence_duration_ms: int = 500
    vad_speech_pad_ms: int = 200
    condition_on_previous_text: bool = False
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 3
    compression_ratio_threshold: float = 2.2
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    temperature_fallback: bool = True
    hallucination_silence_threshold: float = 1.0
    cpu_threads: int = 0
    # If >0, automatically unload the Whisper model after this many seconds of idle time.
    # Set to 0 to disable automatic unload.
    model_unload_seconds: float = 600.0
    max_upload_mb: float = 50.0
    api_token: str | None = None
    # Backward-compatible alias used by bratishkabot-whisper-server.
    # If API_TOKEN is set, it takes precedence over API_KEY.
    api_key: str | None = None

    @classmethod
    def _default(cls, info: ValidationInfo, value: Any) -> Any:
        default = cls.model_fields[info.field_name].default
        logger.warning(
            "Invalid %s=%r; using default %r",
            info.field_name.upper(),
            value,
            default,
        )
        return default

    @field_validator("device", mode="before")
    @classmethod
    def _valid_device(cls, value: Any, info: ValidationInfo) -> str:
        normalized = str(value).lower().strip()
        return normalized if normalized in {"auto", "cpu", "cuda"} else cls._default(info, value)

    @field_validator("compute_type", mode="before")
    @classmethod
    def _valid_compute_type(cls, value: Any, info: ValidationInfo) -> str:
        normalized = str(value).lower().strip()
        valid = {
            "auto",
            "default",
            "int8",
            "int8_float16",
            "int8_float32",
            "int8_bfloat16",
            "int16",
            "float16",
            "float32",
            "bfloat16",
        }
        return normalized if normalized in valid else cls._default(info, value)

    @field_validator("vad_filter", "condition_on_previous_text", "temperature_fallback", mode="before")
    @classmethod
    def _valid_bool(cls, value: Any, info: ValidationInfo) -> bool:
        if isinstance(value, bool):
            return value
        normalized = str(value).lower().strip()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return cls._default(info, value)

    @field_validator(
        "port",
        "queue_max_size",
        mode="before",
    )
    @classmethod
    def _valid_positive_int(cls, value: Any, info: ValidationInfo) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return cls._default(info, value)
        return parsed if parsed > 0 else cls._default(info, value)

    @field_validator(
        "vad_min_silence_duration_ms",
        "vad_speech_pad_ms",
        "no_repeat_ngram_size",
        "cpu_threads",
        mode="before",
    )
    @classmethod
    def _valid_nonnegative_int(cls, value: Any, info: ValidationInfo) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return cls._default(info, value)
        return parsed if parsed >= 0 else cls._default(info, value)

    @field_validator("vad_threshold", "no_speech_threshold", mode="before")
    @classmethod
    def _valid_probability(cls, value: Any, info: ValidationInfo) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return cls._default(info, value)
        return parsed if math.isfinite(parsed) and 0 <= parsed <= 1 else cls._default(info, value)

    @field_validator(
        "default_timeout_seconds",
        "repetition_penalty",
        "compression_ratio_threshold",
        "max_upload_mb",
        mode="before",
    )
    @classmethod
    def _valid_positive_float(cls, value: Any, info: ValidationInfo) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return cls._default(info, value)
        valid = parsed >= 1 if info.field_name == "repetition_penalty" else parsed > 0
        return parsed if math.isfinite(parsed) and valid else cls._default(info, value)

    @field_validator(
        "log_prob_threshold",
        "model_unload_seconds",
        "hallucination_silence_threshold",
        mode="before",
    )
    @classmethod
    def _valid_finite_float(cls, value: Any, info: ValidationInfo) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return cls._default(info, value)
        if not math.isfinite(parsed):
            return cls._default(info, value)
        if info.field_name in {"model_unload_seconds", "hallucination_silence_threshold"} and parsed < 0:
            return cls._default(info, value)
        return parsed

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
