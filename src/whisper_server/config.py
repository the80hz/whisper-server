from pydantic_settings import BaseSettings, SettingsConfigDict


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

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
