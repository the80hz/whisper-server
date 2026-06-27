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
    # If >0, automatically unload the Whisper model after this many seconds of idle time.
    # Set to 0 to disable automatic unload.
    model_unload_seconds: float = 600.0
    max_upload_mb: float = 50.0
    api_token: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
