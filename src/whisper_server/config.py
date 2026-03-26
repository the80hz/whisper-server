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
    max_upload_mb: float = 50.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
