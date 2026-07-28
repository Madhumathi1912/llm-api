from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key_: str
    openai_model: str = "gpt-4o-mini"

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()