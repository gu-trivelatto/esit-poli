from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr


class Settings(BaseSettings):
    TAVILY_API_KEY: SecretStr
    GROQ_API_KEY: SecretStr
    LLAMA_CLOUD_API_KEY: SecretStr
    QDRANT_API_KEY: SecretStr
    QDRANT_URL: str
    CHAT_MODEL: str
    HT_MODEL: str
    HUGGINGFACE_EMBEDDING_MODEL: str
    DB_POSTGRESQL_SERVER: str
    DB_POSTGRESQL_DATA_DATABASE: str
    DB_POSTGRESQL_MEMORY_DATABASE: str
    DB_POSTGRESQL_USER: str
    DB_POSTGRESQL_PWD: str
    DB_POSTGRESQL_PORT: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding='utf-8',
        env_nested_delimiter="__",
        case_sensitive=False,
    )

settings = Settings.model_validate({})
