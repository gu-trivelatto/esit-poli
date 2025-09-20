import logging
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel, SecretStr

logger = logging.getLogger(__name__)

class LLMSettings(BaseModel):
    PROVIDER: Literal['openai', 'gemini', 'groq', 'ollama', 'anthropic'] = "openai"
    API_KEY: SecretStr
    MODEL: str = "gpt-4.1"
    TEMPERATURE: float = 0.7

class MemorySettings(BaseModel):
    STRATEGY: Literal['redis', 'postgres', 'sqlite', 'in_memory'] = "in_memory"
    URL: SecretStr

class LoggerSettings(BaseModel):
    LEVEL: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO'

class Settings(BaseSettings):
    TAVILY_API_KEY: str
    GROQ_API_KEY: str
    LLAMA_CLOUD_API_KEY: str
    QDRANT_API_KEY: str
    QDRANT_URL: str
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
logger.info("Loading configuration from env...")
logger.debug(f"Settings model: {settings.model_dump()}")