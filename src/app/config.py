from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Production RAG Document Assistant"
    app_env: str = "development"
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag_assistant"
    max_upload_size_mb: int = Field(default=25, gt=0)
    chunk_min_tokens: int = Field(default=500, gt=0)
    chunk_max_tokens: int = Field(default=800, gt=0)
    chunk_overlap_tokens: int = Field(default=80, ge=0)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = Field(default=384, gt=0)
    retrieval_dense_weight: float = Field(default=0.55, ge=0, le=1)
    retrieval_candidate_limit: int = Field(default=50, ge=5, le=1000)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("database_url", mode="before")
    @classmethod
    def use_psycopg_driver(cls, value: str) -> str:
        """Normalize common provider URLs to SQLAlchemy's psycopg 3 driver."""
        if value.startswith("postgres://"):
            return value.replace("postgres://", "postgresql+psycopg://", 1)
        if value.startswith("postgresql://"):
            return value.replace("postgresql://", "postgresql+psycopg://", 1)
        return value

    @model_validator(mode="after")
    def validate_chunk_sizes(self) -> "Settings":
        if self.chunk_min_tokens > self.chunk_max_tokens:
            raise ValueError("chunk_min_tokens must not exceed chunk_max_tokens")
        if self.chunk_overlap_tokens >= self.chunk_min_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_min_tokens")
        if self.embedding_dimension != 384:
            raise ValueError("embedding_dimension must match the database vector size of 384")
        return self

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
