"""Configuration management."""

import os
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM backend configuration."""

    api_base: str = Field(
        default="http://10.10.0.20:8000/v1",
        description="OpenAI-compatible API endpoint",
    )
    api_key: str = Field(default="not-needed", description="API key (may be placeholder for local)")
    model: str = Field(default="qwen3.6:27b_256k", description="Model name/ID")
    temperature: float = Field(default=0.3, ge=0, le=2, description="Sampling temperature")


class AgentConfig(BaseModel):
    """Per-agent tuning overrides."""

    max_review_comments: int = Field(default=10, ge=1, le=50)
    max_test_cases: int = Field(default=15, ge=1, le=50)
    doc_max_tokens: int = Field(default=2000, ge=100, le=10000)
    review_min_pass_score: float = Field(default=6.0, ge=0, le=10)


class AppConfig(BaseModel):
    """Top-level application configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if os.getenv("LLM_API_BASE"):
            self.llm.api_base = os.getenv("LLM_API_BASE")
        if os.getenv("LLM_MODEL"):
            self.llm.model = os.getenv("LLM_MODEL")
        if os.getenv("LLM_API_KEY"):
            self.llm.api_key = os.getenv("LLM_API_KEY")
        if os.getenv("LLM_TEMPERATURE"):
            self.llm.temperature = float(os.getenv("LLM_TEMPERATURE"))


def load_config() -> AppConfig:
    """Load configuration from environment variables."""
    return AppConfig()
