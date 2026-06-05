"""FastAPI application for serving the multi-agent workflow."""

import json
import time
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import AppConfig, load_config
from ..workflow.engine import run_workflow

app = FastAPI(
    title="LangGraph Multi-Agent Workflow",
    description="Code review, testing, and documentation agents orchestrated by LangGraph",
    version="1.0.0",
)


class CodeAnalyzeRequest(BaseModel):
    """Request to analyze code through the full agent workflow."""

    code: str = Field(..., min_length=1, max_length=50000)
    language: str = Field(default="python", min_length=1)
    description: str = Field(default="", max_length=2000)
    use_llm: bool = Field(default=False, description="Use LLM agents vs static fallback")


class WorkflowResponse(BaseModel):
    """Complete workflow result."""

    review_score: float
    review_passed: bool
    review_comments: list[dict[str, Any]]
    test_plan: list[dict[str, Any]]
    test_coverage_estimate: float
    doc_sections: list[dict[str, Any]]
    doc_summary: str
    nodes_visited: list[str]
    processing_time_ms: float


@app.get("/health")
def health_check() -> dict:
    """Health endpoint for container orchestrators."""
    return {
        "status": "healthy",
        "service": "langgraph-agents",
        "version": "1.0.0",
    }


@app.get("/config")
def get_config() -> dict:
    """Return current configuration (safe fields only)."""
    cfg = load_config()
    return {
        "model": cfg.llm.model,
        "api_base": cfg.llm.api_base,
        "temperature": cfg.llm.temperature,
        "max_review_comments": cfg.agents.max_review_comments,
        "max_test_cases": cfg.agents.max_test_cases,
    }


@app.post("/analyze", response_model=WorkflowResponse)
def analyze_code(request: CodeAnalyzeRequest) -> WorkflowResponse:
    """
    Run the full multi-agent analysis pipeline.

    Code flows: Review -> Testing -> Documentation.
    Use use_llm=false for fast static analysis, true for LLM-powered agents.
    """
    config = load_config()

    if len(request.code) > 50000:
        raise HTTPException(
            status_code=400,
            detail="Code exceeds 50KB limit. Submit smaller code units.",
        )

    start = time.monotonic()
    try:
        result = run_workflow(
            config=config,
            code=request.code,
            language=request.language,
            description=request.description,
            use_llm=request.use_llm,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Workflow error: {exc}")

    elapsed_ms = (time.monotonic() - start) * 1000

    return WorkflowResponse(
        review_score=result.get("review_score", 0.0),
        review_passed=result.get("review_passed", False),
        review_comments=result.get("review_comments", []),
        test_plan=result.get("test_plan", []),
        test_coverage_estimate=result.get("test_coverage_estimate", 0.0),
        doc_sections=result.get("doc_sections", []),
        doc_summary=result.get("doc_summary", ""),
        nodes_visited=[result.get("current_node", "unknown")],
        processing_time_ms=round(elapsed_ms, 1),
    )
