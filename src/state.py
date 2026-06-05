"""Typed state for the multi-agent workflow graph."""

from typing import Any, List, Literal, Optional
from typing_extensions import TypedDict


class ReviewComment(TypedDict):
    """A single code review finding."""

    line: int
    severity: Literal["error", "warning", "info"]
    category: str
    message: str
    suggestion: str


class TestPlan(TypedDict):
    """A planned test case."""

    name: str
    description: str
    input_summary: str
    expected_behavior: str


class DocSection(TypedDict):
    """A generated documentation section."""

    heading: str
    content: str
    target_file: str


class WorkflowState(TypedDict):
    """
    Shared state flowing through the LangGraph workflow.

    Each agent node reads and writes subsets of this state.
    """

    # Inputs
    code: str
    language: str
    description: str

    # Code review outputs
    review_comments: List[ReviewComment]
    review_score: float
    review_passed: bool

    # Testing outputs
    test_plan: List[TestPlan]
    test_coverage_estimate: float

    # Documentation outputs
    doc_sections: List[DocSection]
    doc_summary: str

    # Routing and metadata
    current_node: str
    iteration: int
    metadata: dict[str, Any]
