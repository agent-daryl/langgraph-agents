"""LangGraph workflow builder and routing logic."""

from typing import Callable, Dict, Literal

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from ..state import WorkflowState
from ..agents.code_review import create_review_agent, review_fallback
from ..agents.testing import create_testing_agent, testing_fallback
from ..agents.documentation import create_documentation_agent, documentation_fallback
from ..config import AppConfig


def route_after_review(state: dict) -> Literal["testing", "documentation"]:
    """
    Conditional edge: after code review, decide next node.

    If review passed, proceed to testing. If review failed with critical errors,
    skip to documentation with a note so the developer sees the issues.
    """
    if state.get("review_passed", False):
        return "testing"
    return "documentation"


def route_after_testing(state: dict) -> Literal["documentation", END]:
    """
    After testing, always proceed to documentation (or end if coverage too low).
    """
    coverage = state.get("test_coverage_estimate", 0.0)
    if coverage < 0.1:
        return END
    return "documentation"


def create_workflow(config: AppConfig) -> StateGraph:
    """
    Build the multi-agent LangGraph workflow.

    Graph topology:
        START -> code_review -> {testing -> documentation -> END}
                          \\-> documentation -> END  (if review fails)

    Args:
        config: Application configuration.

    Returns:
        Compiled LangGraph StateGraph.
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        openai_api_base=config.llm.api_base,
        openai_api_key=config.llm.api_key,
        model_name=config.llm.model,
        temperature=config.llm.temperature,
    )

    review_node = create_review_agent(llm, config.agents.max_review_comments)
    testing_node = create_testing_agent(llm, config.agents.max_test_cases)
    doc_node = create_documentation_agent(llm, config.agents.doc_max_tokens)

    graph = StateGraph(WorkflowState)

    graph.add_node("code_review", review_node)
    graph.add_node("testing", testing_node)
    graph.add_node("documentation", doc_node)

    graph.set_entry_point("code_review")
    graph.add_conditional_edges("code_review", route_after_review)
    graph.add_conditional_edges("testing", route_after_testing)
    graph.add_edge("documentation", END)

    return graph.compile()


def create_fallback_workflow(config: AppConfig):
    """
    Build a deterministic workflow that uses static-analysis fallbacks.

    No LLM calls required — useful for offline testing and CI.
    """
    max_comments = config.agents.max_review_comments
    max_tests = config.agents.max_test_cases

    def fallback_review(state: dict) -> dict:
        result = review_fallback(
            state.get("code", ""),
            state.get("language", "python"),
            max_comments,
        )
        return {**result, "iteration": state.get("iteration", 0) + 1, "current_node": "code_review"}

    def fallback_testing(state: dict) -> dict:
        result = testing_fallback(
            state.get("code", ""),
            state.get("language", "python"),
            max_tests,
        )
        return {**result, "iteration": state.get("iteration", 0) + 1, "current_node": "testing"}

    def fallback_documentation(state: dict) -> dict:
        result = documentation_fallback(
            state.get("code", ""),
            state.get("language", "python"),
            state.get("description", ""),
        )
        return {**result, "iteration": state.get("iteration", 0) + 1, "current_node": "documentation"}

    graph = StateGraph(WorkflowState)
    graph.add_node("code_review", fallback_review)
    graph.add_node("testing", fallback_testing)
    graph.add_node("documentation", fallback_documentation)

    graph.set_entry_point("code_review")
    graph.add_conditional_edges("code_review", route_after_review)
    graph.add_conditional_edges("testing", route_after_testing)
    graph.add_edge("documentation", END)

    return graph.compile()


def run_workflow(config: AppConfig, code: str, language: str = "python",
                 description: str = "", use_llm: bool = True) -> dict:
    """
    Convenience function: run the full workflow with given code.

    Args:
        config: Application configuration.
        code: Source code to analyze.
        language: Programming language.
        description: Human description of the code.
        use_llm: If False, use deterministic fallbacks.

    Returns:
        Final workflow state as a dictionary.
    """
    initial_state: WorkflowState = {
        "code": code,
        "language": language,
        "description": description,
        "review_comments": [],
        "review_score": 0.0,
        "review_passed": False,
        "test_plan": [],
        "test_coverage_estimate": 0.0,
        "doc_sections": [],
        "doc_summary": "",
        "current_node": "START",
        "iteration": 0,
        "metadata": {},
    }

    graph = create_workflow(config) if use_llm else create_fallback_workflow(config)
    return graph.invoke(initial_state)
