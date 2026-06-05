"""Code Review Agent — reviews code for quality, security, and style issues."""

import json
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from ..state import ReviewComment


REVIEW_SYSTEM_PROMPT = """You are an expert code review agent. Your job is to analyze code and find issues.

Review for:
1. Correctness: bugs, logic errors, edge cases
2. Security: injection, hardcoded secrets, unsafe patterns
3. Style: naming conventions, readability, DRY violations
4. Performance: unnecessary allocations, algorithmic concerns
5. Error handling: missing exception handling, silent failures

Return a JSON object with this structure:
{{
  "score": <float 0-10>,
  "comments": [
    {{"line": <int>, "severity": "error|warning|info", "category": "<str>", "message": "<str>", "suggestion": "<str>"}},
    ...
  ]
}}

Maximum {max_comments} comments. Be concise and actionable. Only flag real issues."""

REVIEW_USER_PROMPT = """Language: {language}
Description: {description}

Code to review:
```
{code}
```"""


def create_review_agent(
    llm: BaseChatModel,
    max_comments: int = 10,
) -> Callable:
    """
    Factory: returns a LangGraph node callable for code review.

    Args:
        llm: The language model to use for review.
        max_comments: Maximum number of review comments to return.

    Returns:
        A callable that accepts WorkflowState and returns updated state.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", REVIEW_SYSTEM_PROMPT.format(max_comments=max_comments)),
        ("human", REVIEW_USER_PROMPT),
    ])
    parser = JsonOutputParser()
    chain = prompt | llm | parser

    def review_node(state: dict) -> dict:
        """Execute the code review node."""
        code = state.get("code", "")
        language = state.get("language", "python")
        description = state.get("description", "")

        result = chain.invoke({
            "code": code[:15000],
            "language": language,
            "description": description,
        })

        score = float(result.get("score", 5.0))
        comments_raw = result.get("comments", [])
        comments: list[ReviewComment] = []
        for c in comments_raw:
            comments.append({
                "line": int(c.get("line", 0)),
                "severity": c.get("severity", "info"),
                "category": c.get("category", "general"),
                "message": c.get("message", ""),
                "suggestion": c.get("suggestion", ""),
            })

        errors = [c for c in comments if c["severity"] == "error"]

        return {
            "review_score": score,
            "review_comments": comments[:max_comments],
            "review_passed": len(errors) == 0 and score >= 4.0,
            "current_node": "code_review",
            "iteration": state.get("iteration", 0) + 1,
        }

    return review_node


def review_fallback(code: str, language: str, max_comments: int = 10) -> dict:
    """
    Deterministic fallback review when LLM is unavailable.

    Performs static analysis using simple heuristics.
    """
    comments: list[ReviewComment] = []
    lines = code.split("\n")

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if "password" in stripped.lower() and ("=" in stripped and "'" in stripped):
            comments.append({
                "line": i, "severity": "error",
                "category": "security",
                "message": "Possible hardcoded password detected",
                "suggestion": "Use environment variables or a secrets manager",
            })
            if len(comments) >= max_comments:
                break

        if "todo" in stripped.lower() or "fixme" in stripped.lower():
            comments.append({
                "line": i, "severity": "info",
                "category": "maintenance",
                "message": f"Found marker: {stripped}",
                "suggestion": "Address TODO/FIXME before merging",
            })
            if len(comments) >= max_comments:
                break

        if language == "python" and "eval(" in stripped:
            comments.append({
                "line": i, "severity": "error",
                "category": "security",
                "message": "Use of eval() is dangerous",
                "suggestion": "Use ast.literal_eval() or a safer alternative",
            })
            if len(comments) >= max_comments:
                break

        if language == "python" and "import *" in stripped:
            comments.append({
                "line": i, "severity": "warning",
                "category": "style",
                "message": "Wildcard import obscures namespace",
                "suggestion": "Import specific names explicitly",
            })
            if len(comments) >= max_comments:
                break

    error_count = len([c for c in comments if c["severity"] == "error"])
    warning_count = len([c for c in comments if c["severity"] == "warning"])
    score = max(0.0, 10.0 - error_count * 2.0 - warning_count * 0.5)

    return {
        "review_score": round(score, 1),
        "review_comments": comments[:max_comments],
        "review_passed": error_count == 0 and score >= 4.0,
        "current_node": "code_review_fallback",
        "iteration": 1,
    }
