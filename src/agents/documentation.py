"""Documentation Agent — generates documentation from code and analysis."""

import json
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from ..state import DocSection


DOC_SYSTEM_PROMPT = """You are an expert documentation agent. Generate clear, useful documentation from code.

Produce these sections:
1. Overview: what the code does, key capabilities
2. API Reference: functions/classes with signatures and descriptions
3. Usage Examples: 2-3 practical code snippets
4. Configuration: any tunable parameters or environment variables
5. Architecture: design patterns, key decisions

Return a JSON object:
{{
  "doc_sections": [
    {{"heading": "<h2 heading>", "content": "<markdown content>", "target_file": "README.md|api.md|..."}}
  ],
  "doc_summary": "<one-sentence summary of the documented module>"
}}

Maximum {max_tokens} tokens of output. Write for a technical audience familiar with {language}."""

DOC_USER_PROMPT = """Language: {language}
Module description: {description}

Code review score: {review_score}/10
Review issues to document: {review_notes}

Test coverage plan: {test_count} test cases planned
Coverage estimate: {coverage_estimate:.0%}

Code:
```
{code}
```"""


def create_documentation_agent(
    llm: BaseChatModel,
    max_tokens: int = 2000,
) -> Callable:
    """
    Factory: returns a LangGraph node callable for documentation generation.

    Args:
        llm: The language model to use.
        max_tokens: Approximate max tokens for documentation output.

    Returns:
        A callable node function.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", DOC_SYSTEM_PROMPT.format(max_tokens=max_tokens)),
        ("human", DOC_USER_PROMPT),
    ])
    parser = JsonOutputParser()
    chain = prompt | llm | parser

    def documentation_node(state: dict) -> dict:
        """Execute the documentation node."""
        code = state.get("code", "")
        language = state.get("language", "python")
        description = state.get("description", "")
        review_score = state.get("review_score", 5.0)
        test_plan = state.get("test_plan", [])
        coverage = state.get("test_coverage_estimate", 0.6)

        review_comments = state.get("review_comments", [])
        review_notes = "; ".join(
            f"[{c['severity']}] {c['message']}"
            for c in review_comments[:3]
        ) or "No significant issues."

        result = chain.invoke({
            "code": code[:12000],
            "language": language,
            "description": description,
            "review_score": review_score,
            "review_notes": review_notes,
            "test_count": len(test_plan),
            "coverage_estimate": coverage,
        })

        sections_raw = result.get("doc_sections", [])
        sections: list[DocSection] = []
        for s in sections_raw:
            sections.append({
                "heading": s.get("heading", ""),
                "content": s.get("content", ""),
                "target_file": s.get("target_file", "README.md"),
            })

        return {
            "doc_sections": sections,
            "doc_summary": result.get("doc_summary", ""),
            "current_node": "documentation",
            "iteration": state.get("iteration", 0) + 1,
        }

    return documentation_node


def documentation_fallback(code: str, language: str = "python", description: str = "") -> dict:
    """
    Deterministic fallback: generate documentation from static analysis.

    Extracts function signatures, class names, and docstrings to build docs.
    """
    import re

    sections: list[DocSection] = []

    functions = []
    classes = []
    if language == "python":
        func_re = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*(\w+))?:")
        class_re = re.compile(r"^\s*class\s+(\w+)(?:\(([^)]*)\))?:")
        docstring_re = re.compile(r'("""(.*?)"""|\'\'\'(.*?)\'\'\')', re.DOTALL)

        lines = code.split("\n")
        for line in lines:
            m = func_re.match(line)
            if m and not m.group(1).startswith("_"):
                fname = m.group(1)
                params = m.group(2).strip()
                functions.append(f"- `{fname}({params})`")

            m2 = class_re.match(line)
            if m2:
                cname = m2.group(1)
                bases = m2.group(2) or "object"
                classes.append(f"- `{cname}({bases})`")

    if functions:
        sections.append({
            "heading": "## API Reference — Functions",
            "content": "\n".join(functions),
            "target_file": "README.md",
        })

    if classes:
        sections.append({
            "heading": "## API Reference — Classes",
            "content": "\n".join(classes),
            "target_file": "README.md",
        })

    sections.insert(0, {
        "heading": "## Overview",
        "content": description or f"A {language} module containing {len(functions)} functions and {len(classes)} classes.",
        "target_file": "README.md",
    })

    sections.append({
        "heading": "## Quick Start",
        "content": f"# Import the module\nimport sys\nsys.path.insert(0, 'src')\n# See API Reference for available functions",
        "target_file": "README.md",
    })

    summary = description or f"Auto-generated documentation for a {language} module."

    return {
        "doc_sections": sections,
        "doc_summary": summary,
        "current_node": "documentation_fallback",
        "iteration": 1,
    }
