"""Testing Agent — generates a test plan and coverage estimate for code."""

import json
import re
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from ..state import TestPlan


TEST_SYSTEM_PROMPT = """You are an expert testing agent. Analyze code and produce a test plan.

For each function, class, or logical unit, generate test cases covering:
1. Happy path: normal inputs, expected behavior
2. Edge cases: empty inputs, boundaries, None/NaN
3. Error handling: invalid inputs, exceptions, timeouts
4. Integration: interactions between components

Return a JSON object:
{{
  "test_plan": [
    {{"name": "<test_name>", "description": "<what it tests>", "input_summary": "<inputs>", "expected_behavior": "<outcome>"}}
  ],
  "coverage_estimate": <float 0.0-1.0 estimated coverage if all tests pass>
}}

Maximum {max_tests} test cases. Be specific about inputs and expected behavior."""

TEST_USER_PROMPT = """Language: {language}
Description: {description}
Review score: {review_score}/10
Review passed: {review_passed}

Code:
```
{code}
```

Review feedback to address in tests:
{review_feedback}"""


def create_testing_agent(
    llm: BaseChatModel,
    max_tests: int = 15,
) -> Callable:
    """
    Factory: returns a LangGraph node callable for test planning.

    Args:
        llm: The language model to use.
        max_tests: Maximum test cases to generate.

    Returns:
        A callable node function.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", TEST_SYSTEM_PROMPT.format(max_tests=max_tests)),
        ("human", TEST_USER_PROMPT),
    ])
    parser = JsonOutputParser()
    chain = prompt | llm | parser

    def testing_node(state: dict) -> dict:
        """Execute the testing node."""
        code = state.get("code", "")
        language = state.get("language", "python")
        description = state.get("description", "")
        review_score = state.get("review_score", 5.0)
        review_passed = state.get("review_passed", False)

        review_comments = state.get("review_comments", [])
        review_feedback = "\n".join(
            f"- [{c['severity'].upper()}] L{c['line']}: {c['message']}"
            for c in review_comments[:5]
        ) or "No significant review findings."

        result = chain.invoke({
            "code": code[:15000],
            "language": language,
            "description": description,
            "review_score": review_score,
            "review_passed": review_passed,
            "review_feedback": review_feedback,
        })

        test_plan_raw = result.get("test_plan", [])
        test_plan: list[TestPlan] = []
        for t in test_plan_raw:
            test_plan.append({
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "input_summary": t.get("input_summary", ""),
                "expected_behavior": t.get("expected_behavior", ""),
            })

        return {
            "test_plan": test_plan[:max_tests],
            "test_coverage_estimate": float(result.get("coverage_estimate", 0.6)),
            "current_node": "testing",
            "iteration": state.get("iteration", 0) + 1,
        }

    return testing_node


def testing_fallback(code: str, language: str = "python", max_tests: int = 15) -> dict:
    """
    Deterministic fallback: derive test cases from static analysis.

    Scans for functions, classes, and common patterns to generate test cases.
    """
    test_plan: list[TestPlan] = []
    lines = code.split("\n")

    if language == "python":
        func_pattern = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)")
        class_pattern = re.compile(r"^\s*class\s+(\w+)")

        for i, line in enumerate(lines):
            m = func_pattern.match(line)
            if m:
                func_name = m.group(1)
                params = m.group(2).strip()
                if func_name.startswith("_"):
                    continue

                param_list = [p.strip().split("=")[0].strip().split(":")[0].strip()
                              for p in params.split(",") if p.strip()] if params else []

                test_plan.append({
                    "name": f"test_{func_name}_happy_path",
                    "description": f"Test {func_name} with valid inputs",
                    "input_summary": f"Valid values for: {', '.join(param_list) if param_list else 'no params'}",
                    "expected_behavior": f"{func_name} returns expected result without errors",
                })
                if len(test_plan) >= max_tests:
                    break

                if param_list:
                    test_plan.append({
                        "name": f"test_{func_name}_empty_input",
                        "description": f"Test {func_name} with empty/null inputs",
                        "input_summary": f"Empty or None for: {param_list[0]}",
                        "expected_behavior": "Graceful handling of empty input",
                    })
                    if len(test_plan) >= max_tests:
                        break

            m2 = class_pattern.match(line)
            if m2:
                class_name = m2.group(1)
                test_plan.append({
                    "name": f"test_{class_name.lower()}_instantiation",
                    "description": f"Test {class_name} can be instantiated",
                    "input_summary": "Default constructor arguments",
                    "expected_behavior": f"{class_name} instance created successfully",
                })
                if len(test_plan) >= max_tests:
                    break

    total_defs = len(test_plan)
    coverage = min(0.85, 0.3 + total_defs * 0.05) if total_defs > 0 else 0.2

    return {
        "test_plan": test_plan[:max_tests],
        "test_coverage_estimate": round(coverage, 2),
        "current_node": "testing_fallback",
        "iteration": 1,
    }
