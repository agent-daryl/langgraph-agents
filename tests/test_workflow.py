"""Tests for the multi-agent workflow (static fallback paths)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.code_review import review_fallback
from src.agents.testing import testing_fallback
from src.agents.documentation import documentation_fallback
from src.config import AppConfig, load_config
from src.workflow.engine import create_fallback_workflow


SAMPLE_CODE = '''\
import os
import json

def calculate_total(items, tax_rate=0.08):
    total = sum(item["price"] * item["quantity"] for item in items)
    tax = total * tax_rate
    return total + tax

def get_db_connection():
    password = "supersecret123"
    return f"postgresql://admin:{password}@localhost:5432/mydb"

def process_data(data):
    # TODO: add validation
    result = eval(data)
    return result

class DataProcessor:
    def __init__(self, config_path):
        self.config = json.load(open(config_path))

    def run(self, input_data):
        return {
            "status": "complete",
            "items_processed": len(input_data),
        }
'''


class TestReviewFallback:
    def test_returns_score(self):
        result = review_fallback(SAMPLE_CODE, "python")
        assert 0.0 <= result["review_score"] <= 10.0

    def test_detects_hardcoded_password(self):
        result = review_fallback(SAMPLE_CODE, "python")
        severities = [c["severity"] for c in result["review_comments"]]
        assert "error" in severities

    def test_detects_eval(self):
        result = review_fallback(SAMPLE_CODE, "python")
        messages = [c["message"].lower() for c in result["review_comments"]]
        assert any("eval" in m for m in messages)

    def test_marks_review_failed(self):
        result = review_fallback(SAMPLE_CODE, "python")
        assert result["review_passed"] is False

    def test_clean_code(self):
        clean = "def hello(name):\n    return f'Hello, {name}'\n"
        result = review_fallback(clean, "python")
        assert result["review_passed"] is True
        assert result["review_score"] == 10.0


class TestTestingFallback:
    def test_generates_tests(self):
        result = testing_fallback(SAMPLE_CODE, "python")
        assert len(result["test_plan"]) > 0

    def test_coverage_estimate_valid(self):
        result = testing_fallback(SAMPLE_CODE, "python")
        assert 0.0 <= result["test_coverage_estimate"] <= 1.0

    def test_test_names_meaningful(self):
        result = testing_fallback(SAMPLE_CODE, "python")
        for tc in result["test_plan"]:
            assert tc["name"].startswith("test_")
            assert len(tc["description"]) > 0


class TestDocumentationFallback:
    def test_generates_sections(self):
        result = documentation_fallback(SAMPLE_CODE, "python", "Sample module")
        assert len(result["doc_sections"]) > 0

    def test_includes_overview(self):
        result = documentation_fallback(SAMPLE_CODE, "python", "Sample module")
        headings = [s["heading"] for s in result["doc_sections"]]
        assert any("Overview" in h for h in headings)

    def test_summary_nonempty(self):
        result = documentation_fallback(SAMPLE_CODE, "python", "A sample module")
        assert len(result["doc_summary"]) > 0


class TestConfig:
    def test_defaults(self):
        cfg = AppConfig()
        assert cfg.llm.api_base == "http://10.10.0.20:8000/v1"
        assert cfg.agents.max_review_comments == 10

    def test_env_override(self):
        os.environ["LLM_API_BASE"] = "http://test:9999/v1"
        os.environ["LLM_MODEL"] = "test-model"
        try:
            cfg = AppConfig()
            assert cfg.llm.api_base == "http://test:9999/v1"
            assert cfg.llm.model == "test-model"
        finally:
            del os.environ["LLM_API_BASE"]
            del os.environ["LLM_MODEL"]


class TestFallbackWorkflow:
    def test_end_to_end(self):
        config = AppConfig()
        graph = create_fallback_workflow(config)

        state = graph.invoke({
            "code": SAMPLE_CODE,
            "language": "python",
            "description": "Test module",
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
        })

        assert state["review_score"] > 0
        assert len(state["review_comments"]) > 0
        assert len(state["doc_sections"]) > 0
        assert state["iteration"] > 0
