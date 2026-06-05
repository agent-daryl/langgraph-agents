# LangGraph Multi-Agent Workflow

A production-ready multi-agent system built with [LangGraph](https://langchain.github.io/langgraph/) that orchestrates three specialized AI agents for automated code analysis.

## Architecture

```
┌─────────────┐     ┌────────────────┐     ┌─────────────────┐
│  Code Input  │────▶│ Code Review    │────▶│  Testing        │
│  (POST /     │     │  Agent         │     │  Agent          │
│   analyze)   │     │                │     │                 │
└─────────────┘     └───┬────────────┘     └────────┬────────┘
                        │                           │
                    (review failed)                │
                        │                           │
                        ▼                           ▼
              ┌─────────────────┐           ┌─────────────────┐
              │ Documentation   │◀──────────│ Documentation   │
              │  Agent          │           │  Agent          │
              │  (skip tests)   │           │                 │
              └────────┬────────┘           └────────┬────────┘
                       │                             │
                       └─────────────┬───────────────┘
                                     ▼
                           ┌────────────────┐
                           │   JSON Output  │
                           │  (Analysis)    │
                           └────────────────┘
```

## Agents

| Agent | Purpose | Key Capabilities |
|---|---|---|
| **Code Review** | Static + LLM analysis | Security, correctness, style, performance |
| **Testing** | Test plan generation | Happy path, edge cases, error handling |
| **Documentation** | Auto-documentation | API reference, usage examples, architecture |

## Quick Start

### Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

### API Server

```bash
uvicorn src.serving:app --host 0.0.0.0 --port 8080
```

### Docker

```bash
docker build -t langgraph-agents .
docker run -p 8080:8080 \
  -e LLM_API_BASE=http://your-llm:8000/v1 \
  langgraph-agents
```

### OpenShift Deployment

```bash
oc new-app langgraph-agents --name=langgraph-agents
oc expose svc/langgraph-agents
```

## API

### `POST /analyze`

Run the full multi-agent analysis pipeline.

```bash
curl -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def hello(name):\n    return f\"Hello, {name}\"",
    "language": "python",
    "description": "A greeting function",
    "use_llm": false
  }'
```

### `GET /health`

Health check for orchestrators.

### `GET /config`

View current configuration.

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `LLM_API_BASE` | `http://10.10.0.20:8000/v1` | OpenAI-compatible endpoint |
| `LLM_MODEL` | `qwen3.6:27b_256k` | Model name |
| `LLM_API_KEY` | `not-needed` | API key (for local, placeholder) |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature |

## Project Structure

```
langgraph_agents/
  src/
    agents/
      code_review.py    # Review agent + fallback
      testing.py        # Testing agent + fallback
      documentation.py  # Documentation agent + fallback
    workflow/
      engine.py         # LangGraph graph builder
    state.py            # Typed workflow state
    config.py           # Configuration management
    serving.py          # FastAPI application
  tests/
    test_workflow.py    # Unit + integration tests
  Dockerfile            # Container definition
  requirements.txt      # Dependencies
```

## Design Decisions

- **Fallback-first**: Every agent has a deterministic fallback for offline/CI use
- **Typed state**: `WorkflowState` uses `TypedDict` for type-safe graph transitions
- **Factory pattern**: Agent creation is decoupled from graph construction
- **OpenShift-ready**: Stateless design, external checkpoint support, health checks

## Stack

- LangGraph 0.x (state machine orchestration)
- FastAPI (API layer)
- Pydantic v2 (validation)
- OpenAI-compatible LLM (llama.cpp backend)

## Author

agent-daryl (AI agent) — built for Daryl Allen's MLOps portfolio
