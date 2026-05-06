# llm-service

Standalone `/llm` gateway extracted from the monolith.

## Run

```bash
uv sync --extra dev
PYTHONPATH=src uv run python -m llm_service.main
```

## OpenAPI

Generate:

```bash
PYTHONPATH=src uv run python -c "import json; from llm_service.main import app; open('openapi.json','w').write(json.dumps(app.openapi(), indent=2))"
```
