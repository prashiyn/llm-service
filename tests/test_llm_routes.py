from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


def test_llm_models_lists_config(client: TestClient) -> None:
    r = client.get("/llm/models")
    assert r.status_code == 200
    body = r.json()
    assert "default_model" in body
    assert "fallback_model" in body
    assert isinstance(body["models"], list)


def test_llm_complete_success_mocked(client: TestClient) -> None:
    mock_llm = MagicMock()
    mock_llm.acomplete_with_fallback = AsyncMock(return_value="mocked assistant reply")
    with patch("llm_service.routers.llm._client", return_value=mock_llm):
        r = client.post(
            "/llm/complete",
            json={"provider": "openai", "messages": [{"role": "user", "content": "ping"}]},
        )
    assert r.status_code == 200
    assert r.json() == {"content": "mocked assistant reply", "parsed": None}


def test_llm_embeddings_success_mocked(client: TestClient) -> None:
    mock_embed = MagicMock()
    mock_embed.aembed = AsyncMock(
        return_value={
            "object": "list",
            "model": "openai/text-embedding-3-small",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "usage": {"total_tokens": 5},
        }
    )
    with patch("llm_service.routers.llm._embeddings_client", return_value=mock_embed):
        r = client.post(
            "/llm/embeddings",
            json={"provider": "openai", "input": ["hello"], "model": "openai/text-embedding-3-small"},
        )
    assert r.status_code == 200
    assert r.json()["data"][0]["index"] == 0
