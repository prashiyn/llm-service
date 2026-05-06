from __future__ import annotations

import json
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from llm_service.llms import EmbeddingClient, LLMClient, get_llm_config

router = APIRouter()
_llm_client: LLMClient | None = None
_embedding_client: EmbeddingClient | None = None


def _client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def _embeddings_client() -> EmbeddingClient:
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client


class JsonSchemaPayload(BaseModel):
    name: str
    schema_: dict[str, Any] = Field(..., alias="schema")
    strict: bool | None = None


class ResponseFormatJsonObject(BaseModel):
    type: Literal["json_object"]


class ResponseFormatJsonSchema(BaseModel):
    type: Literal["json_schema"]
    json_schema: JsonSchemaPayload


class CompletionRequest(BaseModel):
    provider: str = Field(..., description="Provider alias: groq, ollama, openai, anthropic, tencent")
    messages: list[dict[str, str]] = Field(..., min_length=1)
    model: str | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    response_format: ResponseFormatJsonObject | ResponseFormatJsonSchema | None = None


class CompletionResponse(BaseModel):
    content: str
    parsed: Any | None = None


class ModelsResponse(BaseModel):
    default_model: str
    fallback_model: str
    models: list[str]


class EmbeddingRequest(BaseModel):
    provider: str
    input: str | list[str]
    model: str | None = None
    encoding_format: Literal["float", "base64"] | None = None
    dimensions: int | None = Field(None, ge=1)
    input_type: str | None = None
    user: str | None = None


class EmbeddingDataItem(BaseModel):
    object: str
    index: int
    embedding: list[float] | str


class EmbeddingResponse(BaseModel):
    object: str
    model: str
    data: list[EmbeddingDataItem]
    usage: dict[str, Any] | None = None


@router.post("/complete", response_model=CompletionResponse)
async def completion(req: CompletionRequest) -> CompletionResponse:
    try:
        content = await _client().acomplete_with_fallback(
            req.messages,
            model=req.model,
            reasoning_effort=req.reasoning_effort,
            response_format=req.response_format.model_dump(by_alias=True) if req.response_format else None,
        )
        parsed: Any | None = None
        if req.response_format is not None:
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = None
        return CompletionResponse(content=content, parsed=parsed)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@router.get("/models", response_model=ModelsResponse)
async def models() -> ModelsResponse:
    cfg = get_llm_config()
    return ModelsResponse(
        default_model=cfg.get("default_model", "gpt-4o-mini"),
        fallback_model=cfg.get("fallback_model", "gpt-3.5-turbo"),
        models=cfg.get("models", []),
    )


@router.post("/embeddings", response_model=EmbeddingResponse)
async def embeddings(req: EmbeddingRequest) -> EmbeddingResponse:
    try:
        result = await _embeddings_client().aembed(
            req.input,
            model=req.model,
            encoding_format=req.encoding_format,
            dimensions=req.dimensions,
            input_type=req.input_type,
            user=req.user,
        )
        return EmbeddingResponse(
            object=str(result.get("object", "list")),
            model=str(result.get("model", req.model or "")),
            data=[
                EmbeddingDataItem(
                    object=str(item.get("object", "embedding")),
                    index=int(item.get("index", idx)),
                    embedding=item.get("embedding"),
                )
                for idx, item in enumerate(result.get("data", []))
            ],
            usage=result.get("usage"),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
