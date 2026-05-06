from contextlib import asynccontextmanager

from fastapi import FastAPI

from llm_service.config import get_settings
from llm_service.routers import health, llm


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        description="Standalone LLM gateway service.",
        version="0.1.0",
        debug=settings.debug,
        lifespan=lifespan,
    )
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(llm.router, prefix="/llm", tags=["llm"])
    return app


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run("llm_service.main:app", host="0.0.0.0", port=8001, reload=True)


if __name__ == "__main__":
    run()
