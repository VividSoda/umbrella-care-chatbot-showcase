"""App factory: lifespan (loads the model once), middleware, exception handlers, router includes."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src import health
from src.chatbot.predictor import Predictor
from src.chatbot.router import router as chatbot_router
from src.config import Settings, get_settings
from src.exceptions import register_exception_handlers


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Load once at startup. A missing or incompatible model file stops the
        # server here instead of failing on the first user request.
        app.state.predictor = Predictor.load(settings.model_path)
        yield
        app.state.predictor = None

    app = FastAPI(
        title="UmbrellaCare Chatbot",
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["POST", "GET"],
        allow_headers=["Content-Type"],
    )
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(chatbot_router)
    return app


app = create_app()
