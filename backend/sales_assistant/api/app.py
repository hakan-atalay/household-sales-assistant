from __future__ import annotations

import json
from collections.abc import Iterator

from pydantic import BaseModel, Field, ValidationError
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.routing import Route

from sales_assistant.container import Services, build_services
from sales_assistant.domain.limitations import DATASET_TITLE, EXAMPLE_QUESTIONS
from sales_assistant.feedback.store import EXPORT_KINDS, MessageNotFound

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[dict] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    message_id: int
    rating: int
    reason: str | None = None
    correction: str | None = None


def _sse(events: Iterator[dict]) -> Iterator[str]:
    for event in events:
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def create_app(services: Services | None = None) -> Starlette:
    services = services or build_services()

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({
            "ollama": services.client.is_ready(),
            "model": services.client.model,
            "title": DATASET_TITLE,
            "examples": list(EXAMPLE_QUESTIONS),
            **services.store.stats(),
        })

    async def chat(request: Request):
        try:
            body = ChatRequest.model_validate(await request.json())
        except (ValidationError, json.JSONDecodeError) as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        question = body.question.strip()
        if not question:
            return JSONResponse({"detail": "question is empty"}, status_code=400)
        return StreamingResponse(
            _sse(services.chat.stream(question, body.history)), media_type="text/event-stream", headers=SSE_HEADERS
        )

    async def feedback(request: Request) -> JSONResponse:
        try:
            body = FeedbackRequest.model_validate(await request.json())
            services.store.set_feedback(body.message_id, body.rating, body.reason, body.correction)
        except MessageNotFound:
            return JSONResponse({"detail": "message not found"}, status_code=404)
        except (ValidationError, ValueError) as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        return JSONResponse({"ok": True, **services.store.stats()})

    async def export(request: Request) -> PlainTextResponse:
        kind = request.path_params["kind"]
        if kind not in EXPORT_KINDS:
            return PlainTextResponse(f"kind must be one of {EXPORT_KINDS}", status_code=404)
        return PlainTextResponse(services.store.export(kind))

    routes = [
        Route("/api/health", health),
        Route("/api/chat", chat, methods=["POST"]),
        Route("/api/feedback", feedback, methods=["POST"]),
        Route("/api/export/{kind}", export),
    ]
    middleware = [Middleware(CORSMiddleware, allow_origins=list(services.settings.cors_origins),
                             allow_methods=["*"], allow_headers=["*"])]
    return Starlette(routes=routes, middleware=middleware)
