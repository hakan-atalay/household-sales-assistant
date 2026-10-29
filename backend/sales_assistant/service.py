from __future__ import annotations

import logging
from collections.abc import Iterator

from sales_assistant.feedback.store import FeedbackStore
from sales_assistant.llm.ollama import OllamaError
from sales_assistant.llm.pipeline import ChatPipeline, Event
from sales_assistant.llm.planner import PlanningError

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, pipeline: ChatPipeline, store: FeedbackStore):
        self.pipeline = pipeline
        self.store = store

    def stream(self, question: str, history: list[dict] | None = None) -> Iterator[Event]:
        answer, calls = "", []
        try:
            for event in self.pipeline.stream(question, history, self.store.liked_examples()):
                kind = event["type"]
                if kind == "token":
                    answer += event["text"]
                elif kind == "replace":
                    answer = event["text"]
                elif kind == "tools":
                    calls = event["calls"]
                elif kind == "done":
                    meta = event["meta"]
                    message_id = self.store.save_message(question, answer.strip(), calls, meta["model"], meta["grounded"])
                    event = {**event, "message_id": message_id}
                yield event
        except (OllamaError, PlanningError) as exc:
            yield {"type": "error", "text": f"{type(exc).__name__}: {exc}"}
        except Exception as exc:
            logger.exception("Unexpected failure while answering")
            yield {"type": "error", "text": f"{type(exc).__name__}: {exc}"}
