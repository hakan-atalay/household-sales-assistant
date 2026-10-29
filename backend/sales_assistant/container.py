from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from sales_assistant.analytics.registry import ToolContext, registry
from sales_assistant.data.repository import HouseholdRepository
from sales_assistant.data.resolver import EntityResolver
from sales_assistant.feedback.store import FeedbackStore
from sales_assistant.llm.ollama import OllamaClient
from sales_assistant.llm.pipeline import ChatPipeline
from sales_assistant.llm.planner import Planner
from sales_assistant.service import ChatService
from sales_assistant.settings import Settings, get_settings


@dataclass(frozen=True)
class Services:
    settings: Settings
    repository: HouseholdRepository
    client: OllamaClient
    pipeline: ChatPipeline
    store: FeedbackStore
    chat: ChatService


def build_services(settings: Settings | None = None, client: OllamaClient | None = None) -> Services:
    settings = settings or get_settings()
    import_module("sales_assistant.analytics.tools")
    repository = HouseholdRepository.from_files(settings)
    context = ToolContext(repository, EntityResolver(repository), settings)
    client = client or OllamaClient(settings)
    pipeline = ChatPipeline(client, Planner(client, registry, repository), registry, context)
    store = FeedbackStore(settings.feedback_db)
    return Services(settings, repository, client, pipeline, store, ChatService(pipeline, store))
