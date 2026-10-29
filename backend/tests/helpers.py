from __future__ import annotations

import tempfile
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from sales_assistant.container import Services, build_services
from sales_assistant.settings import get_settings
from tests.fakes import ScriptedClient


def data_available() -> bool:
    return get_settings().households_path.exists()


@lru_cache(maxsize=1)
def shared_repository():
    from sales_assistant.data.repository import HouseholdRepository
    return HouseholdRepository.from_files()


def make_services(client: ScriptedClient | None = None) -> Services:
    settings = replace(get_settings(), feedback_db=Path(tempfile.mkdtemp()) / "feedback.db")
    return build_services(settings, client or ScriptedClient())
