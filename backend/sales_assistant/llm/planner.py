from __future__ import annotations

import json
from dataclasses import dataclass

from sales_assistant.analytics.registry import ToolRegistry
from sales_assistant.data.repository import HouseholdRepository
from sales_assistant.llm.ollama import OllamaClient
from sales_assistant.llm.prompts import planner_prompt

MAX_CALLS = 3
HISTORY_QUESTIONS = 2


class PlanningError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict

    def as_dict(self) -> dict:
        return {"name": self.name, "args": self.args}


def plan_schema(registry: ToolRegistry) -> dict:
    alternatives = [
        {
            "type": "object",
            "properties": {"tool": {"enum": [spec.name]}, "args": spec.parameters_schema()},
            "required": ["tool", "args"],
        }
        for spec in registry
    ]
    return {
        "type": "object",
        "required": ["calls"],
        "properties": {"calls": {"type": "array", "maxItems": MAX_CALLS, "items": {"anyOf": alternatives}}},
    }


class Planner:
    def __init__(self, client: OllamaClient, registry: ToolRegistry, repository: HouseholdRepository):
        self.client = client
        self.registry = registry
        self.system_prompt = planner_prompt(repository, registry)
        self.schema = plan_schema(registry)

    def plan(self, question: str, history: list[dict] | None = None) -> list[ToolCall]:
        messages = [{"role": "system", "content": self.system_prompt}]
        earlier = [m["content"] for m in (history or []) if m.get("role") == "user"][-HISTORY_QUESTIONS:]
        if earlier:
            messages.append({
                "role": "system",
                "content": "Earlier user questions, ONLY to fill in what a follow-up leaves out such as the place or "
                           "category; NEVER plan calls for them: " + " | ".join(earlier),
            })
        messages.append({"role": "user", "content": f"Plan calls ONLY for this question: {question}"})
        raw = self.client.chat(messages, temperature=0.0, schema=self.schema)
        try:
            calls = json.loads(raw)["calls"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise PlanningError(f"The model returned an unreadable plan: {raw[:200]}") from exc
        return [ToolCall(c["tool"], c.get("args") or {}) for c in calls if c.get("tool") in self.registry][:MAX_CALLS]
