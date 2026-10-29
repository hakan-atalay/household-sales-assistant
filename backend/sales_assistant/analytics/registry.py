from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, ValidationError

from sales_assistant.data.repository import HouseholdRepository
from sales_assistant.data.resolver import EntityNotFound, EntityResolver
from sales_assistant.settings import Settings


class ToolError(Exception):
    def __init__(self, message: str, **details: Any):
        super().__init__(message)
        self.details = details


@dataclass(frozen=True)
class ToolContext:
    repository: HouseholdRepository
    resolver: EntityResolver
    settings: Settings


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[ToolContext, Any], dict]

    def parameters_schema(self) -> dict:
        return compact_schema(self.args_model.model_json_schema())

    def signature(self) -> str:
        schema = self.parameters_schema()
        required = set(schema.get("required", []))
        parts = []
        for name, prop in schema.get("properties", {}).items():
            options = prop.get("enum") or next((p["enum"] for p in prop.get("anyOf", []) if "enum" in p), None)
            kind = "|".join(map(str, options)) if options else _type_name(prop)
            parts.append(f"{name}: {kind}{'' if name in required else '?'}")
        return f"{self.name}({', '.join(parts)})"


def _type_name(prop: dict) -> str:
    if "type" in prop:
        return prop["type"]
    names = [p.get("type") for p in prop.get("anyOf", []) if p.get("type") != "null"]
    return names[0] if names else "any"


def compact_schema(schema: dict) -> dict:
    if isinstance(schema, dict):
        return {k: compact_schema(v) for k, v in schema.items() if k not in {"title", "default"}}
    if isinstance(schema, list):
        return [compact_schema(v) for v in schema]
    return schema


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return round(number, 3) if abs(number) >= 1 else round(number, 5)
    return value


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, name: str, description: str, args_model: type[BaseModel]):
        def decorator(handler: Callable[[ToolContext, Any], dict]):
            self._tools[name] = ToolSpec(name, description, args_model, handler)
            return handler

        return decorator

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools)

    def execute(self, context: ToolContext, name: str, raw_args: dict | None) -> dict:
        spec = self._tools.get(name)
        if spec is None:
            return {"error": f"Unknown tool '{name}'", "available_tools": self.names()}
        try:
            args = spec.args_model.model_validate(raw_args or {})
            return jsonable(spec.handler(context, args))
        except ValidationError as exc:
            return {"error": "Invalid arguments", "details": [e["msg"] for e in exc.errors()]}
        except EntityNotFound as exc:
            return {"error": f"'{exc.text}' not found in the data", "did_you_mean": exc.suggestions}
        except ToolError as exc:
            return {"error": str(exc), **jsonable(exc.details)}


registry = ToolRegistry()
