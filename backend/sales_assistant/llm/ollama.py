from __future__ import annotations

import json
from collections.abc import Iterator

import requests

from sales_assistant.settings import Settings

Message = dict[str, str]


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, settings: Settings, session: requests.Session | None = None):
        self.settings = settings
        self.session = session or requests.Session()

    @property
    def model(self) -> str:
        return self.settings.model

    def is_ready(self) -> bool:
        try:
            return self.session.get(f"{self.settings.ollama_url}/api/tags", timeout=1.5).ok
        except requests.RequestException:
            return False

    def _payload(self, messages: list[Message], *, stream: bool, temperature: float, schema: dict | None) -> dict:
        payload: dict = {
            "model": self.settings.model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": temperature, "num_ctx": self.settings.num_ctx},
        }
        if schema is not None:
            payload["format"] = schema
        if "qwen3" in self.settings.model:
            payload["think"] = False
        return payload

    def chat(self, messages: list[Message], *, temperature: float = 0.0, schema: dict | None = None) -> str:
        try:
            response = self.session.post(
                f"{self.settings.ollama_url}/api/chat",
                json=self._payload(messages, stream=False, temperature=temperature, schema=schema),
                timeout=300,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(str(exc)) from exc
        return response.json()["message"]["content"]

    def stream(self, messages: list[Message], *, temperature: float = 0.2) -> Iterator[str]:
        try:
            with self.session.post(
                f"{self.settings.ollama_url}/api/chat",
                json=self._payload(messages, stream=True, temperature=temperature, schema=None),
                stream=True,
                timeout=300,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    piece = chunk.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        return
        except requests.RequestException as exc:
            raise OllamaError(str(exc)) from exc
