from __future__ import annotations

import json
import re


class ScriptedClient:
    model = "scripted"

    def __init__(self, hallucinate: bool = False, plan_failure: bool = False, stubborn: bool = False):
        self.hallucinate = hallucinate
        self.stubborn = stubborn
        self.plan_failure = plan_failure
        self.rewrites = 0

    def is_ready(self) -> bool:
        return True

    def chat(self, messages, *, temperature=0.0, schema=None) -> str:
        if schema is not None:
            if self.plan_failure:
                return "not json"
            return json.dumps({"calls": self._plan(messages[-1]["content"])})
        self.rewrites += 1
        return "Fark 777 oldu." if self.stubborn else self._faithful(messages)

    def stream(self, messages, *, temperature=0.2):
        text = "Bölgelerde fiyat 777 oldu, %88 fark var." if self.hallucinate or self.stubborn else self._faithful(messages)
        for word in text.split(" "):
            yield word + " "

    @staticmethod
    def _plan(prompt: str) -> list[dict]:
        text = prompt.lower()
        if "merhaba" in text:
            return []
        if "cinsiyet" in text or "kadın" in text:
            return [{"tool": "dataset_overview", "args": {}}]
        if "harcama" in text:
            return [{"tool": "compare_groups", "args": {"metric": "total_spend", "by": "income"}}]
        if "bebek" in text:
            return [{"tool": "category_buyer_profile", "args": {"category": "diapers", "by": "age"}}]
        return [{"tool": "category_preferences", "args": {"by": "age"}}]

    @staticmethod
    def _faithful(messages) -> str:
        user = messages[-1]["content"] if messages[-1]["role"] == "user" and "[TOOL RESULTS" in messages[-1]["content"] else messages[-2]["content"]
        match = re.search(r"^- (.+)$", user, re.M)
        return match.group(1) if match else "Bu soruya uygun bir analiz bulunmuyor."
