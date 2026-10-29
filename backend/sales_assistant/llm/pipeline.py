from __future__ import annotations

from collections.abc import Iterator

from sales_assistant.analytics.registry import ToolContext, ToolRegistry
from sales_assistant.data.repository import HouseholdRepository
from sales_assistant.llm import grounding
from sales_assistant.llm.ollama import OllamaClient
from sales_assistant.llm.planner import Planner, ToolCall
from sales_assistant.llm.prompts import answer_prompt

Event = dict

FALLBACK_INTRO = "Cevabı güvenli biçimde yazamadım; doğrulanmış analiz bulguları aynen şöyle:"
NO_TOOL_TEXT = "(none: no tool applies to this question)"


def findings_text(results: list[dict]) -> str:
    return "\n\n".join(f"[{r['tool']}]\n" + "\n".join(f"- {f}" for f in r["findings"]) for r in results)


class ChatPipeline:
    def __init__(self, client: OllamaClient, planner: Planner, registry: ToolRegistry, context: ToolContext):
        self.client = client
        self.planner = planner
        self.registry = registry
        self.context = context
        self.repository: HouseholdRepository = context.repository
        self.facts = self.repository.facts_text()

    def _execute(self, calls: list[ToolCall]) -> tuple[list[dict], dict | None]:
        results, chart = [], None
        for call in calls:
            outcome = self.registry.execute(self.context, call.name, call.args)
            chart = chart or outcome.pop("chart", None)
            findings = outcome.get("findings") or [f"Analiz yapılamadı: {outcome['error']}"] + (
                [f"Şunu mu demek istediniz: {', '.join(outcome['did_you_mean'])}"] if outcome.get("did_you_mean") else [])
            results.append({"tool": call.name, "findings": findings})
        return results, chart

    def _user_message(self, question: str, evidence: str | None) -> str:
        return f"{question}\n\n[FINDINGS - answer only from these]\n{evidence or NO_TOOL_TEXT}"

    def stream(self, question: str, history: list[dict] | None = None, examples: list[tuple[str, str]] | None = None) -> Iterator[Event]:
        yield {"type": "status", "text": "planning"}
        calls = self.planner.plan(question, history)
        results, chart = self._execute(calls)
        yield {"type": "tools", "calls": [c.as_dict() for c in calls]}
        if chart:
            yield {"type": "chart", "chart": chart}

        evidence = findings_text(results)
        messages = [
            {"role": "system", "content": answer_prompt(self.repository, examples)},
            {"role": "user", "content": self._user_message(question, evidence)},
        ]
        yield {"type": "status", "text": "writing"}
        answer = ""
        for piece in self.client.stream(messages, temperature=0.2):
            answer += piece
            yield {"type": "token", "text": piece}
        answer = answer.strip()
        fallback = False

        report = grounding.check(answer, evidence, question, self.facts)
        if not report.ok:
            yield {"type": "status", "text": "checking"}
            answer = self._rewrite(messages, answer, report)
            report = grounding.check(answer, evidence, question, self.facts)
            if not report.ok:
                answer = FALLBACK_INTRO + "\n" + "\n".join(f for r in results for f in r["findings"])
                report = grounding.GroundingReport([], [])
                fallback = True
            yield {"type": "replace", "text": answer}
        yield {
            "type": "done",
            "meta": {
                "fallback": fallback,
                "grounded": report.ok,
                "ungrounded_numbers": report.numbers,
                "unsupported_terms": report.terms,
                "model": self.client.model,
            },
        }

    def _rewrite(self, messages: list[dict], answer: str, report: grounding.GroundingReport) -> str:
        problems = []
        if report.numbers:
            problems.append(f"These numbers are NOT in the findings: {report.numbers}. Use only numbers from the findings "
                            "(rounding is fine); do not calculate new ones.")
        if report.terms:
            problems.append(f"You used {report.terms}. The data has no gender or brand information, so remove such statements, "
                            "and write 'veri' instead of 'dataset'.")
        retry = [*messages, {"role": "assistant", "content": answer},
                 {"role": "user", "content": "Rewrite the answer. " + " ".join(problems)}]
        return self.client.chat(retry, temperature=0.1).strip()

    def answer(self, question: str, history: list[dict] | None = None, examples: list[tuple[str, str]] | None = None) -> dict:
        text, tools, chart, meta = "", [], None, {}
        for event in self.stream(question, history, examples):
            kind = event["type"]
            if kind == "token":
                text += event["text"]
            elif kind == "replace":
                text = event["text"]
            elif kind == "tools":
                tools = event["calls"]
            elif kind == "chart":
                chart = event["chart"]
            elif kind == "done":
                meta = event["meta"]
        return {"answer": text.strip(), "tools": tools, "chart": chart, **meta}