from __future__ import annotations

import re
from dataclasses import dataclass

NUMBER = re.compile(r"\d{1,3}(?:[.,]\d{3})+(?!\d)|\d+(?:[.,]\d+)?")
LIST_MARKER = re.compile(r"(?m)^\s*(?:[-*]\s*)?\d+[.)]\s")
STAT_CONSTANTS = "0.05 0.01"
IGNORED_INTEGERS = frozenset({10, 100})

UNSUPPORTED_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("gender", re.compile(r"\b(?:kadın\w*|erkek\w*|cinsiyet\w*|gender|female|male|women|men)\b")),
    ("english", re.compile(r"\bdataset\w*")),
    ("brand", re.compile(r"\b(?:marka\w*|brand\w*)\b")),
)


def _candidates(token: str) -> list[tuple[float, int]]:
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", token):
        return [(float(re.sub(r"[.,]", "", token)), 0), (float(token.replace(",", ".")), 3)]
    decimals = len(re.split(r"[.,]", token)[1]) if re.search(r"[.,]", token) else 0
    return [(float(token.replace(",", ".")), decimals)]


def _allowed_values(*texts: str) -> list[float]:
    values: list[float] = []
    for text in texts:
        for match in NUMBER.findall(text):
            values.extend(v for v, _ in _candidates(match))
    return values


def ungrounded_numbers(answer: str, evidence: str, question: str, facts: str) -> list[float]:
    allowed = _allowed_values(evidence, question, facts, STAT_CONSTANTS)
    bad: list[float] = []
    for token in NUMBER.findall(LIST_MARKER.sub("", answer)):
        options = _candidates(token)
        if any(d == 0 and v in IGNORED_INTEGERS for v, d in options):
            continue
        if any(abs(v - a) <= 0.5 * 10 ** (-d) + 1e-6 for v, d in options for a in allowed):
            continue
        bad.append(options[-1][0] if len(options) == 1 else options[0][0])
    return bad


def unsupported_terms(answer: str, question: str, evidence: str) -> list[str]:
    asked, shown, found = question.lower(), answer.lower(), set()
    available = evidence.lower()
    for _, pattern in UNSUPPORTED_PATTERNS:
        if pattern.search(asked) or pattern.search(available):
            continue
        found |= {m.group(0) for m in pattern.finditer(shown)}
    return sorted(found)


@dataclass(frozen=True)
class GroundingReport:
    numbers: list[float]
    terms: list[str]

    @property
    def ok(self) -> bool:
        return not (self.numbers or self.terms)


def check(answer: str, evidence: str, question: str, facts: str) -> GroundingReport:
    return GroundingReport(ungrounded_numbers(answer, evidence, question, facts), unsupported_terms(answer, question, evidence))