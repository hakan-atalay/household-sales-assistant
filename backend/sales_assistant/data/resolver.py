from __future__ import annotations

import difflib

from sales_assistant.data.repository import HouseholdRepository
from sales_assistant.domain.segments import ALIASES, ORDER
from sales_assistant.domain.text import fold

FUZZY_CUTOFF = 0.62
AUTO_ACCEPT_RATIO = 0.8


class EntityNotFound(LookupError):
    def __init__(self, text: str, suggestions: list[str]):
        super().__init__(f"No match for '{text}'")
        self.text = text
        self.suggestions = suggestions


def _resolve(text: str, candidates: dict[str, str]) -> str:
    query = fold(text)
    if query in candidates:
        return candidates[query]
    contained = {value for key, value in candidates.items() if len(query) >= 3 and query in key}
    if len(contained) == 1:
        return contained.pop()
    close = list(dict.fromkeys(
        candidates[k] for k in difflib.get_close_matches(query, list(candidates), n=5, cutoff=FUZZY_CUTOFF)
    ))
    if close and (len(close) == 1 or difflib.SequenceMatcher(None, query, fold(close[0])).ratio() >= AUTO_ACCEPT_RATIO):
        return close[0]
    raise EntityNotFound(text, close or sorted(contained)[:5])


class EntityResolver:
    def __init__(self, repository: HouseholdRepository):
        self._categories = {fold(c): c for c in repository.bought.columns}
        self._segments = {
            dimension: {fold(v): v for v in labels} | {fold(k): v for k, v in ALIASES.get(dimension, {}).items()}
            for dimension, labels in ORDER.items()
        }

    def category(self, text: str) -> str:
        return _resolve(text, self._categories)

    def segment(self, dimension: str, text: str) -> str:
        return _resolve(text, self._segments[dimension])
