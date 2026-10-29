from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _series(series: Mapping[str, Sequence[float | None]]) -> list[dict[str, Any]]:
    return [{"name": name, "values": [None if v is None else round(float(v), 2) for v in values]}
            for name, values in series.items()]


def bar_chart(title: str, x: Sequence[str], series: Mapping[str, Sequence[float | None]], unit: str) -> dict:
    return {"type": "bar", "title": title, "unit": unit, "x": [str(v) for v in x], "series": _series(series)}

