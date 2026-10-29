from __future__ import annotations

import pandas as pd


def fmt(value: float) -> str:
    return f"{value:.2f}" if abs(value) < 10 else f"{value:.1f}"


def percent(value: float) -> str:
    return f"%{value:.1f}" if value >= 1 else f"%{value:.2f}"


def standouts(rows: pd.DataFrame, direction: str, limit: int) -> str:
    chosen = rows[rows["direction"] == direction].assign(size=lambda d: d["gap_pp"].abs()).nlargest(limit, "size")
    return "; ".join(
        f"{r.category}: bütçe payı {percent(r.share_pct)}, diğer haneler {percent(r.rest_share_pct)}" for r in chosen.itertuples()
    )
