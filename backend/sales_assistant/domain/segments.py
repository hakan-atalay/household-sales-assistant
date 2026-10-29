from __future__ import annotations

from typing import Literal

Dimension = Literal["age", "income", "household_size", "household_comp", "kids"]

AGE = ("19-24", "25-34", "35-44", "45-54", "55-64", "65+")
INCOME = ("Under 35K", "35-74K", "75-124K", "125K+")
HOUSEHOLD_SIZE = ("1", "2", "3", "4", "5+")
HOUSEHOLD_COMP = ("1 Adult No Kids", "2 Adults No Kids", "1 Adult Kids", "2 Adults Kids")
KIDS = ("No kids", "Kids")

ORDER: dict[str, tuple[str, ...]] = {
    "age": AGE,
    "income": INCOME,
    "household_size": HOUSEHOLD_SIZE,
    "household_comp": HOUSEHOLD_COMP,
    "kids": KIDS,
}

INCOME_BAND = {
    "Under 15K": "Under 35K", "15-24K": "Under 35K", "25-34K": "Under 35K",
    "35-49K": "35-74K", "50-74K": "35-74K",
    "75-99K": "75-124K", "100-124K": "75-124K",
    "125-149K": "125K+", "150-174K": "125K+", "175-199K": "125K+", "200-249K": "125K+", "250K+": "125K+",
}

SUBJECT_TR: dict[str, str] = {
    "age": "Yaş grupları arasında",
    "income": "Gelir grupları arasında",
    "household_size": "Hane büyüklükleri arasında",
    "household_comp": "Hane tipleri arasında",
    "kids": "Çocuklu ve çocuksuz haneler arasında",
}

COMP_TR = {
    "1 Adult No Kids": "1 yetişkin, çocuksuz", "2 Adults No Kids": "2 yetişkin, çocuksuz",
    "1 Adult Kids": "1 yetişkin, çocuklu", "2 Adults Kids": "2 yetişkin, çocuklu",
}

ALIASES: dict[str, dict[str, str]] = {
    "age": {"genc": "19-24", "yasli": "65+", "emekli": "65+", "orta yas": "45-54"},
    "income": {"dusuk": "Under 35K", "orta": "35-74K", "yuksek": "125K+"},
    "kids": {"cocuklu": "Kids", "cocuksuz": "No kids", "cocuk var": "Kids", "cocuk yok": "No kids"},
}


def segment_label(dimension: str, value: str) -> str:
    if dimension == "age":
        return f"{value} yaş grubu"
    if dimension == "income":
        return f"{value} yıllık gelirli haneler"
    if dimension == "household_size":
        return f"{value} kişilik haneler"
    if dimension == "household_comp":
        return f"{value} ({COMP_TR[value]})"
    return "çocuklu haneler" if value == "Kids" else "çocuksuz haneler"
