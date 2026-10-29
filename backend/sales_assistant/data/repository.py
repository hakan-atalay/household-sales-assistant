from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import pandas as pd

from sales_assistant.domain.segments import ORDER
from sales_assistant.settings import Settings, get_settings


@dataclass(frozen=True)
class DatasetFacts:
    households: int
    categories: int
    spend_min: float
    spend_median: float
    spend_max: float


class HouseholdRepository:
    def __init__(self, households: pd.DataFrame, purchases: pd.DataFrame, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.households = self._ordered(households.set_index("household_id"))
        self.purchases = purchases[purchases["household_id"].isin(self.households.index)]

    @classmethod
    def from_files(cls, settings: Settings | None = None) -> HouseholdRepository:
        settings = settings or get_settings()
        for path in (settings.households_path, settings.purchases_path):
            if not path.exists():
                raise FileNotFoundError(f"{path} not found. Create it with: python -m sales_assistant.data.build --raw <folder>")
        return cls(pd.read_csv(settings.households_path), pd.read_csv(settings.purchases_path), settings)

    @staticmethod
    def _ordered(households: pd.DataFrame) -> pd.DataFrame:
        for column, labels in ORDER.items():
            households[column] = pd.Categorical(households[column].astype(str), categories=labels, ordered=True)
        return households

    @cached_property
    def bought(self) -> pd.DataFrame:
        table = pd.crosstab(self.purchases["household_id"], self.purchases["category"]) > 0
        return table.reindex(self.households.index, fill_value=False)

    @cached_property
    def spend_share(self) -> pd.DataFrame:
        spend = self.purchases.pivot_table(index="household_id", columns="category", values="spend", aggfunc="sum", fill_value=0)
        spend = spend.reindex(self.households.index, fill_value=0)
        return spend.div(self.households["total_spend"], axis=0) * 100

    @cached_property
    def facts(self) -> DatasetFacts:
        spend = self.households["total_spend"]
        return DatasetFacts(len(self.households), self.bought.shape[1], float(spend.min()), float(spend.median()), float(spend.max()))

    def facts_text(self) -> str:
        f = self.facts
        numbers = [f.households, f.categories, round(f.spend_min, 2), round(f.spend_median, 2), round(f.spend_max, 2), 2017,
                   self.settings.min_group_size]
        return " ".join(str(n) for n in numbers)

    def top_categories(self, n: int) -> list[str]:
        return self.bought.sum().sort_values(ascending=False).head(n).index.tolist()
