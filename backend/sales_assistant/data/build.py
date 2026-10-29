from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sales_assistant.domain import segments as seg
from sales_assistant.settings import Settings, get_settings

EXCLUDED_CATEGORIES = {"COUPON/MISC ITEMS"}


def _households(demographics: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    per_basket = lines.groupby("basket_id")["sales_value"].sum()
    baskets = lines.drop_duplicates("basket_id").set_index("basket_id")["household_id"]
    totals = lines.groupby("household_id").agg(total_spend=("sales_value", "sum"))
    totals["baskets"] = per_basket.groupby(baskets).size()
    totals["avg_basket"] = totals["total_spend"] / totals["baskets"]
    totals["categories_bought"] = lines.dropna(subset=["product_category"]).groupby("household_id")["product_category"].nunique()
    private = lines[lines["brand"] == "Private"].groupby("household_id")["sales_value"].sum()
    totals["private_share_pct"] = (private.reindex(totals.index).fillna(0) / totals["total_spend"] * 100)
    people = demographics.set_index("household_id").assign(
        income=lambda d: d["income"].map(seg.INCOME_BAND),
        household_size=lambda d: d["household_size"].astype(str),
        kids=lambda d: (d["kids_count"].astype(str) != "0").map({True: "Kids", False: "No kids"}),
    )[["age", "income", "household_size", "household_comp", "kids"]]
    return people.join(totals, how="inner").dropna().reset_index()


def build_tables(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    demographics = pd.read_csv(raw_dir / "demographics.csv")
    products = pd.read_csv(raw_dir / "products.csv", usecols=["product_id", "brand", "product_category"])
    transactions = pd.read_csv(
        raw_dir / "transactions.csv", usecols=["household_id", "basket_id", "product_id", "quantity", "sales_value"]
    )
    lines = (
        transactions[transactions["household_id"].isin(demographics["household_id"])]
        .query("sales_value > 0 and quantity > 0")
        .merge(products, on="product_id", how="left", validate="many_to_one")
    )
    households = _households(demographics, lines)
    purchases = (
        lines[~lines["product_category"].isin(EXCLUDED_CATEGORIES)]
        .dropna(subset=["product_category"])
        .groupby(["household_id", "product_category"], as_index=False)["sales_value"].sum()
        .rename(columns={"product_category": "category", "sales_value": "spend"})
    )
    return households, purchases[purchases["household_id"].isin(households["household_id"])]


def write_tables(raw_dir: Path, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    households, purchases = build_tables(raw_dir)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    households.round(4).to_csv(settings.households_path, index=False)
    purchases.round(4).to_csv(settings.purchases_path, index=False)
    print(f"{len(households)} households, {len(purchases)} household-category rows -> {settings.data_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the analysis tables from the exported Complete Journey CSVs.")
    parser.add_argument("--raw", type=Path, required=True, help="folder with transactions.csv, products.csv, demographics.csv")
    write_tables(parser.parse_args().raw)


if __name__ == "__main__":
    main()
