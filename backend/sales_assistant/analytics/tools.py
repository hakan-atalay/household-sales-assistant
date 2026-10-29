from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from sales_assistant.analytics import evidence as ev
from sales_assistant.analytics.charts import bar_chart
from sales_assistant.analytics.narration import fmt, percent, standouts
from sales_assistant.analytics.registry import ToolContext, ToolError, registry
from sales_assistant.domain.limitations import LIMITATIONS
from sales_assistant.domain.segments import ORDER, SUBJECT_TR, Dimension, segment_label

Metric = Literal["total_spend", "baskets", "avg_basket", "categories_bought", "private_share_pct"]

METRIC_TR = {
    "total_spend": "yıllık toplam harcama", "baskets": "alışveriş (sepet) sayısı", "avg_basket": "ortalama sepet tutarı",
    "categories_bought": "alınan kategori çeşidi", "private_share_pct": "özel marka harcama payı (%)",
}
DIMENSION_NAME_TR = {"age": "yaş grubu", "income": "gelir grubu", "household_size": "hane büyüklüğü",
                     "household_comp": "hane tipi", "kids": "çocuk durumu"}
SEGMENT_METRICS = ("total_spend", "avg_basket", "categories_bought", "private_share_pct")


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _segment_size_check(ctx: ToolContext, size: int, label: str) -> None:
    if size < ctx.settings.min_group_size:
        raise ToolError(f"Too few households ({size}) in {label} to analyse reliably", minimum=ctx.settings.min_group_size)


class OverviewArgs(ToolArgs):
    pass


@registry.register(
    "dataset_overview",
    "Describe what data is available: households, period, demographic fields, categories and the limitations. "
    "Use for 'what data do you have?', 'what can you analyse?' or questions about gender or other missing fields.",
    OverviewArgs,
)
def dataset_overview(ctx: ToolContext, args: OverviewArgs) -> dict:
    facts, households = ctx.repository.facts, ctx.repository.households
    sizes = [
        f"{DIMENSION_NAME_TR[d].capitalize()}: " + ", ".join(f"{v} ({int((households[d] == v).sum())} hane)" for v in labels)
        for d, labels in ORDER.items()
    ]
    return {"findings": [
        f"Veri: {facts.households} hane, {facts.categories} ürün kategorisi, 2017 yılı bir ABD market zinciri alışverişleri.",
        "Demografi: yaş grubu, yıllık gelir, hane büyüklüğü, hane tipi ve çocuk durumu. Cinsiyet bilgisi YOKTUR; marka adı da yoktur "
        "(yalnızca ulusal marka / özel marka ayrımı vardır).",
        *sizes,
        f"Hane başına yıllık harcama medyanı {fmt(facts.spend_median)} (en az {fmt(facts.spend_min)}, en çok {fmt(facts.spend_max)}).",
        "Yapılabilen analizler: demografiye göre kategori tercihi, bir kategoriyi kimlerin aldığı, harcama/sepet/özel marka "
        "karşılaştırması, bir demografik grubun profili, harcamayı en çok etkileyen özellik.",
    ], "limitations": list(LIMITATIONS)}


class PreferenceArgs(ToolArgs):
    by: Dimension = "age"
    top_n: int = Field(3, ge=1, le=5)


@registry.register(
    "category_preferences",
    "Which product categories each demographic group (age, income, household size, household type, kids) spends a LARGER share "
    "of its budget on than other households, with a significance test. Use for 'which categories do families with kids buy?' "
    "or 'does product preference differ by age?'.",
    PreferenceArgs,
)
def category_preferences(ctx: ToolContext, args: PreferenceArgs) -> dict:
    households = ctx.repository.households
    table = ev.share_differences(ctx.repository.spend_share, households[args.by])
    flagged = table[table["direction"] != "none"]
    findings = [
        f"{table['category'].nunique()} kategori test edildi; {flagged['category'].nunique()} kategoride gruplar arasında anlamlı "
        f"ve belirgin bir bütçe payı farkı var (hane düzeyinde test, yanlış keşif düzeltmesiyle)."
        if len(flagged) else
        f"{table['category'].nunique()} kategori test edildi; hiçbirinde gruplar arasında belirgin bir bütçe payı farkı yok."
    ]
    for label in households[args.by].cat.categories:
        rows = table[table["group"] == label]
        if rows.empty:
            continue
        who = f"{segment_label(args.by, label)} (n={int(rows['group_size'].iloc[0])} hane)"
        over, under = standouts(rows, "over", args.top_n), standouts(rows, "under", 2)
        parts = [f"bütçesinden daha büyük pay ayırdığı kategoriler: {over}" if over else "",
                 f"daha küçük pay ayırdığı kategoriler: {under}" if under else ""]
        findings.append(f"{who}: " + "; ".join(p for p in parts if p) + "." if any(parts)
                        else f"{who}: belirgin biçimde öne çıkan bir kategori yok.")
    top = flagged.assign(size=flagged["gap_pp"].abs()).nlargest(3, "size")["category"].unique().tolist()
    groups = list(households[args.by].cat.categories)
    share = ctx.repository.spend_share
    chart = bar_chart(
        f"Bütçe payı (%) - {DIMENSION_NAME_TR[args.by]}", groups,
        {c: [share.loc[households[args.by] == g, c].mean() for g in groups] for c in top}, "%",
    ) if top else None
    return {"findings": findings, "chart": chart}


class CategoryProfileArgs(ToolArgs):
    category: str
    by: Dimension = "age"


@registry.register(
    "category_buyer_profile",
    "Which demographic groups buy ONE product category more: share of households that bought it and share of budget, per group, "
    "with significance tests. Use for 'who buys baby food?' or 'which income group buys wine?'.",
    CategoryProfileArgs,
)
def category_buyer_profile(ctx: ToolContext, args: CategoryProfileArgs) -> dict:
    category = ctx.resolver.category(args.category)
    households = ctx.repository.households
    bought = ctx.repository.bought[category]
    if bought.sum() < ctx.settings.min_group_size:
        raise ToolError(f"Too few households bought '{category}' to analyse reliably", minimum=ctx.settings.min_group_size)
    groups = households[args.by]
    subject = SUBJECT_TR[args.by]
    reach_evidence = ev.chi_square_independence(pd.crosstab(groups, bought), f"{subject} {category} alan hane oranında")
    frame = households.assign(budget_share=ctx.repository.spend_share[category])
    share_evidence = ev.group_difference(frame, "budget_share", args.by, f"{subject} {category} bütçe payında")
    reach = bought.groupby(groups, observed=True).mean() * 100
    shares = frame.groupby(args.by, observed=True)["budget_share"].mean()
    sizes = groups.value_counts()
    lines = "; ".join(
        f"{segment_label(args.by, g)} (n={sizes[g]} hane): alan hane oranı {percent(reach[g])}, bütçe payı {percent(shares[g])}"
        for g in groups.cat.categories
    )
    findings = [
        f"{category}: {int(bought.sum())} hane (%{bought.mean() * 100:.1f}) en az bir kez aldı.",
        lines + ".",
        reach_evidence.sentence,
        share_evidence.sentence,
    ]
    return {"findings": findings, "chart": bar_chart(
        f"{category}: alan hane oranı (%)", [segment_label(args.by, g) for g in groups.cat.categories],
        {category: reach.reindex(groups.cat.categories).tolist()}, "%")}


class CompareArgs(ToolArgs):
    metric: Metric = "total_spend"
    by: Dimension = "age"


@registry.register(
    "compare_groups",
    "Compare yearly spend, number of shopping trips, average basket, number of categories bought or private-label spend share "
    "across demographic groups with a significance test. Use for 'does spending rise with income?' or "
    "'do older households prefer private label?'.",
    CompareArgs,
)
def compare_groups(ctx: ToolContext, args: CompareArgs) -> dict:
    households = ctx.repository.households
    evidence = ev.group_difference(households, args.metric, args.by, f"{SUBJECT_TR[args.by]} {METRIC_TR[args.metric]} açısından")
    stats = households.groupby(args.by, observed=True)[args.metric].agg(["size", "median", "mean"])
    ranking = "; ".join(
        f"{segment_label(args.by, g)} medyan {fmt(r['median'])}, ortalama {fmt(r['mean'])} (n={int(r['size'])} hane)"
        for g, r in stats.iterrows()
    )
    return {"findings": [evidence.sentence, f"{METRIC_TR[args.metric].capitalize()}, gruplara göre: {ranking}."],
            "chart": bar_chart(f"{METRIC_TR[args.metric].capitalize()} (medyan) - {DIMENSION_NAME_TR[args.by]}",
                               [segment_label(args.by, g) for g in stats.index], {"medyan": stats["median"].tolist()}, "")}


class SegmentArgs(ToolArgs):
    dimension: Dimension
    value: str


@registry.register(
    "segment_profile",
    "Profile of ONE demographic segment (for example age 65+, income 125K+, households with kids): spending, basket, private-label "
    "share compared with all other households, and the categories it spends more on. Use for 'what is the profile of households "
    "over 65?'.",
    SegmentArgs,
)
def segment_profile(ctx: ToolContext, args: SegmentArgs) -> dict:
    households = ctx.repository.households
    value = ctx.resolver.segment(args.dimension, args.value)
    label = segment_label(args.dimension, value)
    inside = households[args.dimension] == value
    _segment_size_check(ctx, int(inside.sum()), label)
    side = pd.Series(pd.Categorical(inside.map({True: "inside", False: "rest"}), categories=["inside", "rest"]), index=households.index)
    findings = [f"{label}: {int(inside.sum())} hane, tüm hanelerin oranı {percent(inside.mean() * 100)}."]
    for metric in SEGMENT_METRICS:
        evidence = ev.group_difference(households.assign(side=side), metric, "side", f"{label} ile diğer haneler arasında {METRIC_TR[metric]} açısından")
        medians = households.groupby(side, observed=True)[metric].median()
        higher = medians["inside"] > medians["rest"]
        direction = "" if evidence.verdict == ev.NO_DIFFERENCE else f" {label} medyanı diğer hanelerden {'daha yüksek' if higher else 'daha düşük'}."
        findings.append(f"{METRIC_TR[metric].capitalize()}: {label} için medyan {fmt(medians['inside'])}, diğer haneler için {fmt(medians['rest'])}. {evidence.sentence}{direction}")
    table = ev.share_differences(ctx.repository.spend_share, side, ["inside"])
    over, under = standouts(table, "over", 4), standouts(table, "under", 2)
    findings.append(
        f"Bütçesinden daha büyük pay ayırdığı kategoriler: {over}." if over else "Diğer hanelere göre belirgin biçimde daha büyük pay ayırdığı kategori yok."
    )
    if under:
        findings.append(f"Daha küçük pay ayırdığı kategoriler: {under}.")
    return {"findings": findings}


class DriversArgs(ToolArgs):
    metric: Metric = "total_spend"


@registry.register(
    "demographic_drivers",
    "Which demographic features (age, income, household size, household type, kids) are most associated with yearly spend, "
    "basket size, categories bought or private-label share, ranked by effect size. Use for 'what affects spending most?'.",
    DriversArgs,
)
def demographic_drivers(ctx: ToolContext, args: DriversArgs) -> dict:
    households = ctx.repository.households
    results = [
        (dimension, ev.group_difference(households, args.metric, dimension, f"{SUBJECT_TR[dimension]} {METRIC_TR[args.metric]} açısından"))
        for dimension in ORDER
    ]
    results.sort(key=lambda item: abs(item[1].effect_size), reverse=True)
    return {"findings": [
        "Etki büyüklüğüne göre sıralama: " + ", ".join(
            f"{DIMENSION_NAME_TR[d]} ({ev.LABELS_TR[e.label]})" for d, e in results) + ".",
        *(e.sentence for _, e in results),
    ]}