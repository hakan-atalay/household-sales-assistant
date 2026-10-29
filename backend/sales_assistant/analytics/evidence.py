from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
MIN_EXPECTED = 5
MIN_GAP_PP = 0.1
MIN_RATIO = 1.3
LABEL_CUTS: dict[str, tuple[float, float, float]] = {
    "w": (0.1, 0.3, 0.5),
    "rank_biserial_r": (0.1, 0.3, 0.5),
    "spearman_rho": (0.1, 0.3, 0.5),
    "epsilon_squared": (0.01, 0.06, 0.14),
}
SHARE_COLUMNS = ["group", "category", "group_size", "share_pct", "rest_share_pct", "p_value", "q_value", "gap_pp", "direction"]
LABELS = ("negligible", "small", "medium", "large")
LABELS_TR = {"negligible": "ihmal edilebilir", "small": "küçük", "medium": "orta", "large": "büyük"}

NO_DIFFERENCE = "no_significant_difference"
NEGLIGIBLE = "significant_but_negligible_effect"
MEANINGFUL = "meaningful_difference"


def effect_label(effect_name: str, value: float) -> str:
    return LABELS[sum(abs(value) >= cut for cut in LABEL_CUTS[effect_name])]


def format_p(p: float) -> str:
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def verdict_for(p: float, label: str) -> str:
    if p >= ALPHA:
        return NO_DIFFERENCE
    return NEGLIGIBLE if label == "negligible" else MEANINGFUL


@dataclass(frozen=True)
class Evidence:
    test: str
    subject: str
    p_value: float
    effect_name: str
    effect_size: float

    @property
    def label(self) -> str:
        return effect_label(self.effect_name, self.effect_size)

    @property
    def verdict(self) -> str:
        return verdict_for(self.p_value, self.label)

    @property
    def sentence(self) -> str:
        p_text = format_p(self.p_value)
        effect_text = f"etki büyüklüğü {self.effect_name}={self.effect_size:.3f}, {LABELS_TR[self.label]}"
        if self.verdict == NO_DIFFERENCE:
            return f"{self.subject} istatistiksel olarak anlamlı bir fark YOK ({p_text}). Görülen sayısal ayrımlar tesadüf olabilir."
        if self.verdict == NEGLIGIBLE:
            return f"{self.subject} fark istatistiksel olarak tespit edilebiliyor ({p_text}) ama etkisi çok küçük ({effect_text})."
        return f"{self.subject} anlamlı ve etkisi {LABELS_TR[self.label]} bir fark var ({p_text}, {effect_text})."


def chi_square_independence(table: pd.DataFrame, subject: str) -> Evidence:
    table = table.loc[table.sum(axis=1) > 0, table.sum(axis=0) > 0]
    observed = table.to_numpy(dtype=float)
    chi2, p, _, _ = stats.chi2_contingency(observed, correction=False)
    n = observed.sum()
    rows, cols = observed.shape
    w = math.sqrt(max(0.0, chi2 / n - (rows - 1) * (cols - 1) / (n - 1)))
    return Evidence("chi-square independence", subject, float(p), "w", w)


def group_difference(frame: pd.DataFrame, value: str, group: str, subject: str) -> Evidence:
    clean = frame[[value, group]].dropna()
    samples = [g[value].to_numpy() for _, g in clean.groupby(group, observed=True) if len(g)]
    n = sum(len(s) for s in samples)
    if len(samples) == 2:
        first, second = samples
        u, p = stats.mannwhitneyu(first, second, alternative="two-sided")
        r = 1 - 2 * u / (len(first) * len(second))
        return Evidence("Mann-Whitney U", subject, float(p), "rank_biserial_r", float(-r))
    h, p = stats.kruskal(*samples)
    return Evidence("Kruskal-Wallis", subject, float(p), "epsilon_squared", float(h / (n - 1)))


def share_differences(share: pd.DataFrame, groups: pd.Series, labels: list[str] | None = None) -> pd.DataFrame:
    values = share.to_numpy()
    buyer_rate = (values > 0).mean(axis=0)
    frames = []
    for label in labels if labels is not None else list(groups.cat.categories):
        inside = (groups == label).to_numpy()
        size = int(inside.sum())
        expected = size * buyer_rate
        eligible = expected >= MIN_EXPECTED
        if size < 2 or not eligible.any():
            continue
        x, y = values[inside][:, eligible], values[~inside][:, eligible]
        frames.append(pd.DataFrame({
            "group": label,
            "category": share.columns[eligible],
            "group_size": size,
            "share_pct": x.mean(axis=0),
            "rest_share_pct": y.mean(axis=0),
            "p_value": stats.mannwhitneyu(x, y, axis=0, method="asymptotic").pvalue,
        }))
    if not frames:
        return pd.DataFrame(columns=SHARE_COLUMNS)
    table = pd.concat(frames, ignore_index=True)
    table["q_value"] = stats.false_discovery_control(table["p_value"].to_numpy(), method="bh")
    table["gap_pp"] = table["share_pct"] - table["rest_share_pct"]
    ratio = table["share_pct"] / table["rest_share_pct"]
    significant = (table["q_value"] < ALPHA) & (table["gap_pp"].abs() >= MIN_GAP_PP)
    table["direction"] = np.select(
        [significant & (ratio >= MIN_RATIO), significant & (ratio <= 1 / MIN_RATIO)], ["over", "under"], default="none"
    )
    return table
