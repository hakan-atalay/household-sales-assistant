import unittest

import numpy as np
import pandas as pd

from sales_assistant.analytics import evidence as ev


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def test_independent_data_is_not_flagged_often(self):
        hits = 0
        for _ in range(300):
            groups = self.rng.choice(list("ABCD"), 4000)
            items = self.rng.choice(list("xyz"), 4000, p=[0.5, 0.3, 0.2])
            hits += ev.chi_square_independence(pd.crosstab(groups, items), "t").p_value < 0.05
        self.assertLess(hits / 300, 0.09)

    def test_planted_association_is_meaningful(self):
        groups = self.rng.choice(list("ABCD"), 20000)
        items = np.where(groups == "A", self.rng.choice(list("xyz"), 20000, p=[0.2, 0.3, 0.5]),
                         self.rng.choice(list("xyz"), 20000, p=[0.5, 0.3, 0.2]))
        result = ev.chi_square_independence(pd.crosstab(groups, items), "t")
        self.assertEqual(result.verdict, ev.MEANINGFUL)
        self.assertEqual(result.label, "medium")

    def test_large_sample_tiny_effect_is_negligible(self):
        groups = self.rng.choice(["A", "B"], 400000)
        items = np.where(groups == "A", self.rng.random(400000) < 0.505, self.rng.random(400000) < 0.495)
        result = ev.chi_square_independence(pd.crosstab(groups, items), "t")
        self.assertLess(result.p_value, 0.05)
        self.assertEqual(result.verdict, ev.NEGLIGIBLE)

    def test_group_difference_uses_two_group_test(self):
        frame = pd.DataFrame({"v": self.rng.lognormal(4, 1, 5000), "g": self.rng.choice(["a", "b"], 5000)})
        self.assertEqual(ev.group_difference(frame, "v", "g", "t").test, "Mann-Whitney U")
        frame["g"] = self.rng.choice(["a", "b", "c"], 5000)
        self.assertEqual(ev.group_difference(frame, "v", "g", "t").test, "Kruskal-Wallis")

    def test_share_differences_flags_only_the_planted_cell(self):
        n = 600
        groups = pd.Series(pd.Categorical(self.rng.choice(["A", "B", "C"], n)))
        share = pd.DataFrame(self.rng.gamma(2, 1, (n, 6)) * (self.rng.random((n, 6)) < 0.7), columns=list("uvwxyz"))
        share.loc[(groups == "A").to_numpy(), "z"] *= 3
        table = ev.share_differences(share, groups)
        flagged = table[table["direction"] != "none"]
        self.assertEqual(set(flagged["category"]), {"z"})
        self.assertIn(("A", "over"), set(zip(flagged["group"], flagged["direction"])))

    def test_sparse_categories_are_not_tested(self):
        n = 200
        groups = pd.Series(pd.Categorical(self.rng.choice(["A", "B"], n)))
        share = pd.DataFrame({"common": self.rng.random(n) + 1, "rare": np.where(np.arange(n) < 3, 5.0, 0.0)})
        self.assertEqual(set(ev.share_differences(share, groups)["category"]), {"common"})
        self.assertTrue(ev.share_differences(share[["rare"]], groups).empty)

    def test_sentence_wording_follows_verdict(self):
        text = ev.chi_square_independence(pd.crosstab(["a", "b"] * 500, ["x", "x", "y", "y"] * 250), "Gruplar arasında")
        self.assertIn("YOK", text.sentence)
        self.assertNotIn("p=<", text.sentence)


if __name__ == "__main__":
    unittest.main()
