import json
import unittest
from importlib import import_module

from sales_assistant.analytics.registry import ToolContext, registry
from sales_assistant.data.resolver import EntityResolver
from sales_assistant.llm.prompts import PLANNER_EXAMPLES
from sales_assistant.settings import get_settings
from tests.helpers import data_available, shared_repository

VERDICTS = {"no_significant_difference", "significant_but_negligible_effect", "meaningful_difference"}


@unittest.skipUnless(data_available(), "household tables missing")
class ToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import_module("sales_assistant.analytics.tools")
        repository = shared_repository()
        cls.context = ToolContext(repository, EntityResolver(repository), get_settings())

    def run_tool(self, name, **args):
        return registry.execute(self.context, name, args)

    def test_all_six_tools_are_registered(self):
        self.assertEqual(len(registry.names()), 6)

    def test_every_planner_example_runs_and_returns_findings(self):
        for question, calls in PLANNER_EXAMPLES:
            for call in calls:
                result = registry.execute(self.context, call["tool"], call["args"])
                self.assertNotIn("error", result, f"{question} -> {call}")
                findings = result["findings"]
                self.assertTrue(findings and all(isinstance(f, str) and f for f in findings), call)
                self.assertLess(sum(len(f) for f in findings), 4500, call)
                json.dumps(result)

    def test_charts_follow_the_frontend_contract(self):
        chart = self.run_tool("category_preferences", by="income")["chart"]
        self.assertEqual(set(chart), {"type", "title", "unit", "x", "series"})
        self.assertTrue(all(len(s["values"]) == len(chart["x"]) for s in chart["series"]))

    def test_known_real_preferences(self):
        kids = " ".join(self.run_tool("category_preferences", by="kids")["findings"])
        self.assertIn("DIAPERS & DISPOSABLES", kids)
        income = " ".join(self.run_tool("category_preferences", by="income")["findings"])
        self.assertIn("IMPORTED WINE", income)

    def test_empty_standout_is_stated_explicitly(self):
        findings = self.run_tool("category_preferences", by="age")["findings"]
        middle = next(f for f in findings if f.startswith("45-54"))
        self.assertIn("öne çıkan bir kategori yok", middle)

    def test_no_difference_is_reported_as_such(self):
        findings = self.run_tool("compare_groups", metric="total_spend", by="kids")["findings"]
        self.assertIn("anlamlı bir fark YOK", findings[0])

    def test_income_raises_spending_but_effect_is_small(self):
        findings = self.run_tool("compare_groups", metric="total_spend", by="income")["findings"]
        self.assertIn("etkisi küçük", findings[0])

    def test_segment_profile_resolves_loose_values(self):
        findings = self.run_tool("segment_profile", dimension="kids", value="çocuklu")["findings"]
        self.assertTrue(findings[0].startswith("çocuklu haneler: 288 hane"))

    def test_errors_are_reported_not_raised(self):
        self.assertIn("did_you_mean", self.run_tool("category_buyer_profile", category="wine"))
        self.assertIn("error", self.run_tool("compare_groups", metric="gender", by="age"))
        self.assertIn("error", self.run_tool("segment_profile", dimension="age", value="200"))
        self.assertIn("error", registry.execute(self.context, "nope", {}))

    def test_rare_category_is_refused(self):
        rare = self.context.repository.bought.sum().idxmin()
        self.assertIn("error", self.run_tool("category_buyer_profile", category=rare))


if __name__ == "__main__":
    unittest.main()
