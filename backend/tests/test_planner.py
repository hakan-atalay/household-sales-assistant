import json
import unittest

from sales_assistant.analytics.registry import registry
from sales_assistant.evaluation.runner import calls_match
from sales_assistant.llm.planner import MAX_CALLS, plan_schema
from tests.helpers import data_available, make_services


@unittest.skipUnless(data_available(), "household tables missing")
class PlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.services = make_services()

    def test_plan_schema_is_flat_enough_for_constrained_decoding(self):
        schema = plan_schema(registry)
        text = json.dumps(schema)
        self.assertNotIn("$ref", text)
        self.assertNotIn("$defs", text)
        alternatives = schema["properties"]["calls"]["items"]["anyOf"]
        self.assertEqual(sorted(a["properties"]["tool"]["enum"][0] for a in alternatives), sorted(registry.names()))
        self.assertEqual(schema["properties"]["calls"]["maxItems"], MAX_CALLS)

    def test_planner_prompt_mentions_every_tool_and_stays_small(self):
        prompt = self.services.pipeline.planner.system_prompt
        for name in registry.names():
            self.assertIn(name + "(", prompt)
        self.assertLess(len(prompt), 9000)

    def test_unreadable_model_output_is_rejected(self):
        from sales_assistant.llm.planner import PlanningError
        from tests.fakes import ScriptedClient
        services = make_services(ScriptedClient(plan_failure=True))
        with self.assertRaises(PlanningError):
            services.pipeline.planner.plan("Merhaba")

    def test_argument_matching_understands_aliases(self):
        expected = [{"tool": "segment_profile", "args": {"dimension": "age", "value": "65+"}}]
        old = [{"tool": "segment_profile", "args": {"dimension": "age", "value": "yaşlı"}}]
        young = [{"tool": "segment_profile", "args": {"dimension": "age", "value": "19-24"}}]
        self.assertTrue(calls_match(self.services, expected, old))
        self.assertFalse(calls_match(self.services, expected, young))
        self.assertFalse(calls_match(self.services, expected, [{"tool": "compare_groups", "args": {"metric": "total_spend"}}]))


if __name__ == "__main__":
    unittest.main()
