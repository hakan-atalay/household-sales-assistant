import unittest

from tests.fakes import ScriptedClient
from tests.helpers import data_available, make_services


def events(services, question, history=None):
    return list(services.chat.stream(question, history))


@unittest.skipUnless(data_available(), "household tables missing")
class PipelineTests(unittest.TestCase):
    def test_faithful_answer_is_grounded_and_saved(self):
        services = make_services(ScriptedClient())
        result = events(services, "Yaşa göre ürün tercihi değişiyor mu?")
        kinds = [e["type"] for e in result]
        self.assertEqual(kinds[0], "status")
        self.assertIn("tools", kinds)
        self.assertIn("chart", kinds)
        done = result[-1]
        self.assertTrue(done["meta"]["grounded"])
        self.assertEqual(done["message_id"], 1)
        self.assertEqual(services.store.stats()["messages"], 1)

    def test_invented_numbers_trigger_one_rewrite(self):
        client = ScriptedClient(hallucinate=True)
        services = make_services(client)
        result = events(services, "Gelir arttıkça harcama artıyor mu?")
        kinds = [e["type"] for e in result]
        self.assertIn("replace", kinds)
        self.assertEqual(client.rewrites, 1)
        self.assertTrue(result[-1]["meta"]["grounded"])

    def test_age_question_gets_the_limit_explained(self):
        services = make_services(ScriptedClient())
        result = events(services, "Kadınlar ne alıyor?")
        tools = next(e for e in result if e["type"] == "tools")["calls"]
        self.assertEqual([c["name"] for c in tools], ["dataset_overview"])

    def test_greeting_runs_no_tool(self):
        result = events(make_services(ScriptedClient()), "Merhaba")
        self.assertEqual(next(e for e in result if e["type"] == "tools")["calls"], [])

    def test_unreadable_plan_becomes_an_error_event(self):
        result = events(make_services(ScriptedClient(plan_failure=True)), "Merhaba")
        self.assertEqual(result[-1]["type"], "error")

    def test_feedback_feeds_back_into_examples_and_exports(self):
        services = make_services(ScriptedClient())
        message_id = events(services, "Yaşa göre ürün tercihi değişiyor mu?")[-1]["message_id"]
        services.store.set_feedback(message_id, 1)
        self.assertEqual(len(services.store.liked_examples()), 1)
        second = events(services, "Gelir arttıkça harcama artıyor mu?")[-1]["message_id"]
        services.store.set_feedback(second, -1, "wrong_number", "Doğrusu: fark yok")
        self.assertIn("Doğrusu", services.store.export("dpo"))
        self.assertEqual(services.store.stats()["corrections"], 1)

    def test_model_that_keeps_inventing_gets_the_verified_findings_instead(self):
        client = ScriptedClient(stubborn=True)
        result = events(make_services(client), "Gelir arttıkça harcama artıyor mu?")
        done, replaced = result[-1], next(e for e in result if e["type"] == "replace")
        self.assertTrue(done["meta"]["fallback"])
        self.assertIn("Under 35K", replaced["text"])
        self.assertNotIn("777", replaced["text"])

    def test_model_only_sees_findings_not_raw_tables(self):
        seen = []

        class Spy(ScriptedClient):
            def stream(self, messages, *, temperature=0.2):
                seen.append(messages[-1]["content"])
                yield from super().stream(messages, temperature=temperature)

        events(make_services(Spy()), "Yaşa göre ürün tercihi değişiyor mu?")
        self.assertIn("[FINDINGS", seen[0])
        self.assertNotIn("q_value", seen[0])
        self.assertIn("öne çıkan bir kategori yok", seen[0])


if __name__ == "__main__":
    unittest.main()
