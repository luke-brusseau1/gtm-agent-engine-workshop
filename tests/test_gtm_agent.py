import importlib
import json
import unittest
from unittest.mock import Mock, patch

from langsmith import tracing_context


with patch("dotenv.load_dotenv"), patch.dict(
    "os.environ", {"LANGSMITH_TRACING": "false"}
), patch("langchain_openai.ChatOpenAI"), patch("deepagents.create_deep_agent"):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ProspectDataTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "name": "Example Prospect",
            "email": "prospect@example.com",
            "disqualified": False,
            "annual_revenue": 1000000,
            "tech_stack": ["Example technology"],
            "account_details": {"segment": "Enterprise"},
            "engagement_history": [],
            "billing_qualification": dict.fromkeys(
                ("tax_id", "date_of_birth", "card_on_file", "credit_check_ref")
            ),
            "extra_field": None,
        }
        self.addCleanup(patch.stopall)
        patch.object(agent.data_service, "PROSPECTS", {"test-prospect": self.record}).start()
        patch.object(agent.data_service, "_PROFILES", {}).start()
        self.tracing = tracing_context(enabled=False)
        self.tracing.__enter__()
        self.addCleanup(self.tracing.__exit__, None, None, None)

    def test_contact_contains_only_contact_fields(self):
        result = agent.get_prospect.invoke({"prospect_id": "test-prospect"})
        self.assertEqual(result, {
            "found": True,
            "prospect": {
                "prospect_id": "test-prospect",
                **{k: self.record[k] for k in ("name", "email", "disqualified")},
            },
        })

    def test_built_and_reused_profiles_exclude_billing(self):
        expected = {
            "prospect_id": "test-prospect",
            **{k: v for k, v in self.record.items() if k != "billing_qualification"},
        }
        for _ in range(2):
            result = agent.build_prospect_profile.invoke({"prospect_id": "test-prospect"})
            self.assertEqual(result, {"prospect_profile": expected, "found": True})
            self.assertEqual(agent.data_service._PROFILES["test-prospect"], expected)
        self.assertIn("billing_qualification", self.record)

    def test_existing_cached_profile_is_cleaned_and_saved(self):
        agent.data_service._PROFILES["test-prospect"] = dict(self.record)
        expected = {k: v for k, v in self.record.items() if k != "billing_qualification"}
        result = agent.build_prospect_profile.invoke({"prospect_id": "test-prospect"})
        self.assertEqual(result, {"prospect_profile": expected, "found": True})
        self.assertEqual(agent.data_service._PROFILES["test-prospect"], expected)

    def test_scorer_receives_only_scoring_inputs(self):
        offering = {
            "required_tech_stack": ["Example technology"],
            "min_annual_revenue": 100000,
            "description": "Example offering",
        }
        expected = {k: self.record[k] for k in (
            "name", "annual_revenue", "tech_stack", "account_details", "engagement_history"
        )}
        scorer = Mock()
        scorer.invoke.return_value.model_dump.return_value = {"score": 75}
        with patch.object(agent, "_scoring_llm", scorer):
            for with_id in (True, False):
                with self.subTest(with_id=with_id):
                    profile = dict(self.record)
                    if with_id:
                        profile["prospect_id"] = "test-prospect"
                        profile["tech_stack"] = ["Caller supplied technology"]
                    result = agent.score_prospect.invoke({
                        "prospect_profile": profile, "offering": offering,
                    })
                    self.assertEqual(result, {"score": 75})
                    messages = scorer.invoke.call_args.args[0]
                    serialized = messages[1]["content"].split("\n\nProspect profile:\n", 1)[1]
                    self.assertEqual(json.loads(serialized), {
                        **expected,
                        **({"prospect_id": "test-prospect"} if with_id else {}),
                    })

    def test_unknown_prospect_is_not_found(self):
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": "unknown"}), {
            "prospect": None, "found": False,
        })
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "unknown"}), {
            "prospect_profile": None, "found": False,
        })


if __name__ == "__main__":
    unittest.main()
