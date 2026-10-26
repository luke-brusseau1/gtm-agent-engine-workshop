import copy
import importlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}), \
        patch("langchain_openai.ChatOpenAI"), \
        patch("deepagents.create_deep_agent"):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ProspectDataTests(unittest.TestCase):
    def setUp(self):
        self.tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)
        self.pid = "TEST-LEAD"
        self.billing = {
            "tax_id": "test-tax-id",
            "date_of_birth": "test-birth-date",
            "card_on_file": "test-full-card-number",
            "credit_check_ref": "test-credit-reference",
        }
        self.safe = {
            "prospect_id": self.pid,
            "name": "Test Contact",
            "email": "contact@example.com",
            "annual_revenue": 70000000,
            "enrichment_source": "Test source",
            "engagement_history": [{"channel": "Webinar"}],
            "account_details": [{"industry": "Software"}],
            "tech_stack": ["Snowflake", "AWS"],
        }
        self.record = {
            **copy.deepcopy(self.safe),
            "prospect_id": "untrusted-source-id",
            "billing_qualification": self.billing,
            "disqualified": False,
            "unknown_private_field": "private value",
        }
        records = patch.dict(agent.data_service.PROSPECTS, {self.pid: self.record})
        records.start()
        self.addCleanup(records.stop)
        profiles = patch.dict(agent.data_service._PROFILES, {}, clear=True)
        profiles.start()
        self.addCleanup(profiles.stop)

    def test_contact_projection_and_email(self):
        result = agent.get_prospect.invoke({"prospect_id": self.pid})
        contact = {k: self.safe[k] for k in ("prospect_id", "name", "email")}
        self.assertEqual(result, {"prospect": contact, "found": True})
        sent = agent.send_prospect_email.func(
            contact, "Hello", "Message", SimpleNamespace(config={}),
            from_rep={"name": "Rep", "email": "rep@example.com"},
        )
        self.assertEqual(sent["status"], "sent")
        self.assertEqual(sent["to"], contact["email"])
        self.assertEqual(sent["to_name"], contact["name"])
        self.assertEqual(sent["from"], "rep@example.com")
        self.assertEqual(sent["body"], "Message")

    def test_new_profile_is_projected_and_cached(self):
        result = agent.build_prospect_profile.invoke({"prospect_id": self.pid})
        self.assertEqual(result, {"prospect_profile": self.safe, "found": True})
        self.assertEqual(agent.data_service._PROFILES[self.pid], self.safe)
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": self.pid}), result)

    def test_existing_profile_is_projected_and_replaced(self):
        agent.data_service._PROFILES[self.pid] = copy.deepcopy(self.record)
        result = agent.build_prospect_profile.invoke({"prospect_id": self.pid})
        self.assertEqual(result, {"prospect_profile": self.safe, "found": True})
        self.assertEqual(agent.data_service._PROFILES[self.pid], self.safe)

    def test_profile_uses_enrichment_fetch_results(self):
        with patch.object(agent.data_service, "fetch_engagement_history", return_value=[]), \
                patch.object(agent.data_service, "fetch_account_details", return_value=[]), \
                patch.object(agent.data_service, "fetch_tech_stack", return_value=["New tech"]):
            result = agent.build_prospect_profile.invoke({"prospect_id": self.pid})
        expected = {**self.safe, "engagement_history": [], "account_details": [],
                    "tech_stack": ["New tech"]}
        self.assertEqual(result["prospect_profile"], expected)
        self.assertEqual(agent.data_service._PROFILES[self.pid], expected)

    def test_missing_prospect(self):
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": "missing"}),
                         {"prospect": None, "found": False})
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}),
                         {"prospect_profile": None, "found": False})

    def test_scoring_filters_billing_and_uses_authoritative_stack(self):
        offering = {"required_tech_stack": ["AWS"], "min_annual_revenue": 1000,
                    "description": "Test offering"}
        for with_id in (True, False):
            with self.subTest(with_id=with_id):
                profile = {**copy.deepcopy(self.safe), "billing_qualification": self.billing,
                           "segment": "Enterprise", "tech_stack": ["Provided tech"]}
                if not with_id:
                    del profile["prospect_id"]
                original = copy.deepcopy(profile)
                expected = {k: v for k, v in profile.items() if k != "billing_qualification"}
                if with_id:
                    expected["tech_stack"] = self.safe["tech_stack"]
                score = {"score": 80, "justification": "Good fit"}
                with patch.object(agent, "_scoring_llm") as llm:
                    llm.invoke.return_value.model_dump.return_value = score
                    result = agent.score_prospect.invoke(
                        {"prospect_profile": profile, "offering": offering})
                self.assertEqual(result, score)
                messages = llm.invoke.call_args.args[0]
                content = messages[1]["content"]
                serialized = content.split("\n\nProspect profile:\n", 1)[1]
                self.assertEqual(json.loads(serialized), expected)
                for sensitive in ("billing_qualification", *self.billing, *self.billing.values()):
                    self.assertNotIn(sensitive, content)
                self.assertEqual(profile, original)

    def test_invalid_offering_does_not_score(self):
        with patch.object(agent, "_scoring_llm") as llm:
            for offering in (None, {}, {"description": "Incomplete"}):
                result = agent.score_prospect.invoke(
                    {"prospect_profile": self.safe, "offering": offering})
                self.assertIsNone(result["score"])
            llm.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
