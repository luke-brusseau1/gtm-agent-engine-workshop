"""Tests for prospect data boundaries and tool flows."""

import copy
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import data_service
    from gtm_agent import gtm_agent as agent


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.cache = patch.object(data_service, "_PROFILES", {})
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.prospect_id = "LEAD-12853"
        self.record = copy.deepcopy(data_service.PROSPECTS[self.prospect_id])
        self.records = patch.dict(data_service.PROSPECTS, {self.prospect_id: self.record})
        self.records.start()
        self.addCleanup(self.records.stop)

    def assert_no_billing(self, value):
        serialized = json.dumps(value)
        self.assertNotIn("billing_qualification", serialized)
        for key, content in self.record["billing_qualification"].items():
            self.assertNotIn(key, serialized)
            self.assertNotIn(content, serialized)

    def test_contact_allowlist_and_email_flow(self):
        for disqualified in (False, True, None):
            with self.subTest(disqualified=disqualified):
                if disqualified is None:
                    self.record.pop("disqualified", None)
                else:
                    self.record["disqualified"] = disqualified
                result = agent.get_prospect.invoke({"prospect_id": self.prospect_id})
                self.assertEqual(result, {"found": True, "prospect": {
                    "prospect_id": self.prospect_id,
                    "name": self.record["name"],
                    "email": self.record["email"],
                    "disqualified": disqualified if disqualified is not None else False,
                }})
                self.assert_no_billing(result)
                arguments = {"prospect": result["prospect"], "subject": "Hello", "body": "Let's connect."}
                self.assert_no_billing(arguments)
                sent = agent.send_prospect_email.func(
                    **arguments, runtime=SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}})
                )
                self.assertEqual(sent["status"], "sent")
                self.assertEqual(sent["to"], self.record["email"])
                self.assertEqual(sent["from"], data_service.get_rep("rep_amills")["email"])
                self.assert_no_billing(sent)

    def test_new_profile_is_clean_and_preserves_scoring_fields(self):
        result = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        expected = {k: v for k, v in self.record.items() if k != "billing_qualification"}
        expected["prospect_id"] = self.prospect_id
        self.assertEqual(result, {"prospect_profile": expected, "found": True})
        self.assertEqual(data_service.get_profile_from_db(self.prospect_id)["prospect_profile"], expected)
        self.assert_no_billing(result)
        offering = agent.lookup_offering.invoke({"offering_id": "OFFER-10001"})["offering"]
        arguments = {"prospect_profile": result["prospect_profile"], "offering": offering}
        self.assert_no_billing(arguments)
        self.check_scoring(arguments)

    def test_cached_profile_is_cleaned_before_return(self):
        cached = {"prospect_id": self.prospect_id, **self.record}
        data_service.save_profile_to_db(self.prospect_id, cached)
        result = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertTrue(result["found"])
        self.assertEqual(result["prospect_profile"]["annual_revenue"], self.record["annual_revenue"])
        self.assert_no_billing(result)
        self.assert_no_billing(data_service.get_profile_from_db(self.prospect_id))

    def check_scoring(self, arguments):
        score = agent.ProspectScore(
            score=75, justification="Good fit", rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=90, tech_stack_match=60, segment_fit=75
            )
        )
        with patch.object(agent, "_scoring_llm") as llm:
            llm.invoke.return_value = score
            result = agent.score_prospect.invoke(arguments)
        self.assertEqual(result, score.model_dump())
        messages = llm.invoke.call_args.args[0]
        self.assert_no_billing(messages)
        profile = json.loads(messages[1]["content"].split("Prospect profile:\n", 1)[1])
        expected = {k: v for k, v in arguments["prospect_profile"].items() if k != "billing_qualification"}
        if "prospect_id" in expected:
            expected["tech_stack"] = self.record["tech_stack"]
        self.assertEqual(profile, expected)

    def test_scoring_strips_billing_even_for_direct_calls(self):
        offering = data_service.get_offering("OFFER-10001")
        for with_id in (True, False):
            with self.subTest(with_id=with_id):
                profile = {**self.record, "tech_stack": ["Stale technology"]}
                if with_id:
                    profile["prospect_id"] = self.prospect_id
                self.check_scoring({"prospect_profile": profile, "offering": offering})

    def test_missing_prospect_and_invalid_offering(self):
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": "missing"}),
                         {"prospect": None, "found": False})
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}),
                         {"prospect_profile": None, "found": False})
        result = agent.score_prospect.invoke({"prospect_profile": self.record})
        self.assertIsNone(result["score"])
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
