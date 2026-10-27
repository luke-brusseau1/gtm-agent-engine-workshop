import importlib
import json
import os
import unittest
from copy import deepcopy
from unittest.mock import patch

from gtm_agent import data_service

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class UpdateProspectInfoTests(unittest.TestCase):
    def setUp(self):
        self.prospect_id = "LEAD-71001"
        self.technology = "Kafka"
        self.records = deepcopy(data_service.PROSPECTS)
        self.enterContext(patch.object(data_service, "PROSPECTS", self.records))
        self.enterContext(patch.object(data_service, "_PROFILES", {}))
        self.enterContext(patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}))
        self.assertNotIn(self.technology, self.records[self.prospect_id]["tech_stack"])

    def test_added_technology_is_persisted_and_in_profile(self):
        result = data_service.update_prospect_info(self.prospect_id, self.technology)

        self.assertTrue(result["updated"])
        self.assertTrue(result["found"])
        self.assertIn(self.technology, result["tech_stack"])
        self.assertEqual(result["tech_stack"], self.records[self.prospect_id]["tech_stack"])
        self.assertEqual(result["tech_stack"], data_service.fetch_tech_stack(self.prospect_id))
        profile = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertTrue(profile["found"])
        self.assertEqual(result["tech_stack"], profile["prospect_profile"]["tech_stack"])

    def test_cached_profile_reflects_added_technology(self):
        args = {"prospect_id": self.prospect_id}
        agent.build_prospect_profile.invoke(args)
        cached = agent.build_prospect_profile.invoke(args)
        self.assertNotIn(self.technology, cached["prospect_profile"]["tech_stack"])

        data_service.update_prospect_info(self.prospect_id, self.technology)

        rebuilt = agent.build_prospect_profile.invoke(args)
        self.assertIn(self.technology, rebuilt["prospect_profile"]["tech_stack"])
        self.assertIn(self.technology, data_service.fetch_tech_stack(self.prospect_id))

    def test_existing_technology_does_not_report_update_or_duplicate(self):
        original = list(self.records[self.prospect_id]["tech_stack"])
        result = data_service.update_prospect_info(self.prospect_id, original[0])

        self.assertEqual(result, {"updated": False, "found": True, "tech_stack": original})
        self.assertEqual(data_service.fetch_tech_stack(self.prospect_id), original)

    def test_unknown_prospect_preserves_not_found_response(self):
        result = data_service.update_prospect_info("unknown", self.technology)

        self.assertEqual(result, {"updated": False, "found": False})
        self.assertNotIn("unknown", self.records)

    def test_rescore_uses_persisted_stack_even_with_old_profile(self):
        profile = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        offering = data_service.get_offering("OFFER-10005")
        scored_profiles = []

        def score(messages):
            supplied = json.loads(messages[1]["content"].split("\n\nProspect profile:\n")[1])
            scored_profiles.append(supplied)
            missing = set(offering["required_tech_stack"]) - set(supplied["tech_stack"])
            return agent.ProspectScore(
                score=85 if missing else 100,
                justification="Missing: " + ", ".join(sorted(missing)),
                rubric_breakdown=agent.RubricBreakdown(
                    revenue_fit=100, tech_stack_match=85 if missing else 100, segment_fit=100
                ),
            )

        with patch.object(agent, "_scoring_llm") as scoring_llm:
            scoring_llm.invoke.side_effect = score
            args = {"prospect_profile": profile["prospect_profile"], "offering": offering}
            before = agent.score_prospect.invoke(args)
            update = agent.update_prospect_info.invoke(
                {"prospect_id": self.prospect_id, "technology": self.technology}
            )
            after = agent.score_prospect.invoke(args)

        self.assertTrue(update["updated"])
        self.assertNotIn(self.technology, scored_profiles[0]["tech_stack"])
        self.assertIn(self.technology, scored_profiles[1]["tech_stack"])
        self.assertIn(self.technology, before["justification"])
        self.assertNotIn(self.technology, after["justification"])
        self.assertGreater(after["score"], before["score"])
