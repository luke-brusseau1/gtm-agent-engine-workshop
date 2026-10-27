import json
import os
import unittest
from unittest.mock import Mock, patch

from langsmith import tracing_context

from gtm_agent import data_service

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent


class UpdateProspectInfoTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(tracing_context(enabled=False))
        self.record = {
            "tech_stack": ["AWS"],
            "engagement_history": [],
            "account_details": [],
            "annual_revenue": 60000000,
        }
        self.enterContext(patch.object(data_service, "PROSPECTS", {"lead": self.record}))
        self.enterContext(patch.object(data_service, "_PROFILES", {}))

    def test_update_persists_and_is_immediately_readable(self):
        result = data_service.update_prospect_info("lead", "Kafka")

        self.assertEqual(result, {"updated": True, "found": True, "tech_stack": ["AWS", "Kafka"]})
        self.assertEqual(self.record["tech_stack"], ["AWS", "Kafka"])
        self.assertEqual(data_service.fetch_tech_stack("lead"), ["AWS", "Kafka"])

    def test_cached_profile_rebuild_and_score_use_persisted_stack(self):
        stale_profile = gtm_agent.build_prospect_profile.invoke({"prospect_id": "lead"})["prospect_profile"]
        result = gtm_agent.update_prospect_info.invoke({"prospect_id": "lead", "technology": "Kafka"})
        self.assertTrue(result["updated"])
        rebuilt = gtm_agent.build_prospect_profile.invoke({"prospect_id": "lead"})["prospect_profile"]
        self.assertEqual(rebuilt["tech_stack"], ["AWS", "Kafka"])

        offering = {
            "required_tech_stack": ["AWS", "Kafka"],
            "min_annual_revenue": 50000000,
            "description": "Data platform",
        }

        def score(messages):
            profile = json.loads(messages[1]["content"].split("\n\nProspect profile:\n", 1)[1])
            self.assertEqual(profile["tech_stack"], ["AWS", "Kafka"])
            missing = set(offering["required_tech_stack"]) - set(profile["tech_stack"])
            self.assertEqual(missing, set())
            return Mock(model_dump=lambda: {"score": 100})

        with patch.object(gtm_agent, "_scoring_llm") as scoring_llm:
            scoring_llm.invoke.side_effect = score
            for profile in (rebuilt, stale_profile):
                with self.subTest(profile=profile):
                    scored = gtm_agent.score_prospect.invoke({"prospect_profile": profile, "offering": offering})
                    self.assertEqual(scored["score"], 100)

    def test_unknown_prospect_does_not_report_success(self):
        result = data_service.update_prospect_info("unknown", "Kafka")

        self.assertEqual(result, {"updated": False, "found": False})
        self.assertNotIn("unknown", data_service.PROSPECTS)
        self.assertEqual(self.record["tech_stack"], ["AWS"])

    def test_existing_technology_does_not_duplicate_or_leave_stale_profile(self):
        data_service.save_profile_to_db("lead", {"prospect_id": "lead", "tech_stack": []})

        result = data_service.update_prospect_info("lead", "AWS")
        rebuilt = gtm_agent.build_prospect_profile.invoke({"prospect_id": "lead"})["prospect_profile"]

        self.assertEqual(result, {"updated": True, "found": True, "tech_stack": ["AWS"]})
        self.assertEqual(self.record["tech_stack"], ["AWS"])
        self.assertEqual(rebuilt["tech_stack"], ["AWS"])


if __name__ == "__main__":
    unittest.main()
