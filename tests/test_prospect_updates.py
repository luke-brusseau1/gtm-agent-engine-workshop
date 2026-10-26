import json
import unittest
from unittest.mock import patch

with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import data_service
    from gtm_agent import gtm_agent as agent


class ProspectUpdateTests(unittest.TestCase):
    def setUp(self):
        self.prospect_id = "LEAD-39002"
        self.original_stack = list(data_service.fetch_tech_stack(self.prospect_id))
        self.original_profiles = dict(data_service._PROFILES)
        data_service._PROFILES.clear()
        self.addCleanup(self.restore_records)
        self.tracing = patch.dict("os.environ", {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)

    def restore_records(self):
        data_service.PROSPECTS[self.prospect_id]["tech_stack"] = self.original_stack
        data_service._PROFILES.clear()
        data_service._PROFILES.update(self.original_profiles)

    def test_update_then_score_uses_persisted_stack(self):
        offering = agent.lookup_offering.invoke({"offering_id": "OFFER-10004"})["offering"]
        stale_profile = {"prospect_id": self.prospect_id, "tech_stack": self.original_stack}
        result = agent.update_prospect_info.invoke({
            "prospect_id": self.prospect_id, "technology": "Terraform",
        })
        self.assertEqual(result, {
            "prospect_id": self.prospect_id, "technology": "Terraform",
            "updated": True, "found": True,
            "tech_stack": self.original_stack + ["Terraform"],
        })
        self.assertEqual(data_service.fetch_tech_stack(self.prospect_id), result["tech_stack"])

        def score(messages):
            profile = json.loads(messages[1]["content"].split("Prospect profile:\n", 1)[1])
            missing = set(offering["required_tech_stack"]) - set(profile["tech_stack"])
            self.assertEqual(missing, set())
            return agent.ProspectScore(
                score=100, justification="Missing technologies: " + (", ".join(sorted(missing)) or "none"),
                rubric_breakdown=agent.RubricBreakdown(
                    revenue_fit=100, tech_stack_match=100, segment_fit=100,
                ),
            )

        with patch.object(agent, "_scoring_llm") as scoring_llm:
            scoring_llm.invoke.side_effect = score
            scored = agent.score_prospect.invoke({"prospect_profile": stale_profile, "offering": offering})
        self.assertEqual(scored["justification"], "Missing technologies: none")

    def test_cached_profile_reflects_update(self):
        agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        result = data_service.update_prospect_info(self.prospect_id, "Terraform")
        cached = data_service.get_profile_from_db(self.prospect_id)["prospect_profile"]
        rebuilt = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertEqual(cached["tech_stack"], result["tech_stack"])
        self.assertEqual(rebuilt["prospect_profile"]["tech_stack"], result["tech_stack"])

    def test_repeated_update_does_not_duplicate_technology(self):
        data_service.update_prospect_info(self.prospect_id, "Terraform")
        data_service.update_prospect_info(self.prospect_id, "Terraform")
        self.assertEqual(data_service.fetch_tech_stack(self.prospect_id).count("Terraform"), 1)

    def test_unknown_prospect_is_not_updated(self):
        self.assertEqual(data_service.update_prospect_info("UNKNOWN", "Terraform"), {
            "updated": False, "found": False,
        })


if __name__ == "__main__":
    unittest.main()
