"""Deterministic prospect scoring tests."""

import importlib
import json
import unittest
from unittest.mock import patch

with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ProspectScoringTests(unittest.TestCase):
    def setUp(self):
        self.offering = {
            "description": "Security platform",
            "required_tech_stack": ["AWS", "Kafka", "Snowflake", "Okta"],
            "min_annual_revenue": 100,
        }
        self.profile = {"annual_revenue": 100, "tech_stack": ["AWS", "Kafka", "Snowflake"]}
        self.llm_patch = patch.object(agent, "_scoring_llm")
        self.llm = self.llm_patch.start()
        self.addCleanup(self.llm_patch.stop)
        self.llm.invoke.return_value = agent.ProspectScore(
            score=12,
            max_score=50,
            justification="Missing Okta.",
            rubric_breakdown=agent.RubricBreakdown(revenue_fit=4, tech_stack_match=9),
        )

    def score(self):
        return agent.score_prospect.invoke({"prospect_profile": self.profile, "offering": self.offering})

    def test_repeated_scores_override_model_values_and_preserve_justification(self):
        first = self.score()
        self.llm.invoke.return_value.score = 99
        self.llm.invoke.return_value.rubric_breakdown.revenue_fit = 80
        second = self.score()
        self.assertEqual(first, second)
        self.assertEqual(first["score"], 87.5)
        self.assertEqual(first["max_score"], 100)
        self.assertEqual(first["rubric_breakdown"], {
            "revenue_fit": 100.0, "tech_stack_match": 75.0, "component_max": 100,
        })
        self.assertEqual(first["justification"], "Missing Okta.")
        content = self.llm.invoke.call_args.args[0][1]["content"]
        computed = json.loads(content.split("Computed scoring:\n")[1])
        self.assertEqual(computed["score"], first["score"])
        self.assertEqual(computed["rubric_breakdown"], first["rubric_breakdown"])
        self.assertEqual(computed["present_technologies"], ["AWS", "Kafka", "Snowflake"])
        self.assertEqual(computed["missing_technologies"], ["Okta"])

    def test_technology_fit(self):
        for stack, expected in [([], 0.0), (["AWS"], 25.0),
                                (self.offering["required_tech_stack"], 100.0)]:
            with self.subTest(stack=stack):
                self.profile["tech_stack"] = stack
                self.assertEqual(self.score()["rubric_breakdown"]["tech_stack_match"], expected)
        self.offering["required_tech_stack"] = []
        self.assertEqual(self.score()["rubric_breakdown"]["tech_stack_match"], 100.0)

    def test_revenue_fit(self):
        for revenue, expected in [(100, 100.0), (200, 100.0), (25, 25.0),
                                  (33.333, 33.3), (None, 0.0)]:
            with self.subTest(revenue=revenue):
                self.profile["annual_revenue"] = revenue
                result = self.score()
                self.assertEqual(result["rubric_breakdown"]["revenue_fit"], expected)
                self.assertEqual(result["score"], round(0.5 * expected + 37.5, 1))
        del self.profile["annual_revenue"]
        self.assertEqual(self.score()["rubric_breakdown"]["revenue_fit"], 0.0)

    def test_zero_and_missing_revenue_minimum(self):
        self.offering["min_annual_revenue"] = 0
        self.assertEqual(self.score()["rubric_breakdown"]["revenue_fit"], 100.0)
        for minimum in [None, "absent"]:
            if minimum is None:
                self.offering["min_annual_revenue"] = None
            else:
                del self.offering["min_annual_revenue"]
            self.assertEqual(self.score(), {"score": None, "error": "Cannot score without a valid offering."})

    def test_segment_is_not_scored(self):
        for segment in [None, "Enterprise"]:
            with self.subTest(segment=segment):
                if segment is not None:
                    self.profile["segment"] = segment
                self.assertNotIn("segment_fit", self.score()["rubric_breakdown"])

    def test_scoring_uses_saved_technology_stack(self):
        self.profile["prospect_id"] = "LEAD-test"
        self.profile["tech_stack"] = self.offering["required_tech_stack"]
        with patch.object(agent.data_service, "fetch_tech_stack", return_value=["AWS"]):
            self.assertEqual(self.score()["rubric_breakdown"]["tech_stack_match"], 25.0)
