import importlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.profiles = patch.object(agent.data_service, "_PROFILES", {})
        self.profiles.start()
        self.addCleanup(self.profiles.stop)
        self.scorer = patch.object(agent, "_scoring_llm")
        self.scoring = self.scorer.start()
        self.addCleanup(self.scorer.stop)
        self.scoring.invoke.return_value.model_dump.return_value = {"score": 75}

    def score(self, prospect_id="LEAD-71001", offering_id="OFFER-10005"):
        return agent.score_prospect.invoke({
            "prospect_id": prospect_id, "offering_id": offering_id,
        })

    def scored_records(self):
        content = self.scoring.invoke.call_args.args[0][1]["content"]
        offering, profile = content.removeprefix("Offering:\n").split("\n\nProspect profile:\n")
        return json.loads(offering), json.loads(profile)

    def test_score_by_ids_loads_complete_records(self):
        self.assertEqual(self.score(), {"score": 75})
        offering, profile = self.scored_records()
        self.assertEqual(offering, agent.data_service.get_offering("OFFER-10005"))
        self.assertEqual(profile, {
            "prospect_id": "LEAD-71001",
            **agent.data_service.get_prospect_record("LEAD-71001"),
        })

    def test_missing_prospect_is_recoverable(self):
        result = self.score(prospect_id="missing")
        self.assertIsNone(result["score"])
        self.assertTrue(result["error"])
        self.scoring.invoke.assert_not_called()

    def test_missing_offering_is_recoverable(self):
        self.assertEqual(self.score(offering_id="missing"), {
            "score": None, "error": "Cannot score without a valid offering.",
        })
        self.scoring.invoke.assert_not_called()

    def test_incomplete_offering_is_recoverable(self):
        for field in ("required_tech_stack", "min_annual_revenue", "description"):
            with self.subTest(field=field):
                offering = dict(agent.data_service.get_offering("OFFER-10005"))
                offering.pop(field)
                self.assertFalse(agent._offering_has_required_fields(offering))
                with patch.object(agent.data_service, "get_offering", return_value=offering):
                    self.assertEqual(self.score(), {
                        "score": None, "error": "Cannot score without a valid offering.",
                    })
        self.scoring.invoke.assert_not_called()

    def test_cached_profile_and_current_stored_tech_stack(self):
        profile = agent.build_prospect_profile.invoke({"prospect_id": "LEAD-71001"})
        profile["prospect_profile"]["account_details"] = {"annual_revenue": 123456789}
        source = dict(agent.data_service.get_prospect_record("LEAD-71001"))
        source["tech_stack"] = ["persisted-technology"]
        with patch.dict(agent.data_service.PROSPECTS, {"LEAD-71001": source}):
            self.score()
        _, scored_profile = self.scored_records()
        self.assertEqual(scored_profile["account_details"], {"annual_revenue": 123456789})
        self.assertEqual(scored_profile["tech_stack"], ["persisted-technology"])

    def test_profile_build_and_score_build_the_same_profile(self):
        self.score()
        _, scored_profile = self.scored_records()
        built = agent.build_prospect_profile.invoke({"prospect_id": "LEAD-71001"})
        self.assertTrue(built["found"])
        self.assertEqual(built["prospect_profile"], scored_profile)


class RepContextTests(unittest.TestCase):
    def invoke_request(self, message, handler):
        with patch.object(agent, "gtm_agent") as graph:
            graph.invoke.side_effect = handler
            return agent.run_agent(message, user_id="rep_amills")

    def test_email_receives_signature_context_without_rep_tool(self):
        rep = agent.data_service.get_rep("rep_amills")
        sent = {}

        def email_handler(inputs, config):
            content = inputs["messages"][0]["content"]
            self.assertIn(f"Signed-in rep: {rep['name']} <{rep['email']}>", content)
            self.assertIn("Email LEAD-12853", content)
            prospect = agent.get_prospect.invoke({"prospect_id": "LEAD-12853"})["prospect"]
            sent.update(agent.send_prospect_email.func(
                prospect, "Demo", f"Let's meet next week.\n{rep['name']}",
                SimpleNamespace(config=config),
            ))
            return {"messages": [SimpleNamespace(content="Sent")]}

        with patch.object(agent.get_current_rep, "func", side_effect=AssertionError("Unneeded lookup")):
            result = self.invoke_request("Email LEAD-12853", email_handler)
        self.assertEqual(result["reply"], "Sent")
        self.assertEqual(sent["status"], "sent")
        self.assertEqual(sent["from"], rep["email"])
        self.assertEqual(sent["from_name"], rep["name"])
        self.assertTrue(sent["body"].endswith(rep["name"]))

    def test_explicit_rep_lookup_remains_available(self):
        def rep_handler(inputs, config):
            rep = agent.get_current_rep.func(SimpleNamespace(config=config))
            self.assertTrue(rep["found"])
            return {"messages": [SimpleNamespace(content=rep["rep"]["name"])]}

        result = self.invoke_request("Who is the signed-in rep?", rep_handler)
        self.assertEqual(result["reply"], "Ava Mills")

    def test_missing_or_failed_context_lookup_preserves_requests(self):
        for message in ("Email LEAD-12853", "Score LEAD-71001 against OFFER-10005"):
            for failure in (None, RuntimeError("Directory unavailable")):
                with self.subTest(message=message, failure=failure):
                    lookup = Mock(return_value=None)
                    if failure is not None:
                        lookup.side_effect = failure

                    def handler(inputs, config):
                        self.assertEqual(inputs["messages"][0]["content"], message)
                        self.assertEqual(config["metadata"]["user_id"], "rep_amills")
                        return {"messages": [SimpleNamespace(content="Completed")]}

                    with patch.object(agent.data_service, "get_rep", lookup):
                        result = self.invoke_request(message, handler)
                    self.assertEqual(result["reply"], "Completed")


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class AgentFlowTests(unittest.TestCase):
    def run_flow(self, tool_name, args, message):
        model = ScriptedModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": tool_name, "args": args, "id": "flow-call",
            }]),
            AIMessage(content="Completed"),
        ])
        graph = agent.create_deep_agent(
            model=model,
            tools=[agent.score_prospect, agent.send_prospect_email, agent.get_current_rep],
            system_prompt=agent.SYSTEM_PROMPT,
        )
        with patch.object(agent, "gtm_agent", graph):
            result = agent.run_agent(message, user_id="rep_amills")
        calls = [call["name"] for msg in result["messages"]
                 if isinstance(msg, AIMessage) for call in msg.tool_calls]
        self.assertEqual(calls, [tool_name])
        outputs = [json.loads(msg.content) for msg in result["messages"]
                   if isinstance(msg, ToolMessage)]
        self.assertEqual(len(outputs), 1)
        return outputs[0]

    def test_email_flow_can_send_without_rep_tool(self):
        rep = agent.data_service.get_rep("rep_amills")
        sent = self.run_flow("send_prospect_email", {
            "prospect": {"name": "Prospect", "email": "prospect@example.com"},
            "subject": "Demo", "body": f"Let's schedule a demo.\n{rep['name']}",
        }, "Email the prospect to schedule a demo")
        self.assertEqual(sent["status"], "sent")
        self.assertEqual(sent["from"], rep["email"])
        self.assertTrue(sent["body"].endswith(rep["name"]))

    def test_email_flow_survives_missing_or_failed_context_lookup(self):
        rep = agent.data_service.get_rep("rep_amills")
        for responses in ([None, None], [RuntimeError("Directory unavailable"), rep]):
            with self.subTest(responses=responses):
                with patch.object(agent.data_service, "get_rep", side_effect=responses):
                    sent = self.run_flow("send_prospect_email", {
                        "prospect": {"name": "Prospect", "email": "prospect@example.com"},
                        "subject": "Demo", "body": "Let's schedule a demo.",
                    }, "Email the prospect to schedule a demo")
                self.assertEqual(sent["status"], "sent")
                self.assertEqual(sent["from"], rep["email"] if responses[1] else None)

    def test_score_flow_passes_only_ids(self):
        with patch.object(agent, "_scoring_llm") as scoring:
            scoring.invoke.return_value.model_dump.return_value = {"score": 75}
            result = self.run_flow("score_prospect", {
                "prospect_id": "LEAD-71001", "offering_id": "OFFER-10005",
            }, "Score LEAD-71001 against OFFER-10005")
        self.assertEqual(result, {"score": 75})

    def test_explicit_rep_question_can_call_tool(self):
        result = self.run_flow("get_current_rep", {}, "Who is the signed-in rep?")
        self.assertTrue(result["found"])
        self.assertEqual(result["rep"], agent.data_service.get_rep("rep_amills"))


if __name__ == "__main__":
    unittest.main()
