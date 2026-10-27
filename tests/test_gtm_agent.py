import importlib
import json
import os
import unittest
from unittest.mock import Mock, patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import Field

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ToolCallingModel(FakeMessagesListChatModel):
    seen_messages: list = Field(default_factory=list)
    tool_names: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.tool_names = [tool.name if hasattr(tool, "name") else tool["name"] for tool in tools]
        return self

    def _generate(self, messages, **kwargs):
        self.seen_messages.append(messages)
        return super()._generate(messages, **kwargs)


def tool_call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class ScoreProspectTests(unittest.TestCase):
    def setUp(self):
        self.tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)
        self.profiles = patch.dict(agent.data_service._PROFILES, {}, clear=True)
        self.profiles.start()
        self.addCleanup(self.profiles.stop)
        self.llm = patch.object(agent, "_scoring_llm")
        self.scoring = self.llm.start()
        self.addCleanup(self.llm.stop)
        self.expected = {
            "score": 80, "max_score": 100, "justification": "Good fit",
            "rubric_breakdown": {"revenue_fit": 90, "tech_stack_match": 70,
                                "segment_fit": 80, "component_max": 100},
        }
        self.scoring.invoke.return_value = agent.ProspectScore(**self.expected)

    def test_score_from_ids_builds_and_refreshes_profile(self):
        for cached in (False, True):
            with self.subTest(cached=cached):
                agent.data_service._PROFILES.clear()
                if cached:
                    agent.build_prospect_profile.invoke({"prospect_id": "LEAD-12853"})
                with patch.object(agent.data_service, "fetch_tech_stack", return_value=["Fresh tech"]):
                    result = agent.score_prospect.invoke({
                        "prospect_id": "LEAD-12853", "offering_id": "OFFER-10001",
                    })
                self.assertEqual(result, self.expected)
                content = self.scoring.invoke.call_args.args[0][1]["content"]
                offering_text, profile_text = content.removeprefix("Offering:\n").split("\n\nProspect profile:\n")
                self.assertEqual(json.loads(offering_text), agent.data_service.get_offering("OFFER-10001"))
                profile = json.loads(profile_text)
                self.assertEqual(profile["tech_stack"], ["Fresh tech"])
                self.assertEqual(profile["name"], agent.data_service.get_prospect_record("LEAD-12853")["name"])
                self.assertEqual(profile["prospect_id"], "LEAD-12853")
                self.assertIn("LEAD-12853", agent.data_service._PROFILES)

    def test_missing_prospect(self):
        self.assertEqual(agent.score_prospect.invoke({
            "prospect_id": "missing", "offering_id": "OFFER-10001",
        }), {"score": None, "error": "Cannot score without a valid prospect."})
        self.scoring.invoke.assert_not_called()

    def test_missing_or_invalid_offering(self):
        valid = agent.data_service.get_offering("OFFER-10001")
        offerings = [None] + [{**valid, field: value} for field, value in (
            ("required_tech_stack", []), ("min_annual_revenue", None), ("description", ""),
        )]
        for offering in offerings:
            with self.subTest(offering=offering):
                with patch.object(agent.data_service, "get_offering", return_value=offering):
                    result = agent.score_prospect.invoke({
                        "prospect_id": "LEAD-12853", "offering_id": "invalid",
                    })
                self.assertEqual(result, {"score": None, "error": "Cannot score without a valid offering."})
                self.scoring.invoke.assert_not_called()
                self.assertEqual(agent.data_service._PROFILES, {})


class RunAgentTests(unittest.TestCase):
    def run_with_model(self, request, responses, user_id="rep_amills"):
        model = ToolCallingModel(responses=responses)
        tools = [agent.lookup_offering, agent.build_prospect_profile, agent.get_prospect,
                 agent.send_prospect_email, agent.score_prospect, agent.update_prospect_info,
                 agent.get_current_rep]
        graph = agent.create_deep_agent(model=model, tools=tools, system_prompt=agent.SYSTEM_PROMPT)
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}), patch.object(agent, "gtm_agent", graph):
            result = agent.run_agent(request, user_id=user_id)
        return result, model

    def test_email_has_rep_identity_without_rep_tool_call(self):
        rep = agent.data_service.get_rep("rep_amills")
        contact = agent.data_service.get_prospect_record("LEAD-12853")
        result, model = self.run_with_model("Email LEAD-12853", [
            tool_call("get_prospect", {"prospect_id": "LEAD-12853"}, "lookup"),
            tool_call("send_prospect_email", {
                "prospect": {"name": contact["name"], "email": contact["email"]},
                "subject": "Hello", "body": "Hello,\n" + rep["name"],
            }, "email"),
            AIMessage(content="Sent"),
        ])
        initial_context = "\n".join(str(message.content) for message in model.seen_messages[0][:-1])
        self.assertIn(rep["name"], initial_context)
        self.assertIn(rep["email"], initial_context)
        outputs = [message for message in result["messages"] if isinstance(message, ToolMessage)]
        self.assertEqual([message.name for message in outputs], ["get_prospect", "send_prospect_email"])
        sent = json.loads(outputs[-1].content)
        self.assertEqual(sent["status"], "sent")
        self.assertEqual(sent["from"], rep["email"])
        self.assertEqual(sent["from_name"], rep["name"])
        self.assertIn("get_current_rep", model.tool_names)

    def test_explicit_rep_lookup_remains_available(self):
        result, _ = self.run_with_model("Who am I?", [
            tool_call("get_current_rep", {}, "rep"), AIMessage(content="Ava Mills"),
        ])
        output = next(message for message in result["messages"] if isinstance(message, ToolMessage))
        self.assertEqual(json.loads(output.content), {
            "rep": agent.data_service.get_rep("rep_amills"), "found": True,
        })

    def test_score_can_start_with_ids_without_rep_lookup(self):
        with patch.object(agent, "_scoring_llm") as scoring:
            scoring.invoke.return_value = Mock(model_dump=lambda: {"score": 80})
            result, _ = self.run_with_model("Score LEAD-12853 for OFFER-10001", [
                tool_call("score_prospect", {"prospect_id": "LEAD-12853", "offering_id": "OFFER-10001"}, "score"),
                AIMessage(content="Score: 80"),
            ])
        outputs = [message for message in result["messages"] if isinstance(message, ToolMessage)]
        self.assertEqual([message.name for message in outputs], ["score_prospect"])
        self.assertEqual(json.loads(outputs[0].content), {"score": 80})

    def test_unknown_rep_preserves_request(self):
        result, model = self.run_with_model("Hello", [AIMessage(content="Hello")], user_id="missing")
        self.assertEqual(result["reply"], "Hello")
        self.assertEqual(model.seen_messages[0][-1].content, "Hello")


if __name__ == "__main__":
    unittest.main()
