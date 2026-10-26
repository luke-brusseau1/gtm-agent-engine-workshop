import importlib
import json
import os
import unittest
import uuid
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI


with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class AgentTests(unittest.TestCase):
    def run_flow(self, request, steps):
        calls = []

        def generate(messages, **kwargs):
            calls.append((messages, kwargs))
            response = steps[len(calls) - 1]
            return ChatResult(generations=[ChatGeneration(message=response)])

        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}), patch.object(
            ChatOpenAI, "_generate", side_effect=generate
        ):
            result = agent.run_agent(
                request, user_id=agent.REP_IDS[0]["rep_id"],
                environment="test", thread_id="test-thread",
            )
        self.assertEqual(result["reply"], steps[-1].content)
        uuid.UUID(result["run_id"])
        expected_tools = {
            "lookup_offering", "build_prospect_profile", "get_prospect",
            "send_prospect_email", "score_prospect", "update_prospect_info",
            "get_current_rep",
        }
        for messages, kwargs in calls:
            self.assertEqual(messages[0].content, agent.SYSTEM_PROMPT)
            self.assertEqual(
                {tool["function"]["name"] for tool in kwargs["tools"]}, expected_tools
            )
            for tool in kwargs["tools"]:
                self.assertNotIn("runtime", tool["function"]["parameters"]["properties"])
        return [json.loads(message.content) for message in result["messages"]
                if isinstance(message, ToolMessage)]

    def test_email_flow(self):
        prospect_id = next(iter(agent.data_service.PROSPECTS))
        prospect = agent.data_service.get_prospect_record(prospect_id)
        outputs = self.run_flow("Email this prospect", [
            AIMessage(content="", tool_calls=[
                {"name": "get_current_rep", "args": {}, "id": "rep"},
                {"name": "get_prospect", "args": {"prospect_id": prospect_id}, "id": "prospect"},
            ]),
            AIMessage(content="", tool_calls=[{
                "name": "send_prospect_email", "id": "email",
                "args": {"prospect": prospect, "subject": "Follow up", "body": "Hello"},
            }]),
            AIMessage(content="Email sent."),
        ])
        self.assertEqual(outputs[0], {"rep": agent.REP_IDS[0], "found": True})
        self.assertEqual(outputs[1]["prospect"]["email"], prospect["email"])
        self.assertEqual(outputs[2]["status"], "sent")
        self.assertEqual(outputs[2]["from"], agent.REP_IDS[0]["email"])
        self.assertEqual(outputs[2]["to"], prospect["email"])
        self.assertEqual(outputs[2]["subject"], "Follow up")
        self.assertEqual(outputs[2]["body"], "Hello")

    def test_score_flow(self):
        prospect_id = next(iter(agent.data_service.PROSPECTS))
        offering_id = next(iter(agent.data_service.OFFERINGS))
        profile = {"prospect_id": prospect_id, **agent.data_service.get_prospect_record(prospect_id)}
        offering = agent.data_service.get_offering(offering_id)
        score = agent.ProspectScore(
            score=70, justification="Partial fit",
            rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=80, tech_stack_match=60, segment_fit=70,
            ),
        )
        with patch.object(agent, "_scoring_llm") as scoring:
            scoring.invoke.return_value = score
            outputs = self.run_flow("Score this prospect", [
                AIMessage(content="", tool_calls=[
                    {"name": "get_current_rep", "args": {}, "id": "rep"},
                    {"name": "lookup_offering", "args": {"offering_id": offering_id}, "id": "offering"},
                    {"name": "build_prospect_profile", "args": {"prospect_id": prospect_id}, "id": "profile"},
                ]),
                AIMessage(content="", tool_calls=[{
                    "name": "score_prospect", "id": "score",
                    "args": {"prospect_profile": profile, "offering": offering},
                }]),
                AIMessage(content="Score: 70/100."),
            ])
        self.assertTrue(outputs[1]["found"])
        self.assertEqual(outputs[2]["prospect_profile"]["prospect_id"], prospect_id)
        self.assertEqual(outputs[3], score.model_dump())
        agent.data_service._PROFILES.clear()


if __name__ == "__main__":
    unittest.main()
