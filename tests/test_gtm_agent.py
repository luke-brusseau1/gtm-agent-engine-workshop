import importlib
import json
import os
import unittest
import uuid
from unittest.mock import patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class ScriptedModel(BaseChatModel):
    responses: list = Field(default_factory=list)
    requests: list = Field(default_factory=list)
    bound_tools: list = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = list(tools)
        return self

    def with_structured_output(self, schema, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.model = ScriptedModel()
        self.env = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.env.start()
        self.addCleanup(self.env.stop)
        with patch("langchain_openai.ChatOpenAI", return_value=self.model):
            self.agent = importlib.import_module("gtm_agent.gtm_agent")
            importlib.reload(self.agent)

    def call(self, name, args):
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": name,
                    "args": args,
                    "id": str(uuid.uuid4()),
                }
            ],
        )

    def results(self, result):
        return {
            message.name: json.loads(message.content)
            for message in result["messages"]
            if isinstance(message, ToolMessage)
        }

    def test_lookup_and_rep_metadata(self):
        self.model.responses = [
            self.call("get_current_rep", {}),
            self.call("lookup_offering", {"offering_id": "OFFER-10001"}),
            AIMessage(content="Core Platform Suite"),
        ]
        result = self.agent.run_agent("Look up OFFER-10001", user_id="rep_amills")
        outputs = self.results(result)
        self.assertEqual(
            outputs["get_current_rep"],
            {
                "rep": self.agent.data_service.get_rep("rep_amills"),
                "found": True,
            },
        )
        self.assertEqual(
            outputs["lookup_offering"],
            {
                "offering": self.agent.data_service.get_offering("OFFER-10001"),
                "found": True,
            },
        )
        self.assertEqual(result["reply"], "Core Platform Suite")
        self.assertEqual(
            {tool.name for tool in self.model.bound_tools},
            {
                "lookup_offering",
                "build_prospect_profile",
                "get_prospect",
                "send_prospect_email",
                "score_prospect",
                "update_prospect_info",
                "get_current_rep",
            },
        )
        self.assertEqual(len(self.model.bound_tools), 7)
        for request in self.model.requests:
            self.assertEqual(
                [
                    message.content
                    for message in request
                    if isinstance(message, SystemMessage)
                ],
                [self.agent.SYSTEM_PROMPT],
            )

    def test_email_defaults_to_signed_in_rep(self):
        prospect_id = next(iter(self.agent.data_service.PROSPECTS))
        prospect = self.agent.data_service.get_prospect_record(prospect_id)
        self.model.responses = [
            self.call("get_current_rep", {}),
            self.call("get_prospect", {"prospect_id": prospect_id}),
            self.call(
                "send_prospect_email",
                {
                    "prospect": {"name": prospect["name"], "email": prospect["email"]},
                    "subject": "Follow up",
                    "body": "Let's talk.",
                },
            ),
            AIMessage(content="Email sent"),
        ]
        result = self.agent.run_agent("Email the prospect", user_id="rep_dweiss")
        outputs = self.results(result)
        rep = self.agent.data_service.get_rep("rep_dweiss")
        self.assertEqual(outputs["get_current_rep"]["rep"], rep)
        self.assertTrue(outputs["get_prospect"]["found"])
        sent = outputs["send_prospect_email"]
        self.assertEqual(sent["status"], "sent")
        self.assertTrue(sent["message_id"].startswith("msg-"))
        self.assertEqual(sent["to"], prospect["email"])
        self.assertEqual(sent["from"], rep["email"])
        self.assertEqual(sent["from_name"], rep["name"])
        self.assertEqual(sent["subject"], "Follow up")
        self.assertEqual(sent["body"], "Let's talk.")
        self.assertEqual(result["reply"], "Email sent")

    def test_run_agent_invocation_config(self):
        self.model.responses = [AIMessage(content="Done")]
        with patch.object(
            self.agent.gtm_agent, "invoke", wraps=self.agent.gtm_agent.invoke
        ) as invoke:
            result = self.agent.run_agent(
                "Look up OFFER-10001",
                user_id="rep_amills",
                environment="test",
                thread_id="thread-123",
            )
        config = invoke.call_args.kwargs["config"]
        self.assertEqual(config["run_name"], "GTM Assistant")
        self.assertIsInstance(config["run_id"], uuid.UUID)
        self.assertEqual(str(config["run_id"]), result["run_id"])
        self.assertEqual(
            config["metadata"],
            {
                "thread_id": "thread-123",
                "user_id": "rep_amills",
                "environment": "test",
                "request_intent": "lookup_offering",
            },
        )
        self.assertEqual(result["reply"], "Done")


if __name__ == "__main__":
    unittest.main()
