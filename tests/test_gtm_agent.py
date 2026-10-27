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
    def test_application_tool_loop(self):
        requests = []
        rep = agent.REP_IDS[0]
        prospect = {"name": "Test Prospect", "email": "prospect@example.com"}
        responses = iter([
            AIMessage(content="", tool_calls=[{
                "name": "get_current_rep", "args": {}, "id": "rep",
            }]),
            AIMessage(content="", tool_calls=[{
                "name": "send_prospect_email",
                "args": {"prospect": prospect, "subject": "Hello", "body": "Welcome"},
                "id": "email",
            }]),
            AIMessage(content="Email sent."),
        ])

        def generate(model, messages, **kwargs):
            requests.append((messages, kwargs["tools"]))
            return ChatResult(generations=[ChatGeneration(message=next(responses))])

        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}), \
                patch.object(ChatOpenAI, "_generate", generate):
            result = agent.run_agent(
                "Email the prospect a welcome message.", user_id=rep["rep_id"],
                environment="test", thread_id="tool-loop",
            )

        expected_tools = {
            "lookup_offering", "build_prospect_profile", "get_prospect",
            "send_prospect_email", "score_prospect", "update_prospect_info",
            "get_current_rep",
        }
        self.assertEqual(len(requests), 3)
        for messages, tools in requests:
            self.assertEqual(messages[0].content, agent.SYSTEM_PROMPT)
            self.assertEqual(len(tools), 7)
            self.assertEqual({tool["function"]["name"] for tool in tools}, expected_tools)
            for tool in tools:
                self.assertNotIn("runtime", tool["function"]["parameters"]["properties"])
        tool_results = {
            message.name: json.loads(message.content)
            for message in result["messages"] if isinstance(message, ToolMessage)
        }
        self.assertEqual(set(tool_results), {"get_current_rep", "send_prospect_email"})
        self.assertEqual(tool_results["get_current_rep"]["rep"], rep)
        self.assertEqual(tool_results["send_prospect_email"]["status"], "sent")
        self.assertEqual(tool_results["send_prospect_email"]["from"], rep["email"])
        self.assertEqual(tool_results["send_prospect_email"]["to"], prospect["email"])
        self.assertEqual(result["reply"], "Email sent.")
        self.assertEqual(result["reply"], result["messages"][-1].content)

    def test_run_agent_tracing_configuration(self):
        messages = [AIMessage(content="Final reply")]
        with patch.object(agent.gtm_agent, "invoke", return_value={"messages": messages}) as invoke:
            result = agent.run_agent(
                "Email a prospect", user_id="rep-test", environment="staging",
                thread_id="thread-test",
            )
        run_id = uuid.UUID(result["run_id"])
        invoke.assert_called_once_with(
            {"messages": [{"role": "user", "content": "Email a prospect"}]},
            config={
                "run_name": "GTM Assistant",
                "run_id": run_id,
                "metadata": {
                    "thread_id": "thread-test", "user_id": "rep-test",
                    "environment": "staging", "request_intent": "send_email",
                },
            },
        )
        self.assertEqual(result, {"reply": "Final reply", "messages": messages, "run_id": str(run_id)})


if __name__ == "__main__":
    unittest.main()
