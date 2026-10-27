import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

LIVE_MODEL_AVAILABLE = bool(os.environ.get("OPENAI_API_KEY"))

with patch.dict(os.environ, {"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "test-key"),
                             "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent as agent


class ProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}})
        self.prospect_id = next(
            pid for pid, record in agent.data_service.PROSPECTS.items()
            if record["disqualified"]
        )
        self.prospect = agent.get_prospect.invoke({"prospect_id": self.prospect_id})["prospect"]

    def send(self, prospect=None):
        return agent.send_prospect_email.func(
            self.prospect if prospect is None else prospect,
            "Demo", "Let's schedule a demo.", self.runtime,
        )

    def test_disqualified_source_blocks_even_if_input_claims_eligible(self):
        with patch.object(agent.uuid, "uuid4") as message_id:
            result = self.send({**self.prospect, "disqualified": False})
        self.assertEqual(result, {
            "status": "blocked",
            "error": "Prospect is disqualified; email not sent.",
            "to": self.prospect["email"],
        })
        message_id.assert_not_called()
        self.assertNotIn("message_id", result)

    def test_eligible_prospect_keeps_existing_send_behavior(self):
        pid = next(pid for pid, rec in agent.data_service.PROSPECTS.items()
                   if not rec["disqualified"])
        prospect = agent.get_prospect.invoke({"prospect_id": pid})["prospect"]
        result = self.send(prospect)
        self.assertEqual(result["status"], "sent")
        self.assertTrue(result["message_id"].startswith("msg-"))
        self.assertEqual(result["to"], prospect["email"])
        self.assertEqual(result["from"], agent.data_service.get_rep("rep_amills")["email"])
        self.assertEqual(result["subject"], "Demo")
        self.assertEqual(result["body"], "Let's schedule a demo.")

    def test_missing_source_or_identifier_blocks_send(self):
        for prospect in ({**self.prospect, "prospect_id": "unknown"},
                         {k: v for k, v in self.prospect.items() if k != "prospect_id"}):
            with self.subTest(prospect_id=prospect.get("prospect_id")):
                with patch.object(agent.uuid, "uuid4") as message_id:
                    result = self.send(prospect)
                self.assertEqual(result, {
                    "status": "blocked",
                    "error": "Prospect source record not found; email not sent.",
                    "to": prospect["email"],
                })
                message_id.assert_not_called()
                self.assertNotIn("message_id", result)

    def test_unavailable_source_blocks_send(self):
        with patch.object(agent.data_service, "get_prospect_record", side_effect=ConnectionError), \
                patch.object(agent.uuid, "uuid4") as message_id:
            result = self.send()
        self.assertEqual(result, {
            "status": "blocked",
            "error": "Cannot verify prospect eligibility; email not sent.",
            "to": self.prospect["email"],
        })
        message_id.assert_not_called()
        self.assertNotIn("message_id", result)

    @unittest.skipUnless(LIVE_MODEL_AVAILABLE, "Requires OpenAI model credentials")
    def test_agent_reports_actual_tool_block(self):
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        result = self.send()
        call = {"name": "send_prospect_email", "args": {
            "prospect": self.prospect, "subject": "Demo", "body": "Let's schedule a demo."
        }, "id": "email-call", "type": "tool_call"}
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
            response = agent.gtm_agent.invoke({"messages": [
                HumanMessage(content=f"Email {self.prospect_id} to schedule a demo."),
                AIMessage(content="", tool_calls=[call]),
                ToolMessage(content=json.dumps(result), tool_call_id="email-call"),
            ]}, config={"metadata": {"user_id": "rep_amills"}})
        reply = response["messages"][-1].content.lower()
        self.assertIn("blocked", reply)
        self.assertIn("disqualified", reply)
        self.assertRegex(reply, r"not (?:been )?sent|wasn't sent|did not send|no email (?:was |has been )?sent")
        self.assertNotRegex(reply, r"successfully sent|successfully delivered|i(?:'ve| have)? sent")


if __name__ == "__main__":
    unittest.main()
