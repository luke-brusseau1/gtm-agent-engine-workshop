import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

with patch.dict(os.environ, {"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent as agent


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}})
        self.prospect = {"prospect_id": "LEAD-123", "name": "Prospect", "email": "supplied@example.com"}

    def send(self, prospect=None):
        return agent.send_prospect_email.func(
            prospect=self.prospect if prospect is None else prospect,
            subject="Demo",
            body="Let's schedule a demo.",
            runtime=self.runtime,
        )

    def test_disqualified_source_blocks_even_when_supplied_record_says_eligible(self):
        with patch.object(agent.data_service, "get_prospect_record", return_value={"disqualified": True}), \
                patch.object(agent.uuid, "uuid4") as message_id:
            result = self.send({**self.prospect, "disqualified": False})
        self.assertEqual(result["status"], "blocked")
        self.assertIn("disqualified", result["error"])
        self.assertIn("not sent", result["error"])
        self.assertNotIn("message_id", result)
        message_id.assert_not_called()

    def test_eligible_source_sends_to_source_email(self):
        with patch.object(agent.data_service, "get_prospect_record", return_value={
            "disqualified": False, "email": "source@example.com",
        }):
            result = self.send()
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], "source@example.com")
        self.assertEqual(result["subject"], "Demo")
        self.assertEqual(result["from"], agent.data_service.get_rep("rep_amills")["email"])

    def test_source_email_works_without_supplied_email(self):
        with patch.object(agent.data_service, "get_prospect_record", return_value={"email": "source@example.com"}):
            result = self.send({"prospect_id": "LEAD-123"})
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], "source@example.com")

    def test_missing_id_preserves_supplied_email(self):
        result = self.send({"name": "Prospect", "email": "supplied@example.com"})
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], "supplied@example.com")

    def test_missing_record_or_source_email_preserves_supplied_email(self):
        for record in (None, {}, {"disqualified": False, "email": ""}):
            with self.subTest(record=record), \
                    patch.object(agent.data_service, "get_prospect_record", return_value=record):
                result = self.send()
            self.assertEqual(result["status"], "sent")
            self.assertEqual(result["to"], "supplied@example.com")

    def test_missing_email_still_fails(self):
        result = self.send({"name": "Prospect"})
        self.assertEqual(result["status"], "failed")
        self.assertIn("no email address", result["error"])

    def test_source_records_replay_through_contact_lookup_and_send(self):
        for pid, record in agent.data_service.PROSPECTS.items():
            with self.subTest(prospect_id=pid):
                prospect = agent.get_prospect.invoke({"prospect_id": pid})["prospect"]
                result = self.send(prospect)
                if record.get("disqualified"):
                    self.assertEqual(result["status"], "blocked")
                    self.assertIn("disqualified", result["error"])
                else:
                    self.assertEqual(result["status"], "sent")
                    self.assertEqual(result["to"], record["email"])


@unittest.skipUnless(os.environ.get("OPENAI_API_KEY"), "Live agent replay requires OPENAI_API_KEY")
class AgentEmailReplayTests(unittest.TestCase):
    def test_email_requests_report_tool_outcome_without_retrying_blocked_send(self):
        for disqualified in (True, False):
            pid = next(
                pid for pid, record in agent.data_service.PROSPECTS.items()
                if bool(record.get("disqualified")) == disqualified
            )
            with self.subTest(disqualified=disqualified), \
                    patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
                result = agent.run_agent(
                    f"Email {pid} to schedule a demo next week.", user_id="rep_amills",
                )
                sends = [
                    json.loads(message.content) for message in result["messages"]
                    if getattr(message, "type", None) == "tool"
                    and getattr(message, "name", None) == "send_prospect_email"
                ]
                self.assertTrue(sends)
                if disqualified:
                    self.assertEqual(len(sends), 1)
                    self.assertEqual(sends[0]["status"], "blocked")
                    self.assertIn("disqualified", result["reply"].lower())
                    self.assertRegex(result["reply"].lower(), r"not|wasn't|cannot|can't|unable|blocked")
                else:
                    self.assertTrue(all(send["status"] == "sent" for send in sends))


if __name__ == "__main__":
    unittest.main()
