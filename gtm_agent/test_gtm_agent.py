import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

with patch.dict("os.environ", {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}})

    def send(self, prospect, **kwargs):
        return agent.send_prospect_email.func(
            prospect, "Subject", "Body", self.runtime, **kwargs
        )

    def test_disqualified_source_blocks_untrusted_contact(self):
        for supplied in ({}, {"disqualified": False}, {"disqualified": True}):
            with self.subTest(supplied=supplied), patch.object(agent.uuid, "uuid4") as message_id:
                result = self.send({
                    "prospect_id": "LEAD-50001", "email": "override@example.com", **supplied
                })
                self.assertEqual(result, {
                    "status": "blocked",
                    "error": "Prospect is disqualified; do not contact.",
                    "to": None,
                })
                message_id.assert_not_called()

    def test_unknown_or_missing_id_fails_without_sending(self):
        for prospect in ({"prospect_id": "unknown"}, {}):
            with self.subTest(prospect=prospect), patch.object(agent.uuid, "uuid4") as message_id:
                self.assertEqual(self.send({"email": "override@example.com", **prospect}), {
                    "status": "failed", "error": "Unknown prospect."
                })
                message_id.assert_not_called()

    def test_missing_source_email_fails_without_sending(self):
        with patch.object(agent.data_service, "get_prospect_record", return_value={"disqualified": False}), \
                patch.object(agent.uuid, "uuid4") as message_id:
            self.assertEqual(self.send({"prospect_id": "lead", "email": "override@example.com"}), {
                "status": "failed", "error": "Prospect record has no email address."
            })
            message_id.assert_not_called()

    def test_eligible_source_email_preserves_success(self):
        record = agent.data_service.get_prospect_record("LEAD-70001")
        for supplied in ({}, {"email": "override@example.com", "disqualified": True}):
            with self.subTest(supplied=supplied):
                result = self.send({"prospect_id": "LEAD-70001", "name": record["name"], **supplied})
                message_id = result.pop("message_id")
                self.assertRegex(message_id, r"^msg-[0-9a-f]{12}$")
                self.assertEqual(result, {
                    "status": "sent", "to": record["email"], "to_name": record["name"],
                    "from": "ava.mills@northpoint.com", "from_name": "Ava Mills",
                    "subject": "Subject", "body": "Body",
                })

    def test_explicit_sender_is_preserved(self):
        result = self.send({"prospect_id": "LEAD-70001"}, from_rep={"name": "Rep", "email": "rep@example.com"})
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["from"], "rep@example.com")
        self.assertEqual(result["from_name"], "Rep")


if __name__ == "__main__":
    unittest.main()
