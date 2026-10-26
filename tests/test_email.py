import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from dotenv import load_dotenv

load_dotenv()
HAS_MODEL_KEY = bool(os.environ.get("OPENAI_API_KEY"))

with patch.dict(os.environ, {"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or "test-only"}):
    from gtm_agent.gtm_agent import run_agent, send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def test_sends_body_and_link_without_attachments(self):
        body = "Please review https://example.com/pricing.pdf before our call."
        result = send_prospect_email.func(
            prospect={"name": "Sofia", "email": "sofia@example.com"},
            subject="Pricing review",
            body=body,
            runtime=SimpleNamespace(config={"metadata": {"user_id": "rep_sbrown"}}),
        )
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["body"], body)
        self.assertEqual(result["to"], "sofia@example.com")
        self.assertEqual(result["from"], "sara.brown@northpoint.com")
        self.assertNotIn("attachments", result)


@unittest.skipUnless(HAS_MODEL_KEY, "Requires OPENAI_API_KEY for live agent behavior")
class AgentEmailTests(unittest.TestCase):
    def send_email(self, request):
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
            result = run_agent(request, user_id="rep_sbrown", environment="test")
        emails = [
            json.loads(message.content)
            for message in result["messages"]
            if message.type == "tool" and message.name == "send_prospect_email"
        ]
        self.assertEqual(len(emails), 1)
        self.assertEqual(emails[0]["status"], "sent")
        self.assertEqual(emails[0]["to"], "sofia.rossi@greenfieldnetworks.com")
        return emails[0], result["reply"]

    def assert_file_not_attached(self, email, reply):
        positive_claim = r"\b(?:attached (?:is|are)|(?:I(?:'ve| have)?|we(?:'ve| have)?|file|deck|study) (?:is |was |has been )?attached|please find .*attached)\b"
        self.assertNotRegex(email["body"].lower(), positive_claim)
        self.assertNotRegex(reply.lower(), positive_claim)
        self.assertRegex(
            reply.lower(),
            r"(?:not|n't|no|cannot|unable to|without)[^.\n]{0,100}attach|attach[^.\n]{0,100}(?:not|n't|unavailable|unsupported)",
        )

    def test_pricing_deck_without_link(self):
        email, reply = self.send_email("Email LEAD-50003 our pricing deck.")
        self.assert_file_not_attached(email, reply)
        self.assertNotRegex(email["body"], r"https?://")
        self.assertNotRegex(reply.lower(), r"(?:deck|file) (?:was |has been )?(?:delivered|included|sent)")

    def test_pricing_deck_with_rep_link(self):
        link = "https://example.com/pricing.pdf"
        email, reply = self.send_email(f"Email LEAD-50003 our pricing deck: {link}")
        self.assert_file_not_attached(email, reply)
        self.assertIn(link, email["body"])
        self.assertIn("link", reply.lower())

    def test_case_study_without_link(self):
        email, reply = self.send_email("Email LEAD-50003 our customer case study.")
        self.assert_file_not_attached(email, reply)
        self.assertNotRegex(email["body"], r"https?://")

    def test_ordinary_email(self):
        email, reply = self.send_email(
            "Email LEAD-50003 asking for availability for a pricing review next week."
        )
        self.assertIn("pricing", email["body"].lower())
        self.assertRegex(email["body"].lower(), r"availab|time|schedule")
        self.assertRegex(reply.lower(), r"sent|emailed")
        self.assertNotIn("attach", email["body"].lower())
        self.assertNotIn("attach", reply.lower())


if __name__ == "__main__":
    unittest.main()
