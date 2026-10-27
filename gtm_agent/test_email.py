import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent.gtm_agent import run_agent, send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def send(self, body, subject="Following up"):
        return send_prospect_email.func(
            prospect={"name": "Prospect", "email": "prospect@example.com"},
            subject=subject,
            body=body,
            runtime=SimpleNamespace(config={"metadata": {"user_id": "rep_sbrown"}}),
        )

    def test_attachment_claims_fail_before_sending(self):
        for body in (
            "Attached is our pricing deck for your review.",
            "The pricing deck is included as an attachment.",
            "I've attached the brochure.",
            "Please find attached the proposal.",
            "The document is attached.",
            "No deck is attached, but the proposal is attached.",
            "No deck is attached, and the proposal is attached.",
            "Attachments are unsupported. Attached is the deck.",
        ):
            with self.subTest(body=body), patch(
                "gtm_agent.gtm_agent.uuid.uuid4",
                side_effect=AssertionError("A rejected email must not be sent"),
            ):
                self.assertEqual(self.send(body), {
                    "status": "failed",
                    "error": "Attachments are not supported; remove references to attachments or share a link.",
                })

    def test_plain_text_and_truthful_limitations_are_sent(self):
        for body in (
            "Would you like to discuss pricing?",
            "No document is attached.",
            "No document is included as an attachment.",
            "The pricing deck is not attached.",
            "I haven't attached a document.",
            "The deck is not included as an attachment.",
            "The deck cannot be attached.",
            "The deck has never been attached.",
            "Attachments are unsupported; I cannot attach the deck.",
            "We do not support attachments.",
            "The deck is available at https://example.com/deck.",
        ):
            with self.subTest(body=body):
                result = self.send(body)
                self.assertEqual(result["status"], "sent")
                self.assertEqual(result["body"], body)
                self.assertEqual(result["to"], "prospect@example.com")
                self.assertTrue(result["message_id"])

    def test_attachment_claim_in_subject_is_rejected(self):
        self.assertEqual(
            self.send("Please review.", subject="Attached is the proposal")["status"],
            "failed",
        )

    @unittest.skipUnless(os.environ.get("RUN_LIVE_AGENT_TESTS"), "Requires model credentials and explicit opt-in")
    def test_pricing_deck_request_discloses_unsent_document(self):
        from langchain_openai import ChatOpenAI
        from deepagents import create_deep_agent
        from gtm_agent import gtm_agent as module

        graph = create_deep_agent(
            model=ChatOpenAI(model=module.MODEL_NAME, temperature=0),
            tools=[module.get_current_rep, module.get_prospect, send_prospect_email],
            system_prompt=module.SYSTEM_PROMPT,
        )
        with patch.object(module, "gtm_agent", graph):
            result = run_agent("Email LEAD-50003 to send over our pricing deck.", user_id="rep_sbrown")
        reply = result["reply"].lower()
        self.assertRegex(reply, r"(?:cannot|can't|unable|unsupported|not support)")
        self.assertIn("deck", reply)
        sends = [
            message for message in result["messages"]
            if getattr(message, "name", None) == "send_prospect_email"
        ]
        self.assertTrue(sends)
        self.assertFalse(module._claims_attachment(reply))


if __name__ == "__main__":
    unittest.main()
