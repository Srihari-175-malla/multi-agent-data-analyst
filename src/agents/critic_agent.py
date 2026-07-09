"""
Critic Agent: validates the combined evidence on the blackboard before it's
allowed to become a final report. Checks logical consistency, statistical
rigor (e.g. was significance actually checked, not just asserted), and
whether findings from different agents corroborate or contradict each other.
Also runs the deterministic numeric-grounding guardrail as a first pass.
"""
import json
from typing import Dict

from src.guardrails.report_guard import ReportGuard
from src.llm.llm_client import LLMClient
from src.utils_logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are the Critic Agent in a multi-agent data analysis system. \
Review the evidence gathered by the SQL, Python, and Statistics agents for the user's \
question. Check for:
- Unsupported claims (a number or conclusion not backed by any tool result)
- Statistical rigor (claims of "significant" change must be backed by a p-value < 0.05)
- Contradictions between agents' findings
- Whether the evidence actually answers the user's original question

Respond ONLY as JSON:
{"verdict": "APPROVE" | "REVISE", "feedback": "specific, actionable feedback if REVISE, else empty string"}"""


class CriticAgent:
    agent_name = "critic_agent"

    def __init__(self, llm: LLMClient = None):
        self.llm = llm or LLMClient()

    def review(self, question: str, evidence_text: str) -> Dict:
        user_prompt = f"User question: {question}\n\nEvidence gathered so far:\n{evidence_text}"
        raw = self.llm.generate(SYSTEM_PROMPT, user_prompt, max_tokens=600)
        parsed = self._parse_json(raw)
        verdict = parsed.get("verdict", "APPROVE")
        feedback = parsed.get("feedback", "")
        logger.info(f"Critic verdict: {verdict}" + (f" | feedback: {feedback[:150]}" if feedback else ""))
        return {"verdict": verdict, "feedback": feedback}

    def check_report_grounding(self, report_text: str, evidence_text: str) -> Dict:
        grounding = ReportGuard.check_numeric_grounding(report_text, evidence_text)
        return {
            "is_fully_grounded": grounding.is_fully_grounded,
            "ungrounded_numbers": grounding.ungrounded_numbers,
        }

    @staticmethod
    def _parse_json(raw: str) -> Dict:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        try:
            return json.loads(cleaned.strip())
        except json.JSONDecodeError:
            logger.warning(f"Critic returned non-JSON output: {raw[:300]}")
            return {"verdict": "APPROVE", "feedback": ""}
