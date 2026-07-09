"""
Manager Agent: decomposes the user's question into sub-questions for the
SQL / Python / Statistics agents, and — on a Critic revision request —
figures out which specific worker(s) need to redo their work and with what
targeted feedback.
"""
import json
from typing import Dict, List

from src.llm.llm_client import LLMClient
from src.utils_logging import get_logger

logger = get_logger(__name__)

PLAN_SYSTEM_PROMPT = """You are the Manager Agent coordinating a team of data-analysis \
specialists: a SQL Agent (queries the raw data), a Python Agent (pandas/matplotlib \
analysis and charts), and a Statistics Agent (formal statistical tests). Given the \
user's question and the dataset schema, decompose it into 1-3 concrete sub-questions, \
assigning each to the most appropriate agent(s). A sub-question can be assigned to \
more than one agent if cross-validation would help (e.g. SQL for the raw numbers, \
Statistics for whether the change is significant).

Respond ONLY as JSON in this exact shape, no prose:
{
  "sub_tasks": [
    {"agent": "sql_agent" | "python_agent" | "statistics_agent", "sub_question": "..."}
  ]
}"""

REVISION_SYSTEM_PROMPT = """You are the Manager Agent. The Critic Agent rejected the \
team's evidence with the feedback below. Decide which agent(s) need to redo their \
work and with what specific instruction to fix the issue. Respond ONLY as JSON:
{
  "revisions": [
    {"agent": "sql_agent" | "python_agent" | "statistics_agent", "feedback": "specific instruction"}
  ]
}"""


class ManagerAgent:
    agent_name = "manager_agent"

    def __init__(self, llm: LLMClient = None):
        self.llm = llm or LLMClient()

    def plan(self, question: str, schema: Dict[str, str]) -> List[Dict]:
        user_prompt = f"User question: {question}\n\nDataset schema: {json.dumps(schema)}"
        raw = self.llm.generate(PLAN_SYSTEM_PROMPT, user_prompt, max_tokens=800)
        parsed = self._parse_json(raw)
        sub_tasks = parsed.get("sub_tasks", [])
        if not sub_tasks:
            logger.warning("Manager produced no sub_tasks; falling back to a single SQL sub-task")
            sub_tasks = [{"agent": "sql_agent", "sub_question": question}]
        return sub_tasks

    def plan_revision(self, critic_feedback: str, prior_sub_tasks: List[Dict]) -> List[Dict]:
        user_prompt = (
            f"Prior sub-tasks: {json.dumps(prior_sub_tasks)}\n\nCritic feedback: {critic_feedback}"
        )
        raw = self.llm.generate(REVISION_SYSTEM_PROMPT, user_prompt, max_tokens=600)
        parsed = self._parse_json(raw)
        return parsed.get("revisions", [])

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
            logger.warning(f"Manager returned non-JSON output: {raw[:300]}")
            return {}
