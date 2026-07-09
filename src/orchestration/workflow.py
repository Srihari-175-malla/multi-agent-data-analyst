"""
End-to-end workflow: Manager plans -> workers execute -> Critic reviews ->
(REVISE -> Manager re-routes feedback -> targeted workers re-run, bounded by
max_revision_rounds) -> APPROVE -> Report agent writes the final report.

This is the single object the API / CLI / evaluator call.
"""
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List

from src.agents.critic_agent import CriticAgent
from src.agents.manager_agent import ManagerAgent
from src.agents.python_agent import PythonAgent
from src.agents.report_agent import ReportAgent
from src.agents.sql_agent import SQLAgent
from src.agents.statistics_agent import StatisticsAgent
from src.config import settings
from src.data.data_loader import Dataset, DataLoader
from src.memory.agent_memory import AgentMemory
from src.utils_logging import get_logger

logger = get_logger(__name__)

_AGENT_REGISTRY = {
    "sql_agent": SQLAgent,
    "python_agent": PythonAgent,
    "statistics_agent": StatisticsAgent,
}


@dataclass
class AnalysisResult:
    session_id: str
    question: str
    report: str
    chart_paths: List[str]
    critic_rounds: List[Dict]
    audit_trail: List[Dict]
    revision_rounds_used: int
    elapsed_seconds: float


class AnalysisWorkflow:
    def __init__(self):
        self.manager = ManagerAgent()
        self.critic = CriticAgent()
        self.reporter = ReportAgent()
        self._workers = {name: cls() for name, cls in _AGENT_REGISTRY.items()}
        self.max_revision_rounds = settings.max_revision_rounds

    def run(self, dataset_id: str, question: str) -> AnalysisResult:
        start = time.time()
        dataset = DataLoader.get(dataset_id)
        session_id = uuid.uuid4().hex[:12]
        memory = AgentMemory(session_id=session_id)

        sub_tasks = self.manager.plan(question, dataset.schema)
        logger.info(f"[{session_id}] Manager plan: {sub_tasks}")

        round_num = 0
        self._execute_sub_tasks(sub_tasks, dataset, memory, round_num)

        verdict, feedback = self._run_critic(question, memory, round_num)

        while verdict == "REVISE" and round_num < self.max_revision_rounds:
            round_num += 1
            logger.info(f"[{session_id}] Critic requested revision (round {round_num}): {feedback}")
            revisions = self.manager.plan_revision(feedback, sub_tasks)
            if not revisions:
                logger.warning(f"[{session_id}] Manager produced no revisions; stopping revision loop")
                break
            self._execute_sub_tasks(revisions, dataset, memory, round_num)
            verdict, feedback = self._run_critic(question, memory, round_num)

        chart_paths = [e.content for e in memory.blackboard if e.kind == "chart"]
        report = self.reporter.write(question, memory.evidence_as_text(), chart_paths)

        grounding = self.critic.check_report_grounding(report, memory.evidence_as_text())
        if not grounding["is_fully_grounded"]:
            logger.warning(f"[{session_id}] Report contains numbers not traced to evidence: {grounding['ungrounded_numbers']}")
            report += (
                f"\n\n> ⚠️ Grounding check flagged potentially unsupported figures: "
                f"{grounding['ungrounded_numbers']}"
            )

        memory.persist()
        elapsed = time.time() - start
        logger.info(f"[{session_id}] Analysis complete in {elapsed:.1f}s, {round_num} revision round(s)")

        return AnalysisResult(
            session_id=session_id,
            question=question,
            report=report,
            chart_paths=chart_paths,
            critic_rounds=memory.critic_rounds,
            audit_trail=[t.__dict__ for t in memory.audit_trail],
            revision_rounds_used=round_num,
            elapsed_seconds=elapsed,
        )

    def _execute_sub_tasks(self, sub_tasks: List[Dict], dataset: Dataset, memory: AgentMemory, round_num: int):
        for task in sub_tasks:
            agent_name = task.get("agent")
            sub_question = task.get("sub_question", "")
            feedback = task.get("feedback", "")
            worker = self._workers.get(agent_name)
            if worker is None:
                logger.warning(f"Unknown agent '{agent_name}' in plan; skipping")
                continue
            logger.info(f"Dispatching to {agent_name}: {sub_question or feedback}")
            worker.analyze(sub_question or feedback, dataset, memory, round_num=round_num, feedback=feedback)

    def _run_critic(self, question: str, memory: AgentMemory, round_num: int):
        result = self.critic.review(question, memory.evidence_as_text())
        memory.add_critic_round(round_num, result["verdict"], result["feedback"])
        return result["verdict"], result["feedback"]
