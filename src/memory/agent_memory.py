"""
Shared memory for a single analysis session:
- `blackboard`: structured evidence contributed by each worker agent (SQL
  results, Python findings, stat test results, chart paths) — this is what
  the Critic checks claims against and what the Report agent cites.
- `audit_trail`: every tool call made by every agent, in order, with inputs
  and raw outputs — this is what the evaluation harness and the UI's
  "show agent trace" view consume.
- `long_term_log`: appended to a JSON file on disk so past sessions can be
  reviewed or replayed later (a very small step toward persistent memory
  across sessions, not just within one).
"""
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ToolCallRecord:
    agent: str
    tool: str
    arguments: Dict[str, Any]
    result_summary: str
    success: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class EvidenceItem:
    agent: str
    kind: str  # "sql_result" | "python_finding" | "stat_result" | "chart"
    content: str
    round: int = 0


class AgentMemory:
    def __init__(self, session_id: str, log_path: str = "data/reports/session_log.jsonl"):
        self.session_id = session_id
        self.blackboard: List[EvidenceItem] = []
        self.audit_trail: List[ToolCallRecord] = []
        self.critic_rounds: List[Dict] = []
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def add_evidence(self, agent: str, kind: str, content: str, round: int = 0):
        self.blackboard.append(EvidenceItem(agent=agent, kind=kind, content=content, round=round))

    def add_tool_call(self, agent: str, tool: str, arguments: Dict, result_summary: str, success: bool):
        self.audit_trail.append(
            ToolCallRecord(agent=agent, tool=tool, arguments=arguments, result_summary=result_summary, success=success)
        )

    def add_critic_round(self, round_num: int, verdict: str, feedback: str):
        self.critic_rounds.append({"round": round_num, "verdict": verdict, "feedback": feedback})

    def evidence_as_text(self) -> str:
        lines = []
        for item in self.blackboard:
            lines.append(f"[{item.agent} | {item.kind} | round {item.round}]\n{item.content}\n")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "blackboard": [asdict(e) for e in self.blackboard],
            "audit_trail": [asdict(t) for t in self.audit_trail],
            "critic_rounds": self.critic_rounds,
        }

    def persist(self):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(self.to_dict(), default=str) + "\n")
