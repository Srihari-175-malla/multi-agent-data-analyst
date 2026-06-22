"""
Grounding guardrail for the Report Agent: extracts numeric tokens from the
draft report and checks that each one plausibly appears somewhere in the
evidence blackboard. This is a cheap, deterministic first line of defense
against numeric hallucination; the Critic Agent's LLM-based review is the
second, more semantic line of defense.
"""
import re
from dataclasses import dataclass
from typing import List

_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")


@dataclass
class GroundingResult:
    ungrounded_numbers: List[str]

    @property
    def is_fully_grounded(self) -> bool:
        return len(self.ungrounded_numbers) == 0


class ReportGuard:
    @staticmethod
    def check_numeric_grounding(report_text: str, evidence_text: str) -> GroundingResult:
        report_numbers = set(_NUMBER_RE.findall(report_text))
        evidence_numbers = set(_NUMBER_RE.findall(evidence_text))

        def _normalize(n: str) -> str:
            return n.replace(",", "").rstrip("%")

        evidence_normalized = {_normalize(n) for n in evidence_numbers}
        ungrounded = []
        for n in report_numbers:
            norm = _normalize(n)
            if not norm or norm in {"0", "1", "2", "3"}:  # trivial numbers, skip
                continue
            if norm not in evidence_normalized:
                ungrounded.append(n)
        return GroundingResult(ungrounded_numbers=ungrounded)
