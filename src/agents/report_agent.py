"""
Report Agent: synthesizes the approved evidence blackboard into a final,
citation-grounded markdown report, embedding references to generated charts.
"""
from typing import Dict, List

from src.llm.llm_client import LLMClient

SYSTEM_PROMPT = """You are the Report Agent. Write a final analytical report answering \
the user's question, using ONLY the evidence provided below (from SQL queries, Python \
analysis, and statistical tests). Structure:

## Executive Summary
(2-3 sentences with the headline finding)

## Key Findings
(bulleted, each finding citing the concrete number/test result it's based on)

## Statistical Evidence
(summarize the statistical tests run and their significance)

## Caveats
(anything the evidence does NOT establish, or data limitations)

Do not introduce any number that isn't present in the evidence. If evidence is \
insufficient for a strong causal claim, say so explicitly rather than overstating."""


class ReportAgent:
    agent_name = "report_agent"

    def __init__(self, llm: LLMClient = None):
        self.llm = llm or LLMClient()

    def write(self, question: str, evidence_text: str, chart_paths: List[str]) -> str:
        chart_note = ""
        if chart_paths:
            chart_note = "\n\nCharts generated during analysis (reference by filename where relevant):\n" + "\n".join(chart_paths)
        user_prompt = f"User question: {question}\n\nEvidence:\n{evidence_text}{chart_note}"
        return self.llm.generate(SYSTEM_PROMPT, user_prompt, max_tokens=1800)
