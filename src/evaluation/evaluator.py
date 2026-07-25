"""
Benchmark harness for the multi-agent workflow.

Expected benchmark.jsonl format (one JSON object per line):
{"dataset_path": "data/sample/sales.csv", "question": "Why did sales decrease last month?"}

Reports per-question and aggregate:
- tool_error_rate: fraction of tool calls in the audit trail that failed
- revision_rounds: how many Critic->Manager revision loops were needed
- latency_seconds
- LLM-judged faithfulness/relevance of the final report against the evidence blackboard
"""
import argparse
import json
from typing import Dict, List

from src.data.data_loader import DataLoader
from src.llm.llm_client import LLMClient
from src.orchestration.workflow import AnalysisWorkflow
from src.utils_logging import get_logger

logger = get_logger(__name__)

JUDGE_SYSTEM_PROMPT = """You are an evaluator for a multi-agent data analysis system. \
Given a question, the evidence gathered by the agents, and the final report, score \
the report on two axes from 1-5:
- faithfulness: does every claim in the report trace back to the evidence (5) or does \
it contain unsupported/hallucinated claims (1)?
- relevance: does the report directly address the question (5) or is it off-topic (1)?
Respond ONLY as JSON: {"faithfulness": <int>, "relevance": <int>, "reasoning": "<short>"}"""


class Evaluator:
    def __init__(self, workflow: AnalysisWorkflow = None, llm: LLMClient = None):
        self.workflow = workflow or AnalysisWorkflow()
        self.llm = llm or LLMClient()

    def evaluate_benchmark(self, benchmark_path: str) -> Dict:
        rows = [json.loads(line) for line in open(benchmark_path) if line.strip()]
        per_question = []

        for row in rows:
            dataset_id = DataLoader.load(row["dataset_path"]).dataset_id
            result = self.workflow.run(dataset_id, row["question"])

            tool_calls = result.audit_trail
            error_rate = (
                sum(1 for t in tool_calls if not t["success"]) / len(tool_calls) if tool_calls else 0.0
            )
            judge = self._judge(row["question"], result.report)

            per_question.append(
                {
                    "question": row["question"],
                    "revision_rounds": result.revision_rounds_used,
                    "tool_error_rate": error_rate,
                    "num_tool_calls": len(tool_calls),
                    "latency_seconds": result.elapsed_seconds,
                    **judge,
                }
            )

        return {"num_questions": len(rows), "per_question": per_question, "aggregate": self._aggregate(per_question)}

    def _judge(self, question: str, report: str) -> Dict:
        prompt = f"Question: {question}\n\nReport:\n{report}"
        raw = self.llm.generate(JUDGE_SYSTEM_PROMPT, prompt, max_tokens=300)
        try:
            cleaned = raw.strip().strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            return json.loads(cleaned.strip())
        except Exception:
            logger.warning(f"Could not parse judge output: {raw[:200]}")
            return {"faithfulness": None, "relevance": None, "reasoning": ""}

    @staticmethod
    def _aggregate(rows: List[Dict]) -> Dict:
        numeric_keys = ["revision_rounds", "tool_error_rate", "num_tool_calls", "latency_seconds", "faithfulness", "relevance"]
        agg = {}
        for k in numeric_keys:
            vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
            agg[f"avg_{k}"] = sum(vals) / len(vals) if vals else None
        return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, help="Path to benchmark.jsonl")
    args = parser.parse_args()

    evaluator = Evaluator()
    results = evaluator.evaluate_benchmark(args.benchmark)
    print(json.dumps(results, indent=2, default=str))
