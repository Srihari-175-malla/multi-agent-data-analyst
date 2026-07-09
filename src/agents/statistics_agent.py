"""Statistics Agent: chooses and runs formal statistical tests to answer a sub-question."""
from typing import Dict

from src.agents.base_agent import BaseAgent, ToolExecutionOutcome
from src.data.data_loader import Dataset
from src.memory.agent_memory import AgentMemory
from src.tools.stats_tool import StatsTool

SYSTEM_PROMPT = """You are the Statistics Agent in a multi-agent data analysis system. \
You are given a sub-question and a dataset. Use the `run_statistical_test` tool to run \
rigorous statistical tests rather than eyeballing numbers. Available tests:
- period_over_period_change(date_col, metric_col, period_a=[start,end], period_b=[start,end]): \
Welch's t-test comparing the metric's mean between two date ranges.
- correlation(col_a, col_b, method="pearson"|"spearman")
- trend_test(date_col, metric_col): linear regression slope of the metric over time.
- anova(group_col, metric_col): one-way ANOVA across a categorical grouping.
- seasonal_decompose(date_col, metric_col, period=7): additive seasonal decomposition.
- changepoint_detection(date_col, metric_col): finds the point of largest mean shift.
Always report the test's statistic, p-value, and whether the result is statistically \
significant (p < 0.05) in your final answer — never claim significance without checking p-value."""

TOOLS = [
    {
        "name": "run_statistical_test",
        "description": "Run a named statistical test against the dataset and return statistic/p-value/details.",
        "input_schema": {
            "type": "object",
            "properties": {
                "test": {
                    "type": "string",
                    "enum": [
                        "period_over_period_change", "correlation", "trend_test",
                        "anova", "seasonal_decompose", "changepoint_detection",
                    ],
                },
                "kwargs": {"type": "object", "description": "Keyword arguments for the chosen test."},
            },
            "required": ["test", "kwargs"],
        },
    }
]


class StatisticsAgent(BaseAgent):
    agent_name = "statistics_agent"

    def analyze(self, sub_question: str, dataset: Dataset, memory: AgentMemory, round_num: int = 0, feedback: str = "") -> str:
        stats_tool = StatsTool(dataset.df)

        def executor(tool_name: str, tool_input: Dict) -> ToolExecutionOutcome:
            if tool_name != "run_statistical_test":
                return ToolExecutionOutcome(False, f"Unknown tool '{tool_name}'")
            test = tool_input.get("test")
            kwargs = tool_input.get("kwargs", {}) or {}
            result = stats_tool.run(test, **kwargs)
            if result.success:
                summary = f"test={result.test_name} statistic={result.statistic} p_value={result.p_value} details={result.details}"
                memory.add_evidence(self.agent_name, "stat_result", summary, round=round_num)
            else:
                summary = f"ERROR running {test}: {result.error}"
            return ToolExecutionOutcome(success=result.success, summary_for_model=summary, raw=result)

        user_prompt = (
            f"Dataset columns and dtypes: {dataset.schema}\n"
            f"Number of rows: {len(dataset.df)}\n\n"
            f"Sub-question: {sub_question}\n"
        )
        if feedback:
            user_prompt += f"\nRevision feedback from the Critic (address this): {feedback}\n"

        return self.run(SYSTEM_PROMPT, user_prompt, TOOLS, executor, memory=memory, round_num=round_num)
