"""Python Agent: writes and executes sandboxed pandas/matplotlib code to answer a sub-question."""
from typing import Dict

from src.agents.base_agent import BaseAgent, ToolExecutionOutcome
from src.data.data_loader import Dataset
from src.memory.agent_memory import AgentMemory
from src.tools.python_exec_tool import PythonExecTool

SYSTEM_PROMPT = """You are the Python Agent in a multi-agent data analysis system. \
You are given a sub-question and a pandas DataFrame available as `df` inside the \
sandbox. Use the `execute_python` tool to write pandas/numpy/matplotlib/scipy code \
to compute what's needed. Conventions:
- Assign your final finding to a variable named `result`.
- To produce a chart, call plt.plot/bar/etc. and do NOT call plt.show(); the chart \
is auto-captured when the code finishes.
- Only the following are available: pandas as pd, numpy as np, math, statistics, \
scipy, scipy.stats as stats, statsmodels.api as sm, matplotlib.pyplot as plt, json, datetime.
- No file/network access, no imports beyond those already provided.
Run as many code snippets as needed, then give a concise final answer stating \
concrete numbers, not estimates."""

TOOLS = [
    {
        "name": "execute_python",
        "description": "Execute Python code in a sandboxed environment with `df` (the dataset) already loaded.",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python code to execute."}},
            "required": ["code"],
        },
    }
]


class PythonAgent(BaseAgent):
    agent_name = "python_agent"

    def analyze(self, sub_question: str, dataset: Dataset, memory: AgentMemory, round_num: int = 0, feedback: str = "") -> str:
        exec_tool = PythonExecTool(dataset.df)

        def executor(tool_name: str, tool_input: Dict) -> ToolExecutionOutcome:
            if tool_name != "execute_python":
                return ToolExecutionOutcome(False, f"Unknown tool '{tool_name}'")
            result = exec_tool.execute(tool_input.get("code", ""))
            if result.success:
                summary_parts = []
                if result.stdout:
                    summary_parts.append(f"stdout:\n{result.stdout}")
                if result.result_repr:
                    summary_parts.append(f"result: {result.result_repr}")
                if result.chart_paths:
                    summary_parts.append(f"charts saved: {result.chart_paths}")
                    memory.add_evidence(self.agent_name, "chart", ", ".join(result.chart_paths), round=round_num)
                summary = "\n".join(summary_parts) or "(code executed with no output; consider assigning to `result`)"
                memory.add_evidence(
                    self.agent_name, "python_finding",
                    f"Code:\n{tool_input.get('code')}\nOutput:\n{summary}", round=round_num,
                )
            else:
                summary = f"ERROR: {result.error}"
            return ToolExecutionOutcome(success=result.success, summary_for_model=summary, raw=result)

        user_prompt = (
            f"Dataset columns and dtypes: {dataset.schema}\n"
            f"Number of rows: {len(dataset.df)}\n\n"
            f"Sub-question: {sub_question}\n"
        )
        if feedback:
            user_prompt += f"\nRevision feedback from the Critic (address this): {feedback}\n"

        return self.run(SYSTEM_PROMPT, user_prompt, TOOLS, executor, memory=memory, round_num=round_num)
