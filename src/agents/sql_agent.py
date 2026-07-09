"""SQL Agent: writes and executes read-only SQL against the loaded dataset to answer a sub-question."""
from typing import Dict

from src.agents.base_agent import BaseAgent, ToolExecutionOutcome
from src.data.data_loader import Dataset
from src.memory.agent_memory import AgentMemory
from src.tools.sql_tool import SQLTool

SYSTEM_PROMPT = """You are the SQL Agent in a multi-agent data analysis system. \
You are given a sub-question and a dataset schema. Use the `execute_sql` tool to \
query the data (DuckDB dialect, table name is given) to gather the facts needed \
to answer the sub-question. Only read-only SELECT/WITH queries are allowed. \
Run as many queries as you need, then give a concise final answer summarizing \
the concrete numbers you found — always state the actual values, never estimate."""

TOOLS = [
    {
        "name": "execute_sql",
        "description": "Execute a read-only SQL SELECT query against the dataset table and return the result rows.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A SELECT/WITH SQL query. Use {table} as the table name placeholder."}
            },
            "required": ["query"],
        },
    }
]


class SQLAgent(BaseAgent):
    agent_name = "sql_agent"

    def analyze(self, sub_question: str, dataset: Dataset, memory: AgentMemory, round_num: int = 0, feedback: str = "") -> str:
        sql_tool = SQLTool(dataset)

        def executor(tool_name: str, tool_input: Dict) -> ToolExecutionOutcome:
            if tool_name != "execute_sql":
                return ToolExecutionOutcome(False, f"Unknown tool '{tool_name}'")
            result = sql_tool.execute(tool_input.get("query", ""))
            summary = result.to_markdown()
            if result.success:
                memory.add_evidence(self.agent_name, "sql_result", f"Query: {tool_input.get('query')}\nResult:\n{summary}", round=round_num)
            return ToolExecutionOutcome(success=result.success, summary_for_model=summary, raw=result)

        user_prompt = (
            f"Table name: {dataset.table_name}\n"
            f"Schema: {dataset.schema}\n\n"
            f"Sub-question: {sub_question}\n"
        )
        if feedback:
            user_prompt += f"\nRevision feedback from the Critic (address this): {feedback}\n"

        return self.run(SYSTEM_PROMPT, user_prompt, TOOLS, executor, memory=memory, round_num=round_num)
