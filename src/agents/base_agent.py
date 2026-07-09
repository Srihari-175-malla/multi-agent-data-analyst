"""
Generic tool-calling agent loop shared by SQL/Python/Statistics agents.

Runs Claude turn-by-turn: if the model requests a tool, the loop executes it
via `tool_executor` (which raises on hard failure, catches and reports soft
guardrail failures) and feeds the result back as a tool_result block. This
continues until the model stops requesting tools or `max_iterations` is hit
— which itself is a guardrail against runaway agent loops.
"""
from typing import Any, Callable, Dict, List, Optional

from src.config import settings
from src.llm.llm_client import LLMClient
from src.memory.agent_memory import AgentMemory
from src.utils_logging import get_logger

logger = get_logger(__name__)

ToolExecutor = Callable[[str, Dict[str, Any]], "ToolExecutionOutcome"]


class ToolExecutionOutcome:
    """What a tool executor returns to the agent loop."""

    def __init__(self, success: bool, summary_for_model: str, raw: Any = None):
        self.success = success
        self.summary_for_model = summary_for_model
        self.raw = raw


class BaseAgent:
    agent_name: str = "base_agent"

    def __init__(self, llm: LLMClient = None, max_iterations: int = None):
        self.llm = llm or LLMClient()
        self.max_iterations = max_iterations or settings.max_tool_iterations

    def run(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: List[Dict],
        tool_executor: ToolExecutor,
        memory: Optional[AgentMemory] = None,
        round_num: int = 0,
    ) -> str:
        messages = [{"role": "user", "content": user_prompt}]

        for iteration in range(self.max_iterations):
            response = self.llm.generate_with_tools(system_prompt, messages, tools)
            messages.append({"role": "assistant", "content": response.content})

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            if not tool_use_blocks:
                final_text = "".join(b.text for b in response.content if b.type == "text")
                logger.info(f"[{self.agent_name}] finished after {iteration + 1} iteration(s)")
                return final_text

            tool_result_blocks = []
            for block in tool_use_blocks:
                outcome = self._safe_execute(tool_executor, block.name, block.input)

                if memory is not None:
                    memory.add_tool_call(
                        agent=self.agent_name,
                        tool=block.name,
                        arguments=block.input,
                        result_summary=outcome.summary_for_model[:500],
                        success=outcome.success,
                    )

                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": outcome.summary_for_model,
                        "is_error": not outcome.success,
                    }
                )

            messages.append({"role": "user", "content": tool_result_blocks})

        logger.warning(f"[{self.agent_name}] hit max_iterations={self.max_iterations} without finishing")
        return "(Agent stopped: reached the maximum number of tool-call iterations without a final answer.)"

    @staticmethod
    def _safe_execute(tool_executor: ToolExecutor, tool_name: str, tool_input: Dict) -> ToolExecutionOutcome:
        try:
            return tool_executor(tool_name, tool_input)
        except Exception as e:
            logger.warning(f"Tool '{tool_name}' raised an unhandled exception: {e}")
            return ToolExecutionOutcome(success=False, summary_for_model=f"Tool execution error: {e}")
