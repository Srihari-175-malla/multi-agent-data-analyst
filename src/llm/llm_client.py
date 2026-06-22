"""
Thin wrapper around the Anthropic Claude API with native tool use.
Every agent in this system is built on top of `generate_with_tools`, which
runs a single model turn and returns whatever mix of text / tool_use blocks
Claude produced — the agent's own loop (src/agents/base_agent.py) decides
whether to execute a tool and feed the result back.
"""
from typing import Dict, List, Optional

import anthropic

from src.config import settings
from src.utils_logging import get_logger

logger = get_logger(__name__)


class LLMClient:
    def __init__(self, model: str = None, max_tokens: int = None, temperature: float = None):
        if not settings.anthropic_api_key:
            logger.warning("ANTHROPIC_API_KEY is not set — LLM calls will fail until it is configured.")
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = model or settings.llm_model
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.temperature = temperature if temperature is not None else settings.llm_temperature

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: Optional[int] = None) -> str:
        """Simple single-turn text generation (used by Critic/Report agents' final write-ups)."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(b.text for b in response.content if b.type == "text")

    def generate_with_tools(
        self,
        system_prompt: str,
        messages: List[Dict],
        tools: List[Dict],
        max_tokens: Optional[int] = None,
    ):
        """Runs one model turn with tools available. Returns the raw Anthropic message object
        so the caller can inspect stop_reason and content blocks (text and/or tool_use)."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=messages,
            tools=tools,
        )
        return response
