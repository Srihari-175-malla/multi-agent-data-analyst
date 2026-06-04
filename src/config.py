"""Load YAML config + environment variables into a single settings object."""
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


def _deep_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


class Settings:
    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        with open(config_path, "r") as f:
            self._cfg = yaml.safe_load(f)

        self.llm_model = os.getenv("LLM_MODEL", _deep_get(self._cfg, "llm", "model", default="claude-sonnet-5"))
        self.llm_max_tokens = _deep_get(self._cfg, "llm", "max_tokens", default=2000)
        self.llm_temperature = _deep_get(self._cfg, "llm", "temperature", default=0.2)
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

        self.max_tool_iterations = int(
            os.getenv("MAX_AGENT_TOOL_ITERATIONS", _deep_get(self._cfg, "agents", "max_tool_iterations", default=6))
        )
        self.max_revision_rounds = int(
            os.getenv("MAX_REVISION_ROUNDS", _deep_get(self._cfg, "agents", "max_revision_rounds", default=2))
        )

        self.python_exec_timeout_seconds = int(
            os.getenv(
                "PYTHON_EXEC_TIMEOUT_SECONDS",
                _deep_get(self._cfg, "guardrails", "python_exec_timeout_seconds", default=15),
            )
        )
        self.sql_max_rows = int(
            os.getenv("SQL_MAX_ROWS", _deep_get(self._cfg, "guardrails", "sql_max_rows", default=5000))
        )
        self.allowed_python_modules: List[str] = _deep_get(
            self._cfg, "guardrails", "allowed_python_modules", default=[]
        )

        self.uploads_dir = ROOT_DIR / _deep_get(self._cfg, "paths", "uploads_dir", default="data/uploads")
        self.reports_dir = ROOT_DIR / _deep_get(self._cfg, "paths", "reports_dir", default="data/reports")
        self.charts_dir = ROOT_DIR / _deep_get(self._cfg, "paths", "charts_dir", default="data/reports/charts")

        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
