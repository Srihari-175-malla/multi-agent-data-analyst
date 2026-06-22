"""Executes guardrailed, read-only SQL against a loaded dataset's DuckDB connection."""
from dataclasses import dataclass
from typing import List

from src.config import settings
from src.data.data_loader import Dataset
from src.guardrails.sql_guard import SQLGuard
from src.utils_logging import get_logger

logger = get_logger(__name__)


@dataclass
class SQLResult:
    success: bool
    columns: List[str] = None
    rows: List[list] = None
    error: str = None
    row_count: int = 0

    def to_markdown(self, max_display_rows: int = 20) -> str:
        if not self.success:
            return f"SQL ERROR: {self.error}"
        if not self.rows:
            return "(query returned 0 rows)"
        header = "| " + " | ".join(self.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(self.columns)) + " |"
        body = [
            "| " + " | ".join(str(v) for v in row) + " |"
            for row in self.rows[:max_display_rows]
        ]
        truncation_note = ""
        if self.row_count > max_display_rows:
            truncation_note = f"\n(showing {max_display_rows} of {self.row_count} rows)"
        return "\n".join([header, sep] + body) + truncation_note


class SQLTool:
    def __init__(self, dataset: Dataset, max_rows: int = None):
        self.dataset = dataset
        self.max_rows = max_rows or settings.sql_max_rows

    def execute(self, query: str) -> SQLResult:
        guard = SQLGuard.check(query)
        if not guard.allowed:
            logger.warning(f"SQL guard blocked query: {guard.reason}")
            return SQLResult(success=False, error=f"Blocked by guardrail: {guard.reason}")

        safe_query = SQLGuard.enforce_row_limit(query, self.max_rows)
        # substitute a friendly {table} placeholder if the agent used it
        safe_query = safe_query.replace("{table}", self.dataset.table_name)

        try:
            result = self.dataset.con.execute(safe_query)
            columns = [d[0] for d in result.description]
            rows = result.fetchall()
            return SQLResult(success=True, columns=columns, rows=rows, row_count=len(rows))
        except Exception as e:
            logger.warning(f"SQL execution error: {e}")
            return SQLResult(success=False, error=str(e))
