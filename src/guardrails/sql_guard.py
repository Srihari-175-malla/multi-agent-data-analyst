"""
Static guardrail for SQL: only a single read-only SELECT/WITH statement is
allowed. This runs *before* anything touches DuckDB, so a malicious or
mistaken write/DDL statement never reaches the engine.
"""
import re
from dataclasses import dataclass

_FORBIDDEN_KEYWORDS = [
    "insert", "update", "delete", "drop", "alter", "create", "truncate",
    "attach", "detach", "copy", "export", "import", "pragma", "call",
    "grant", "revoke", "vacuum", "install", "load",
]

_STATEMENT_SPLIT_RE = re.compile(r";\s*\S")  # detects a second statement after a semicolon


@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""


class SQLGuard:
    @staticmethod
    def check(query: str) -> GuardResult:
        stripped = query.strip().rstrip(";").strip()
        if not stripped:
            return GuardResult(False, "Empty query")

        lowered = stripped.lower()
        if not (lowered.startswith("select") or lowered.startswith("with")):
            return GuardResult(False, "Only SELECT / WITH (read-only) queries are allowed")

        if _STATEMENT_SPLIT_RE.search(query):
            return GuardResult(False, "Multiple statements are not allowed")

        for kw in _FORBIDDEN_KEYWORDS:
            if re.search(rf"\b{kw}\b", lowered):
                return GuardResult(False, f"Forbidden keyword detected: '{kw}'")

        return GuardResult(True)

    @staticmethod
    def enforce_row_limit(query: str, max_rows: int) -> str:
        stripped = query.strip().rstrip(";").strip()
        if re.search(r"\blimit\s+\d+", stripped, re.IGNORECASE):
            return stripped
        return f"{stripped} LIMIT {max_rows}"
