"""Unit tests for the report numeric-grounding guardrail."""
from src.guardrails.report_guard import ReportGuard


def test_fully_grounded_report_passes():
    evidence = "Sales dropped from 1200 to 900, a 25% decrease."
    report = "Sales decreased 25%, falling from 1200 to 900."
    result = ReportGuard.check_numeric_grounding(report, evidence)
    assert result.is_fully_grounded


def test_ungrounded_number_is_flagged():
    evidence = "Sales dropped from 1200 to 900."
    report = "Sales decreased by 47%, a massive and unprecedented decline."
    result = ReportGuard.check_numeric_grounding(report, evidence)
    assert not result.is_fully_grounded
    assert "47%" in result.ungrounded_numbers
