"""Unit tests for statistical test computations against a small synthetic DataFrame."""
import pandas as pd

from src.tools.stats_tool import StatsTool


def make_df():
    dates = pd.date_range("2025-01-01", periods=20, freq="D")
    sales = [100] * 10 + [50] * 10  # obvious drop halfway through
    return pd.DataFrame({"date": dates, "sales": sales, "region": ["A"] * 10 + ["B"] * 10})


def test_period_over_period_change_detects_drop():
    df = make_df()
    tool = StatsTool(df)
    result = tool.run(
        "period_over_period_change",
        date_col="date", metric_col="sales",
        period_a=["2025-01-01", "2025-01-10"],
        period_b=["2025-01-11", "2025-01-20"],
    )
    assert result.success
    assert result.details["pct_change"] < 0
    assert result.details["significant_at_0.05"] is True


def test_trend_test_detects_decreasing_trend():
    df = make_df()
    tool = StatsTool(df)
    result = tool.run("trend_test", date_col="date", metric_col="sales")
    assert result.success
    assert result.details["direction"] == "decreasing"


def test_anova_detects_group_difference():
    df = make_df()
    tool = StatsTool(df)
    result = tool.run("anova", group_col="region", metric_col="sales")
    assert result.success
    assert result.details["significant_at_0.05"] is True


def test_unknown_test_fails_gracefully():
    df = make_df()
    tool = StatsTool(df)
    result = tool.run("not_a_real_test")
    assert not result.success
