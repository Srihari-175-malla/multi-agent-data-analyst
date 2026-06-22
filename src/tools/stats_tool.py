"""
Statistical test execution for the Statistics Agent. Each test takes plain
column names + a pandas DataFrame and returns a structured, JSON-friendly
result — no LLM involved in the computation itself, only in choosing which
test to run and interpreting the result afterward.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


@dataclass
class StatTestResult:
    success: bool
    test_name: str
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    details: Dict[str, Any] = None
    error: Optional[str] = None


class StatsTool:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def run(self, test: str, **kwargs) -> StatTestResult:
        method = getattr(self, f"_{test}", None)
        if method is None:
            return StatTestResult(success=False, test_name=test, error=f"Unknown test '{test}'")
        try:
            return method(**kwargs)
        except Exception as e:
            return StatTestResult(success=False, test_name=test, error=str(e))

    # ---- period_over_period_change: compares mean of a metric between two date ranges
    def _period_over_period_change(self, date_col: str, metric_col: str, period_a: List[str], period_b: List[str]) -> StatTestResult:
        df = self.df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        a = df[(df[date_col] >= period_a[0]) & (df[date_col] <= period_a[1])][metric_col].dropna()
        b = df[(df[date_col] >= period_b[0]) & (df[date_col] <= period_b[1])][metric_col].dropna()
        if len(a) < 2 or len(b) < 2:
            return StatTestResult(success=False, test_name="period_over_period_change",
                                   error="Not enough data points in one or both periods")
        t_stat, p_val = scipy_stats.ttest_ind(a, b, equal_var=False)
        pct_change = ((b.mean() - a.mean()) / a.mean() * 100) if a.mean() != 0 else float("nan")
        return StatTestResult(
            success=True,
            test_name="period_over_period_change",
            statistic=float(t_stat),
            p_value=float(p_val),
            details={
                "period_a_mean": float(a.mean()), "period_b_mean": float(b.mean()),
                "pct_change": float(pct_change), "period_a_n": len(a), "period_b_n": len(b),
                "significant_at_0.05": bool(p_val < 0.05),
            },
        )

    # ---- correlation between two numeric columns
    def _correlation(self, col_a: str, col_b: str, method: str = "pearson") -> StatTestResult:
        a, b = self.df[col_a].dropna(), self.df[col_b].dropna()
        joined = pd.concat([a, b], axis=1, join="inner").dropna()
        if len(joined) < 3:
            return StatTestResult(success=False, test_name="correlation", error="Not enough overlapping data points")
        if method == "spearman":
            corr, p_val = scipy_stats.spearmanr(joined[col_a], joined[col_b])
        else:
            corr, p_val = scipy_stats.pearsonr(joined[col_a], joined[col_b])
        return StatTestResult(
            success=True, test_name="correlation", statistic=float(corr), p_value=float(p_val),
            details={"method": method, "n": len(joined)},
        )

    # ---- trend: linear regression of metric over time, reports slope + significance
    def _trend_test(self, date_col: str, metric_col: str) -> StatTestResult:
        df = self.df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).dropna(subset=[metric_col])
        if len(df) < 3:
            return StatTestResult(success=False, test_name="trend_test", error="Not enough data points")
        x = np.arange(len(df))
        y = df[metric_col].values
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)
        direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
        return StatTestResult(
            success=True, test_name="trend_test", statistic=float(slope), p_value=float(p_value),
            details={"r_squared": float(r_value ** 2), "direction": direction, "n": len(df)},
        )

    # ---- anova across a categorical grouping column
    def _anova(self, group_col: str, metric_col: str) -> StatTestResult:
        groups = [g[metric_col].dropna().values for _, g in self.df.groupby(group_col) if len(g) > 1]
        if len(groups) < 2:
            return StatTestResult(success=False, test_name="anova", error="Need at least 2 groups with data")
        f_stat, p_val = scipy_stats.f_oneway(*groups)
        return StatTestResult(
            success=True, test_name="anova", statistic=float(f_stat), p_value=float(p_val),
            details={"num_groups": len(groups), "significant_at_0.05": bool(p_val < 0.05)},
        )

    # ---- seasonal_decompose: additive decomposition into trend/seasonal/residual
    def _seasonal_decompose(self, date_col: str, metric_col: str, period: int = 7) -> StatTestResult:
        from statsmodels.tsa.seasonal import seasonal_decompose

        df = self.df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).set_index(date_col)[metric_col].dropna()
        if len(df) < period * 2:
            return StatTestResult(success=False, test_name="seasonal_decompose",
                                   error=f"Need at least {period * 2} observations for period={period}")
        result = seasonal_decompose(df, period=period, model="additive", extrapolate_trend="freq")
        return StatTestResult(
            success=True, test_name="seasonal_decompose", details={
                "trend_last_5": [round(v, 3) for v in result.trend.dropna().tail(5).tolist()],
                "seasonal_amplitude": float(result.seasonal.max() - result.seasonal.min()),
                "residual_std": float(result.resid.dropna().std()),
            },
        )

    # ---- changepoint: simple CUSUM-style detection of the largest mean shift point
    def _changepoint_detection(self, date_col: str, metric_col: str) -> StatTestResult:
        df = self.df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).dropna(subset=[metric_col])
        y = df[metric_col].values
        if len(y) < 6:
            return StatTestResult(success=False, test_name="changepoint_detection", error="Not enough data points")

        best_idx, best_score = None, -np.inf
        overall_mean = y.mean()
        cum_dev = np.cumsum(y - overall_mean)
        best_idx = int(np.argmax(np.abs(cum_dev)))
        change_date = df[date_col].iloc[best_idx]
        before_mean = y[:best_idx].mean() if best_idx > 0 else float("nan")
        after_mean = y[best_idx:].mean()
        return StatTestResult(
            success=True, test_name="changepoint_detection",
            details={
                "changepoint_date": str(change_date.date()) if hasattr(change_date, "date") else str(change_date),
                "mean_before": float(before_mean), "mean_after": float(after_mean),
                "shift_pct": float((after_mean - before_mean) / before_mean * 100) if before_mean else None,
            },
        )
