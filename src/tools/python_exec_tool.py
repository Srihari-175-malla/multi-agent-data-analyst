"""
Sandboxed Python execution for the Python Agent.

Safety layers:
1. AST guardrail (src/guardrails/python_guard.py) rejects disallowed imports/builtins
   before anything is executed.
2. Execution happens in a separate process with a wall-clock timeout, so an
   infinite loop or heavy computation can't hang the whole system.
3. The execution namespace only exposes an allowlisted set of libraries plus
   the dataset as `df` — no filesystem/network access beyond saving charts to
   a fixed output directory.

Convention the agent must follow: assign final findings to a variable named
`result` (any JSON-serializable-ish value, printed via repr) and save any
chart with `plt.savefig(path)` where `path` is under the provided chart_dir.
"""
import io
import multiprocessing as mp
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

from src.config import settings
from src.guardrails.python_guard import PythonGuard
from src.utils_logging import get_logger

logger = get_logger(__name__)


@dataclass
class PythonExecResult:
    success: bool
    stdout: str = ""
    result_repr: Optional[str] = None
    chart_paths: List[str] = field(default_factory=list)
    error: str = None


def _worker(code: str, df: pd.DataFrame, chart_dir: str, queue: mp.Queue):
    import math
    import statistics
    import json
    import datetime

    import numpy as np
    import scipy
    import scipy.stats
    import statsmodels.api as sm
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    safe_globals = {
        "__builtins__": {
            # a minimal, safe builtins subset
            "len": len, "range": range, "enumerate": enumerate, "zip": zip,
            "sum": sum, "min": min, "max": max, "sorted": sorted, "abs": abs,
            "round": round, "list": list, "dict": dict, "set": set, "tuple": tuple,
            "str": str, "int": int, "float": float, "bool": bool, "print": print,
            "isinstance": isinstance, "map": map, "filter": filter, "any": any, "all": all,
        },
        "pd": pd, "np": np, "math": math, "statistics": statistics,
        "scipy": scipy, "stats": scipy.stats, "sm": sm,
        "plt": plt, "json": json, "datetime": datetime,
        "df": df, "chart_dir": chart_dir,
    }
    local_vars: dict = {}
    stdout_buffer = io.StringIO()
    try:
        with redirect_stdout(stdout_buffer):
            exec(code, safe_globals, local_vars)
        result_val = local_vars.get("result")
        if hasattr(result_val, "item") and hasattr(result_val, "ndim") and getattr(result_val, "ndim", None) == 0:
            # unwrap 0-d numpy/pandas scalars (e.g. np.int64(15)) to plain Python types
            try:
                result_val = result_val.item()
            except Exception:
                pass
        chart_paths = []
        for fig_num in plt.get_fignums():
            path = Path(chart_dir) / f"chart_{fig_num}_{id(code) % 100000}.png"
            plt.figure(fig_num).savefig(path, bbox_inches="tight")
            chart_paths.append(str(path))
        plt.close("all")
        queue.put(
            {
                "success": True,
                "stdout": stdout_buffer.getvalue(),
                "result_repr": repr(result_val) if result_val is not None else None,
                "chart_paths": chart_paths,
                "error": None,
            }
        )
    except Exception:
        queue.put(
            {
                "success": False,
                "stdout": stdout_buffer.getvalue(),
                "result_repr": None,
                "chart_paths": [],
                "error": traceback.format_exc(limit=3),
            }
        )


class PythonExecTool:
    def __init__(self, df: pd.DataFrame, chart_dir: str = None, timeout: int = None):
        self.df = df
        self.chart_dir = str(chart_dir or settings.charts_dir)
        Path(self.chart_dir).mkdir(parents=True, exist_ok=True)
        self.timeout = timeout or settings.python_exec_timeout_seconds
        self.guard = PythonGuard()

    def execute(self, code: str) -> PythonExecResult:
        guard_result = self.guard.check(code)
        if not guard_result.allowed:
            logger.warning(f"Python guard blocked code: {guard_result.reason}")
            return PythonExecResult(success=False, error=f"Blocked by guardrail: {guard_result.reason}")

        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        process = ctx.Process(target=_worker, args=(code, self.df, self.chart_dir, queue))
        process.start()
        process.join(timeout=self.timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            return PythonExecResult(success=False, error=f"Execution timed out after {self.timeout}s")

        if queue.empty():
            return PythonExecResult(success=False, error="Process exited without returning a result (possible crash)")

        payload = queue.get()
        return PythonExecResult(**payload)
