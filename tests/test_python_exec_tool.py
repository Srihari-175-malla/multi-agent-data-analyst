"""
Unit tests for the sandboxed Python execution tool. These exercise the real
subprocess execution path (no mocking) since correctness of the sandbox
boundary is the entire point of this tool.
"""
import pandas as pd

from src.tools.python_exec_tool import PythonExecTool


def make_df():
    return pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})


def test_execute_simple_computation():
    tool = PythonExecTool(make_df(), timeout=10)
    result = tool.execute("result = df['x'].sum()")
    assert result.success
    assert result.result_repr == "15"


def test_execute_blocked_import_never_runs():
    tool = PythonExecTool(make_df(), timeout=10)
    result = tool.execute("import os\nresult = os.getcwd()")
    assert not result.success
    assert "guardrail" in result.error.lower()


def test_execute_timeout_is_enforced():
    tool = PythonExecTool(make_df(), timeout=2)
    result = tool.execute("while True:\n    pass")
    assert not result.success
    assert "timed out" in result.error.lower()
