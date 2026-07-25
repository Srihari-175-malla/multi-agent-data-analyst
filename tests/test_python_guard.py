"""Unit tests for the Python sandbox guardrail."""
from src.guardrails.python_guard import PythonGuard


def make_guard():
    return PythonGuard(allowed_modules=["pandas", "numpy", "math"])


def test_allows_whitelisted_import():
    guard = make_guard()
    assert guard.check("import pandas as pd\nresult = 1").allowed


def test_blocks_os_import():
    guard = make_guard()
    result = guard.check("import os\nos.system('ls')")
    assert not result.allowed


def test_blocks_eval():
    guard = make_guard()
    result = guard.check("result = eval('1+1')")
    assert not result.allowed


def test_blocks_open():
    guard = make_guard()
    result = guard.check("f = open('/etc/passwd')")
    assert not result.allowed


def test_blocks_dunder_escape():
    guard = make_guard()
    result = guard.check("x = ().__class__.__bases__")
    assert not result.allowed


def test_blocks_import_from_disallowed_module():
    guard = make_guard()
    result = guard.check("from subprocess import call")
    assert not result.allowed
