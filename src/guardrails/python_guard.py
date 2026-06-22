"""
Static guardrail for the Python Agent's code, using AST inspection: rejects
any import outside an allowlist, any use of dangerous builtins (eval, exec,
open, __import__, compile), and any attribute access on dunder names that
could be used to escape the sandbox (e.g. `().__class__.__bases__`).
"""
import ast
from dataclasses import dataclass
from typing import List

from src.config import settings

_FORBIDDEN_BUILTINS = {"eval", "exec", "open", "__import__", "compile", "input", "exit", "quit"}
_FORBIDDEN_NAME_PREFIXES = ("__",)


@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""


class PythonGuard:
    def __init__(self, allowed_modules: List[str] = None):
        self.allowed_modules = set(allowed_modules or settings.allowed_python_modules)

    def check(self, code: str) -> GuardResult:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return GuardResult(False, f"Syntax error: {e}")

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if not self._module_allowed(alias.name, root):
                        return GuardResult(False, f"Import of '{alias.name}' is not allowed")

            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                root = mod.split(".")[0]
                if not self._module_allowed(mod, root):
                    return GuardResult(False, f"Import from '{mod}' is not allowed")

            elif isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name) and fn.id in _FORBIDDEN_BUILTINS:
                    return GuardResult(False, f"Use of '{fn.id}' is not allowed")
                if isinstance(fn, ast.Attribute) and fn.attr in _FORBIDDEN_BUILTINS:
                    return GuardResult(False, f"Use of '.{fn.attr}' is not allowed")

            elif isinstance(node, ast.Name):
                if any(node.id.startswith(p) and node.id.endswith(p) for p in _FORBIDDEN_NAME_PREFIXES):
                    return GuardResult(False, f"Access to dunder name '{node.id}' is not allowed")

            elif isinstance(node, ast.Attribute):
                if node.attr.startswith("__") and node.attr.endswith("__"):
                    return GuardResult(False, f"Access to dunder attribute '.{node.attr}' is not allowed")

        return GuardResult(True)

    def _module_allowed(self, full_name: str, root: str) -> bool:
        if root in {m.split(".")[0] for m in self.allowed_modules}:
            return True
        return full_name in self.allowed_modules
