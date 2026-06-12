"""
Best-effort sandbox for LLM-generated Python code.
Not a security boundary — a safety net against common mistakes and shell disguises.
"""

import re
import subprocess
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Shell-disguise detector
# ---------------------------------------------------------------------------

_SHELL_PREFIXES = (
    "python", "python3", "pip", "bash", "sh", "cd", "ls", "git", "sudo",
)
_PYTHON_TOKENS = re.compile(r"\b(def|import|print|for|while|class)\b|=")


def _looks_like_shell(code: str) -> bool:
    """Return True when *code* looks like a shell script rather than Python.

    Checks the first non-empty line for shell command prefixes, the ``-m``
    flag pattern, or the total absence of Python tokens in the whole snippet.
    """
    first_line = ""
    for line in code.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            first_line = stripped.lower()
            break

    if not first_line:
        return False

    for prefix in _SHELL_PREFIXES:
        if first_line.startswith(prefix):
            return True

    if first_line.startswith("-m "):
        return True

    if not _PYTHON_TOKENS.search(code):
        return True

    return False


# ---------------------------------------------------------------------------
# Blocklist
# ---------------------------------------------------------------------------

_BLOCKLIST: list[re.Pattern] = [
    re.compile(r"\bos\.system\s*\("),
    re.compile(r"\bsubprocess\.(call|run|Popen|check_output|check_call)\s*\("),
    re.compile(r"\bopen\s*\([^)]*['\"]w['\"]"),          # open(..., 'w') outside sandbox/
    re.compile(r"\b__import__\s*\("),
    re.compile(r"\beval\s*\("),
    re.compile(r"\bexec\s*\("),
    re.compile(r"\bsocket\."),
    re.compile(r"\bshutil\.rmtree\s*\("),
    re.compile(r"rm\s+-rf"),
]


def _is_safe(code: str) -> tuple[bool, Optional[str]]:
    """Check *code* against the blocklist.

    Returns ``(True, None)`` when safe, ``(False, pattern_string)`` on the
    first match.
    """
    for pattern in _BLOCKLIST:
        if pattern.search(code):
            return False, pattern.pattern
    return True, None


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

def execute_code(code: str, timeout: int = 10) -> dict:
    """Run *code* in an isolated subprocess and return a result dict.

    The dict always contains:
    - ``output``  (str) — captured stdout
    - ``error``   (str) — captured stderr or block/timeout message
    - ``blocked`` (bool) — True when execution was refused

    Execution order:
    1. Shell-disguise check → block immediately.
    2. Blocklist check → block immediately.
    3. Run via ``sys.executable -c <code>`` with a hard timeout.
    """
    if _looks_like_shell(code):
        return {
            "output": "",
            "error": (
                "Blocked: code looks like a shell command, not Python. "
                "Only pure Python snippets are executed by the sandbox."
            ),
            "blocked": True,
        }

    safe, bad_pattern = _is_safe(code)
    if not safe:
        return {
            "output": "",
            "error": f"Blocked: forbidden pattern detected — {bad_pattern}",
            "blocked": True,
        }

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "output": result.stdout,
            "error": result.stderr,
            "blocked": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "output": "",
            "error": f"Blocked: execution exceeded {timeout}s timeout.",
            "blocked": True,
        }
    except Exception as exc:
        return {
            "output": "",
            "error": f"Execution error: {exc}",
            "blocked": False,
        }
