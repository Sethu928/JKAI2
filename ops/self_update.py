"""
Controlled self-modification pipeline for JKAI2.

Only brain/*.py files may be modified — enforced by core.safety.is_modifiable.
All core files, including this module, are hard-protected and cannot be patched
by the agent.
"""

import logging
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import config
from core.safety import is_modifiable

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ERROR_PATTERN = re.compile(r"Traceback|\[ERREUR\]|\w+Error:")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], log: Callable, **kwargs) -> subprocess.CompletedProcess:
    """Run *cmd*, log stdout/stderr, return the CompletedProcess."""
    log(f"$ {' '.join(cmd)}")
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(_REPO_ROOT), **kwargs
    )
    if result.stdout.strip():
        log(result.stdout.strip())
    if result.stderr.strip():
        log(result.stderr.strip())
    return result


def _git_push(log: Callable) -> bool:
    """Push to origin, injecting GITHUB_TOKEN into the URL without logging it.

    Returns True on success.
    """
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        url_res = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, cwd=str(_REPO_ROOT),
        )
        remote_url = url_res.stdout.strip()
        if remote_url.startswith("https://"):
            authed_url = remote_url.replace("https://", f"https://{token}@")
            log("$ git push (authenticated)")  # token intentionally omitted
            result = subprocess.run(
                ["git", "push", authed_url],
                capture_output=True, text=True, cwd=str(_REPO_ROOT),
            )
            if result.stderr.strip():
                log(result.stderr.replace(token, "***").strip())
            return result.returncode == 0

    # No token — attempt unauthenticated push (works on SSH remotes or public repos)
    result = _run(["git", "push"], log)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_and_test(path: str, code: str, log: Callable) -> tuple[bool, str]:
    """Validate and write *code* to *path*.

    Refuses immediately if :func:`~core.safety.is_modifiable` returns False.
    Runs ``py_compile`` in a temp file first — the real file is only touched
    when the syntax check passes.

    Returns ``(True, "")`` on success, ``(False, error_message)`` on any failure.
    """
    if not is_modifiable(path):
        msg = f"Refused: {path!r} is protected by the safety allowlist"
        log(msg)
        return False, msg

    # Syntax check in a temp file — never writes to the repo on failure
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write(code)
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", tmp_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            error = result.stderr.replace(tmp_path, path).strip()
            log(f"Syntax error in {path!r}: {error}")
            return False, error
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    target = _REPO_ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(code, encoding="utf-8")
    log(f"Wrote {path!r} ({len(code)} chars)")
    return True, ""


def _run_tests(log: Callable) -> bool:
    """Run the full test suite. Returns True if all tests pass."""
    result = _run([sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"], log)
    passed = result.returncode == 0
    log(f"Tests: {'PASSED' if passed else 'FAILED'}")
    return passed


def _run_rollback(log: Callable) -> bool:
    """Revert HEAD, push, and redeploy the local service.

    Steps:
      1. ``git revert --no-edit HEAD``  — creates a new revert commit
      2. ``git push``                   — with GITHUB_TOKEN if set
      3. ``git pull``                   — sync local checkout
      4. ``systemctl restart jkai2``    — restart service (non-fatal if absent)

    Returns True if steps 1–3 all succeed.
    """
    log("Rollback: starting")

    if _run(["git", "revert", "--no-edit", "HEAD"], log).returncode != 0:
        log("Rollback: git revert failed")
        return False

    if not _git_push(log):
        log("Rollback: git push failed")
        return False

    if _run(["git", "pull"], log).returncode != 0:
        log("Rollback: git pull failed")
        return False

    restart = _run(["systemctl", "restart", "jkai2"], log)
    if restart.returncode != 0:
        log("Rollback: systemctl restart failed (non-fatal — may not be a service)")

    log("Rollback: complete")
    return True


def self_update_cycle(
    path: str,
    code: str,
    message: str,
    log: Callable,
) -> dict:
    """Full self-modification pipeline for a single file.

    Steps:
      1. :func:`write_and_test` — syntax-check then write
      2. ``git add`` + ``git commit`` + push
      3. :func:`_run_tests` — full suite
      4. If tests fail → :func:`_run_rollback`

    Returns a dict: ``{success, rolled_back, error}``.
    """
    ok, error = write_and_test(path, code, log)
    if not ok:
        return {"success": False, "rolled_back": False, "error": error}

    _run(["git", "add", path], log)

    commit = _run(["git", "commit", "-m", message], log)
    if commit.returncode != 0:
        return {"success": False, "rolled_back": False, "error": "git commit failed"}

    if not _git_push(log):
        log("self_update_cycle: push failed — continuing with local tests")

    if _run_tests(log):
        log("self_update_cycle: update successful")
        return {"success": True, "rolled_back": False, "error": ""}

    log("self_update_cycle: tests failed — rolling back")
    rolled_back = _run_rollback(log)
    return {
        "success": False,
        "rolled_back": rolled_back,
        "error": "tests failed after update",
    }


def check_post_update_health(
    log: Callable,
    window_minutes: int = 10,
    max_errors: int = 3,
) -> dict:
    """Scan ``logs/jkai2.log`` for fatal errors in the last *window_minutes* minutes.

    Counted patterns: ``Traceback``, ``[ERREUR]``, ``*Error:``
    Triggers :func:`_run_rollback` when ``error_count >= max_errors``.

    Returns ``{"error_count": int, "rolled_back": bool}``.
    """
    log_path = Path(config.LOG_DIR) / "jkai2.log"
    if not log_path.exists():
        return {"error_count": 0, "rolled_back": False}

    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        log(f"check_post_update_health: cannot read log — {exc}")
        return {"error_count": 0, "rolled_back": False}

    cutoff = datetime.now() - timedelta(minutes=window_minutes)
    error_count = 0

    for line in lines:
        # Log lines start with an ISO timestamp: 2024-01-15T10:30:45.123456
        if len(line) < 19:
            continue
        try:
            ts = datetime.fromisoformat(line[:26])
        except ValueError:
            continue
        if ts < cutoff:
            continue
        if _ERROR_PATTERN.search(line):
            error_count += 1

    rolled_back = False
    if error_count >= max_errors:
        log(
            f"check_post_update_health: {error_count} error(s) in "
            f"{window_minutes}min — triggering rollback"
        )
        rolled_back = _run_rollback(log)

    return {"error_count": error_count, "rolled_back": rolled_back}
