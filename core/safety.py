"""
Two hard safety layers that no other module may bypass or modify.

A) Killswitch  — POST /ks shuts the process down immediately.
B) Allowlist   — gate on which files the agent may auto-modify.
"""

import hmac
import logging
import os
import signal
import threading
from pathlib import Path
from typing import Callable

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# A) KILLSWITCH
# ---------------------------------------------------------------------------

def register_killswitch(app, log_fn: Callable[[str], None]) -> None:
    """Register POST /ks on *app* (a Flask application).

    A correct password triggers a graceful shutdown:
    1. The response is sent to the caller.
    2. 0.6 s later SIGINT is raised in the main thread, stopping the server.

    A wrong password always returns a plain 401 with no diagnostic detail
    (timing-safe comparison prevents oracle attacks).
    """

    @app.route("/ks", methods=["POST"])
    def _killswitch():
        from flask import request, jsonify  # local import — Flask optional dep

        provided = request.json.get("password", "") if request.is_json else ""
        expected = config.KS_PASSWORD

        if not hmac.compare_digest(
            provided.encode("utf-8"),
            expected.encode("utf-8"),
        ):
            return jsonify({"error": "Unauthorized"}), 401

        msg = "KILLSWITCH activated"
        log_fn(msg)
        logger.critical(msg)

        _write_killswitch_log(msg)

        def _shutdown():
            os.kill(os.getpid(), signal.SIGINT)

        threading.Timer(0.6, _shutdown).start()
        return jsonify({"status": "shutting down"}), 200


def _write_killswitch_log(message: str) -> None:
    log_path = Path(config.LOG_DIR) / "killswitch.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            import datetime
            fh.write(f"{datetime.datetime.utcnow().isoformat()} {message}\n")
    except Exception as exc:
        logger.error("Could not write killswitch log: %s", exc)


# ---------------------------------------------------------------------------
# B) AUTO-MODIFICATION ALLOWLIST
# ---------------------------------------------------------------------------

# Files the agent must never touch, regardless of any other rule.
_HARD_BLOCKED: set[str] = {
    "server/app.py",
    "core/safety.py",
    "core/llm.py",
    "config.py",
    ".env",
    "ops/self_update.py",
    "ops/watchdog.py",
    "conftest.py",
}

# Repo root is the directory that contains this file's parent (core/).
_REPO_ROOT = Path(__file__).resolve().parent.parent


def is_modifiable(path: str | Path) -> bool:
    """Return True only when the agent is allowed to auto-modify *path*.

    Rules (all must pass):
    1. No ``..`` components — no path-traversal.
    2. Path resolves inside the repo root.
    3. Not in ``_HARD_BLOCKED``.
    4. Matches one of the permitted patterns:
       - ``brain/*.py``
       - ``memory/*.json``
    """
    raw = str(path)

    # Rule 1 — no traversal
    if ".." in Path(raw).parts:
        return False

    # Rule 2 — must be inside the repo
    try:
        resolved = Path(raw).resolve()
        resolved.relative_to(_REPO_ROOT)
    except ValueError:
        return False

    # Normalise to a repo-relative POSIX string for blocklist/allowlist checks
    try:
        rel = resolved.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return False

    # Rule 3 — hard-blocked files
    if rel in _HARD_BLOCKED:
        return False

    # Rule 4 — permitted patterns only
    p = Path(rel)
    if p.parent.name == "brain" and p.suffix == ".py":
        return True
    if p.parent.name == "memory" and p.suffix == ".json":
        return True

    return False
