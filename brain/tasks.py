"""
Task generation, verification, and lifecycle management.

A task is NEVER marked 'done' without a passing sandbox verification of its
done_check assertion. This is the single invariant this module enforces.
"""

import json
import logging
import re
from typing import Callable

import core.memory as memory
from core.llm import llm_call
from core.sandbox import execute_code

logger = logging.getLogger(__name__)

_TASK_PROMPT = """\
Tu es un assistant IA autonome. Génère entre 1 et 3 tâches concrètes et \
vérifiables liées au contexte suivant :

CONTEXTE : {context}

Réponds UNIQUEMENT avec un tableau JSON valide (sans markdown, sans explication).
Chaque objet doit avoir exactement ces trois champs :
  "title"      : titre court de la tâche (str)
  "description": description de ce qu'il faut faire (str)
  "done_check" : une instruction Python assert() exécutable qui prouve que la
                 tâche est accomplie (str). Exemples valides :
                   assert len("résultat") > 0
                   assert 2 + 2 == 4
                 Le done_check doit être autosuffisant — pas d'import externe.

Ne réponds qu'avec le JSON, rien d'autre.\
"""


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------

def generate_tasks(context: str) -> list[int]:
    """Ask the LLM to propose 1–3 tasks for *context*.

    Each task must include a ``done_check`` assertion. Parsed tasks are
    persisted via :func:`memory.add_task` and their IDs returned.
    Returns an empty list on LLM or parse failure.
    """
    messages = [
        {"role": "system", "content": "Tu es un assistant IA autonome sur Raspberry Pi."},
        {"role": "user", "content": _TASK_PROMPT.format(context=context)},
    ]

    raw = llm_call(messages, use_cloud=False, max_tokens=600)
    if not raw:
        logger.warning("generate_tasks: llm_call returned empty string")
        return []

    # Strip markdown fences the model sometimes adds despite instructions
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    try:
        parsed = json.loads(raw)
        tasks = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError as exc:
        logger.error("generate_tasks: JSON parse error — %s | raw=%r", exc, raw[:200])
        return []

    ids: list[int] = []
    for task in tasks[:3]:
        title = str(task.get("title") or "Sans titre")
        desc = str(task.get("description") or "")
        done_check = str(task.get("done_check") or "")
        tid = memory.add_task(title, desc, done_check)
        ids.append(tid)
        logger.info("Task #%d added: %s", tid, title)

    return ids


# ---------------------------------------------------------------------------
# Done-check verification
# ---------------------------------------------------------------------------

def verify_done_check(task: dict) -> str:
    """Execute ``task['done_check']`` inside the sandbox.

    Returns:
        ``"done"``          — assertion passed; task is verifiably complete.
        ``"in_progress"``   — assertion raised (AssertionError or any error).
        ``"needs_review"``  — no done_check present, or sandbox blocked it.
    """
    done_check = (task.get("done_check") or "").strip()
    if not done_check:
        return "needs_review"

    result = execute_code(done_check)

    if result["blocked"]:
        logger.warning(
            "verify_done_check: sandbox blocked done_check for task #%s",
            task.get("id"),
        )
        return "needs_review"

    if result["error"]:
        return "in_progress"

    return "done"


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_pending_tasks(log: Callable[[str], None]) -> None:
    """Verify done_checks for all pending/in_progress tasks and update statuses.

    A task is marked ``'done'`` only when :func:`verify_done_check` returns
    ``"done"``.  Tasks whose status does not change are left untouched.
    """
    tasks = memory.get_tasks("pending") + memory.get_tasks("in_progress")
    if not tasks:
        log("run_pending_tasks: no active tasks")
        return

    for task in tasks:
        task_id: int = task["id"]
        title: str = task["title"]

        log(f"Verifying task #{task_id}: {title!r}")
        new_status = verify_done_check(task)

        if new_status != task["status"]:
            memory.update_task_status(task_id, new_status)
            log(f"Task #{task_id} '{task['status']}' → '{new_status}'")
        else:
            log(f"Task #{task_id} unchanged: '{new_status}'")


# ---------------------------------------------------------------------------
# Archival
# ---------------------------------------------------------------------------

def archive_done() -> dict:
    """Remove done, failed, and needs_review tasks from the active queue.

    Failed and needs_review tasks are re-marked as ``'done'`` so that
    :func:`memory.archive_done` (which only deletes ``'done'`` rows) can
    sweep them out in a second pass.

    Returns a dict with counts: ``{"done": n, "failed": n, "needs_review": n}``.
    """
    done_count = memory.archive_done()

    failed_tasks = memory.get_tasks("failed")
    for t in failed_tasks:
        memory.update_task_status(t["id"], "done")
    failed_count = memory.archive_done() if failed_tasks else 0

    review_tasks = memory.get_tasks("needs_review")
    for t in review_tasks:
        memory.update_task_status(t["id"], "done")
    review_count = memory.archive_done() if review_tasks else 0

    result = {"done": done_count, "failed": failed_count, "needs_review": review_count}
    logger.info("archive_done: %s", result)
    return result
