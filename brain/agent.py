"""
Central autonomous agent loop.

Entry-points for the rest of the project:
  start_agent(log)  — launch the worker thread
  stop_agent(log)   — signal it to stop
"""

import hashlib
import json
import logging
import re
import threading
from typing import Callable

import config
import core.memory as memory
from core.llm import is_alive, llm_call
from core.sandbox import execute_code
from brain import tasks as task_module

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action implementations
# ---------------------------------------------------------------------------

def _action_generate_tasks(log: Callable[[str], None]) -> str:
    context = memory.get_state("last_context", "Améliore le projet JKAI2")
    ids = task_module.generate_tasks(str(context))
    msg = f"generate_tasks: {len(ids)} tâche(s) créée(s)"
    log(msg)
    return msg


def _action_run_tasks(log: Callable[[str], None]) -> str:
    task_module.run_pending_tasks(log)
    return "run_tasks: cycle terminé"


def _action_reflect(log: Callable[[str], None]) -> str:
    # Stub — brain/consciousness not yet implemented.
    msg = "reflect: stub (consciousness not yet implemented)"
    log(msg)
    return msg


def _action_save_knowledge(log: Callable[[str], None]) -> str:
    history = memory.load_history(limit=5)
    if not history:
        return "save_knowledge: no recent history to extract from"

    messages = [
        {
            "role": "system",
            "content": (
                "Extrait UN fait concret et utile de cet historique. "
                "Réponds UNIQUEMENT avec : CATEGORY|KEY|VALUE "
                "(une seule ligne, sans explication)."
            ),
        }
    ] + [{"role": m["role"], "content": m["content"]} for m in history]

    raw = llm_call(messages, max_tokens=60).strip()
    if raw and raw.count("|") >= 2:
        cat, key, val = (p.strip() for p in raw.split("|", 2))
        memory.save_knowledge(cat, key, val, source="agent_reflection")
        msg = f"save_knowledge: saved {cat}/{key}"
        log(msg)
        return msg

    return "save_knowledge: nothing extracted"


def _action_idle(log: Callable[[str], None]) -> str:
    msg = "idle: nothing to do this cycle"
    log(msg)
    return msg


# ---------------------------------------------------------------------------
# self_correct
# ---------------------------------------------------------------------------

_SELF_CORRECT_PROMPT = """\
Tu es un agent IA autonome. Une erreur s'est produite et tu dois proposer un correctif.

ERREUR :
{error}

Génère UN correctif. Réponds UNIQUEMENT avec un objet JSON valide (sans markdown) :

Pour exécuter du code correctif dans le sandbox :
  {{"type": "sandbox", "code": "<code Python autosuffisant>"}}

Pour réécrire un fichier brain/*.py :
  {{"type": "file", "path": "brain/<fichier>.py", "content": "<contenu complet du fichier>"}}

Aucun import externe non standard. Aucune commande shell.\
"""

_TRIES_PREFIX = "self_correct_tries:"


def _error_key(error: str) -> str:
    """Stable 12-char key derived from the error message text."""
    return hashlib.sha256(error.encode()).hexdigest()[:12]


def self_correct(error: str, log: Callable[[str], None]) -> str:
    """Attempt to fix *error* automatically.

    Guards:
    - Tries counter per error key; abandons after config.ERROR_MAX_TRIES.
    - Runs the full test suite BEFORE applying any fix; rejects on failure.
    - Applies via sandbox (execute_code) or write_and_test depending on LLM choice.

    Returns a one-line outcome string for the action registry.
    """
    # Lazy import to avoid a circular dependency at module load time.
    from ops.self_update import _run_tests, write_and_test  # noqa: PLC0415

    key = _TRIES_PREFIX + _error_key(error)
    tries: int = memory.get_state(key, 0)

    if tries >= config.ERROR_MAX_TRIES:
        msg = f"[SELF_CORRECT] abandonné après {config.ERROR_MAX_TRIES} tentatives — erreur marquée unsolved"
        log(msg)
        memory.save_knowledge("errors", _error_key(error), error, source="self_correct_unsolved")
        return msg

    memory.set_state(key, tries + 1)
    log(f"[SELF_CORRECT] tentative {tries + 1}/{config.ERROR_MAX_TRIES} pour : {error[:80]}")

    # 1. Generate fix
    messages = [
        {"role": "system", "content": "Tu es un agent IA autonome qui corrige ses propres erreurs."},
        {"role": "user", "content": _SELF_CORRECT_PROMPT.format(error=error)},
    ]
    raw = llm_call(messages, max_tokens=800).strip()
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    try:
        fix = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"[SELF_CORRECT] JSON parse error: {exc}"
        log(msg)
        return msg

    fix_type: str = fix.get("type", "")

    # 2. Pre-flight: tests must pass BEFORE we touch anything
    log("[SELF_CORRECT] vérification pré-application des tests…")
    if not _run_tests(log):
        msg = "[SELF_CORRECT] fix rejeté — tests cassés avant application"
        log(msg)
        memory.set_state("last_error", msg)
        return msg

    # 3. Apply
    if fix_type == "sandbox":
        code = fix.get("code", "")
        if not code:
            return "[SELF_CORRECT] champ 'code' vide"
        result = execute_code(code)
        if result["blocked"]:
            return f"[SELF_CORRECT] sandbox bloqué: {result['error']}"
        if result["error"]:
            return f"[SELF_CORRECT] erreur sandbox: {result['error'][:120]}"
        log(f"[SELF_CORRECT] sandbox OK — output: {result['output'][:120]}")
        memory.set_state(key, 0)  # reset counter on success
        return "[SELF_CORRECT] correctif sandbox appliqué"

    if fix_type == "file":
        path = fix.get("path", "")
        content = fix.get("content", "")
        if not path or not content:
            return "[SELF_CORRECT] champs 'path'/'content' manquants"
        ok, err = write_and_test(path, content, log)
        if not ok:
            return f"[SELF_CORRECT] write_and_test échoué: {err[:120]}"
        memory.set_state(key, 0)  # reset counter on success
        return f"[SELF_CORRECT] fichier {path!r} mis à jour"

    return f"[SELF_CORRECT] type de fix inconnu: {fix_type!r}"


def _action_self_correct(log: Callable[[str], None]) -> str:
    """Registry wrapper: reads last_error from memory and calls self_correct."""
    error = memory.get_state("last_error") or ""
    if not error:
        msg = "self_correct: aucune erreur en mémoire"
        log(msg)
        return msg
    return self_correct(error, log)


# ---------------------------------------------------------------------------
# Action registry
# ---------------------------------------------------------------------------

ACTION_REGISTRY: dict[str, Callable] = {
    "generate_tasks": _action_generate_tasks,
    "run_tasks":      _action_run_tasks,
    "reflect":        _action_reflect,
    "save_knowledge": _action_save_knowledge,
    "self_correct":   _action_self_correct,
    "idle":           _action_idle,
}

# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

_DECIDE_PROMPT = """\
Tu es un agent IA autonome. Choisis UNE action à effectuer ce cycle.

État actuel :
{context}

Actions disponibles : {actions}

Réponds UNIQUEMENT avec le nom exact de l'action (un seul mot), rien d'autre.\
"""


def _decide_action(context: str) -> str:
    """Ask the LLM to choose one action from ACTION_REGISTRY.

    Falls back to ``"idle"`` if the model returns an unrecognised name.
    """
    messages = [
        {"role": "system", "content": "Tu es un agent IA autonome qui choisit ses actions."},
        {"role": "user", "content": _DECIDE_PROMPT.format(
            context=context,
            actions=", ".join(ACTION_REGISTRY),
        )},
    ]
    choice = llm_call(messages, max_tokens=20).strip().lower()
    if choice not in ACTION_REGISTRY:
        logger.warning("_decide_action: unrecognised choice %r — defaulting to idle", choice)
        return "idle"
    return choice


# ---------------------------------------------------------------------------
# Worker cycle
# ---------------------------------------------------------------------------

def _build_context() -> str:
    pending    = len(memory.get_tasks("pending"))
    in_progress = len(memory.get_tasks("in_progress"))
    last_error = memory.get_state("last_error") or "aucune"
    last_action = memory.get_state("last_action") or "aucune"
    return (
        f"Tâches en attente: {pending}, en cours: {in_progress}. "
        f"Dernière action: {last_action}. "
        f"Dernière erreur: {last_error}."
    )


def _run_worker_cycle(log: Callable[[str], None]) -> None:
    """Execute one agent cycle.

    Skips gracefully when the LLM is unreachable so no task state is corrupted.
    """
    if not is_alive():
        log("LLM indisponible, cycle sauté")
        return

    context = _build_context()
    log(f"Cycle — {context}")

    try:
        action_name = _decide_action(context)
        log(f"Action choisie: {action_name}")

        outcome = ACTION_REGISTRY[action_name](log)

        memory.set_state("last_action", action_name)
        memory.set_state("last_outcome", outcome)
        memory.set_state("last_error", None)

        log(f"Cycle terminé — {outcome}")

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error("_run_worker_cycle: %s", error_msg)
        memory.set_state("last_error", error_msg)
        log(f"Erreur cycle: {error_msg}")


# ---------------------------------------------------------------------------
# Thread control
# ---------------------------------------------------------------------------

_agent_thread: threading.Thread | None = None
_stop_event = threading.Event()


def start_agent(log: Callable[[str], None]) -> bool:
    """Launch the worker loop in a daemon thread named ``'agent-main'``.

    Returns True if the agent was started, False if it was already running.
    """
    global _agent_thread

    if _agent_thread and _agent_thread.is_alive():
        log("start_agent: agent already running")
        return False

    _stop_event.clear()

    def _worker_loop() -> None:
        log("Agent démarré")
        while not _stop_event.is_set():
            try:
                _run_worker_cycle(log)
            except Exception as exc:
                logger.error("worker_loop unhandled: %s", exc)
            # Use wait() instead of sleep() so stop_agent() wakes us immediately.
            _stop_event.wait(timeout=config.WORKER_INTERVAL)
        log("Agent arrêté")

    _agent_thread = threading.Thread(
        target=_worker_loop, name="agent-main", daemon=True
    )
    _agent_thread.start()
    return True


def stop_agent(log: Callable[[str], None]) -> bool:
    """Signal the agent loop to stop after the current cycle.

    Returns True if a running agent was signalled, False if none was running.
    """
    if not _agent_thread or not _agent_thread.is_alive():
        log("stop_agent: no agent running")
        return False
    _stop_event.set()
    log("Signal d'arrêt envoyé à l'agent")
    return True
