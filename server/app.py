"""
Flask entry-point for JKAI2.
Provides /chat, /health, /config, /history, /agent and the killswitch route.
"""

import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

import config
import core.memory as memory
from core.llm import is_alive, llm_call
from core.safety import register_killswitch
from brain.agent import start_agent, stop_agent

# ---------------------------------------------------------------------------
# App & globals
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

AGENT_ACTIVE: bool = False
AUTO_UPDATE_FROM_CHAT: bool = False
START_TIME: datetime = datetime.now()

SYSTEM_PROMPT = (
    "Tu es J-KAI, un assistant IA personnel qui tourne en local. "
    "Tu réponds toujours en français, de façon concise et directe. "
    "Tu tutoies l'utilisateur."
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log_lock = threading.Lock()


def log(message: str) -> None:
    """Append a timestamped line to logs/jkai2.log and stdout."""
    line = f"{datetime.now().isoformat()} {message}"
    print(line)
    log_path = Path(config.LOG_DIR) / "jkai2.log"
    try:
        with _log_lock:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception as exc:
        print(f"[log error] {exc}")


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

config.ensure_dirs()
memory.init_db()
START_TIME = datetime.now()
register_killswitch(app, log)
log("JKAI2 server initialising")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/chat", methods=["POST"])
def chat():
    global AUTO_UPDATE_FROM_CHAT

    data = request.get_json(silent=True) or {}
    user_message: str = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "message field required"}), 400

    if not is_alive():
        log("LLM local unreachable — returning fallback message")
        return jsonify({
            "reply": "Le LLM local est actuellement indisponible. "
                     "Vérifie que LM Studio tourne et est accessible."
        })

    history = memory.load_history(limit=10)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for entry in history:
        messages.append({"role": entry["role"], "content": entry["content"]})
    messages.append({"role": "user", "content": user_message})

    reply = llm_call(messages, use_cloud=False)

    memory.save_message("user", user_message)
    memory.save_message("assistant", reply)

    log(f"chat — user: {user_message[:60]!r}  reply: {reply[:60]!r}")
    return jsonify({"reply": reply})


@app.route("/health", methods=["GET"])
def health():
    uptime = (datetime.now() - START_TIME).total_seconds()

    try:
        db_ok = memory.count_history() >= 0
    except Exception:
        db_ok = False

    agent_alive = any(
        t.name == "agent-main" for t in threading.enumerate()
    )

    return jsonify({
        "status": "ok",
        "uptime_seconds": round(uptime, 1),
        "db_ok": db_ok,
        "agent_alive": agent_alive,
        "llm_local_ok": is_alive(),
    })


@app.route("/config", methods=["GET", "POST"])
def config_route():
    global AGENT_ACTIVE, AUTO_UPDATE_FROM_CHAT

    if request.method == "GET":
        return jsonify({
            "AGENT_ACTIVE": AGENT_ACTIVE,
            "AUTO_UPDATE_FROM_CHAT": AUTO_UPDATE_FROM_CHAT,
        })

    data = request.get_json(silent=True) or {}
    changed = {}

    if "AGENT_ACTIVE" in data:
        AGENT_ACTIVE = bool(data["AGENT_ACTIVE"])
        changed["AGENT_ACTIVE"] = AGENT_ACTIVE
        log(f"config — AGENT_ACTIVE set to {AGENT_ACTIVE}")

    if "AUTO_UPDATE_FROM_CHAT" in data:
        AUTO_UPDATE_FROM_CHAT = bool(data["AUTO_UPDATE_FROM_CHAT"])
        changed["AUTO_UPDATE_FROM_CHAT"] = AUTO_UPDATE_FROM_CHAT
        log(f"config — AUTO_UPDATE_FROM_CHAT set to {AUTO_UPDATE_FROM_CHAT}")

    if not changed:
        return jsonify({"error": "No recognised keys. Use AGENT_ACTIVE or AUTO_UPDATE_FROM_CHAT"}), 400

    return jsonify({"updated": changed})


@app.route("/agent", methods=["POST"])
def agent_control():
    global AGENT_ACTIVE

    data = request.get_json(silent=True) or {}
    action = data.get("action", "").strip().lower()

    if action == "start":
        AGENT_ACTIVE = True
        started = start_agent(log)
        return jsonify({
            "agent_active": AGENT_ACTIVE,
            "started": started,
            "message": "Agent démarré" if started else "Agent déjà en cours",
        })

    if action == "stop":
        AGENT_ACTIVE = False
        stopped = stop_agent(log)
        return jsonify({
            "agent_active": AGENT_ACTIVE,
            "stopped": stopped,
            "message": "Signal d'arrêt envoyé" if stopped else "Aucun agent en cours",
        })

    return jsonify({"error": "action must be 'start' or 'stop'"}), 400


@app.route("/history", methods=["GET"])
def history():
    limit = min(int(request.args.get("limit", 50)), 200)
    entries = memory.load_history(limit=limit)
    return jsonify({"messages": entries, "count": len(entries)})


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log("JKAI2 server starting on 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
