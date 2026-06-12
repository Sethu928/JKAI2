"""
Smoke tests — Flask test client, no real LLM, no network.
"""

import pytest


# ---------------------------------------------------------------------------
# Fixture: patched Flask test client
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Flask test client with LLM stubbed out and an isolated DB."""
    import config
    import core.memory as memory

    # Isolated DB
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "smoke.db"))
    memory._conn = None
    memory.init_db()

    # Stub out LLM calls and the local-server health check
    import server.app as app_module
    monkeypatch.setattr(app_module, "llm_call", lambda *a, **kw: "réponse test")
    monkeypatch.setattr(app_module, "is_alive", lambda *a, **kw: True)

    # Stub agent start/stop so no real threads are spawned
    monkeypatch.setattr(app_module, "start_agent", lambda log: True)
    monkeypatch.setattr(app_module, "stop_agent", lambda log: True)

    # Reset agent flag between tests
    app_module.AGENT_ACTIVE = False

    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c

    if memory._conn:
        memory._conn.close()
    memory._conn = None


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_status_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_required_fields(self, client):
        data = resp = client.get("/health").get_json()
        for field in ("status", "uptime_seconds", "db_ok", "agent_alive", "llm_local_ok"):
            assert field in data, f"missing field: {field}"

    def test_status_ok(self, client):
        data = client.get("/health").get_json()
        assert data["status"] == "ok"

    def test_db_ok_true(self, client):
        data = client.get("/health").get_json()
        assert data["db_ok"] is True

    def test_llm_local_ok_stubbed(self, client):
        data = client.get("/health").get_json()
        assert data["llm_local_ok"] is True


# ---------------------------------------------------------------------------
# /chat
# ---------------------------------------------------------------------------

class TestChat:
    def test_status_200(self, client):
        resp = client.post("/chat", json={"message": "salut"})
        assert resp.status_code == 200

    def test_reply_field_present(self, client):
        data = client.post("/chat", json={"message": "salut"}).get_json()
        assert "reply" in data

    def test_reply_is_string(self, client):
        data = client.post("/chat", json={"message": "salut"}).get_json()
        assert isinstance(data["reply"], str)
        assert len(data["reply"]) > 0

    def test_reply_is_stub(self, client):
        data = client.post("/chat", json={"message": "salut"}).get_json()
        assert data["reply"] == "réponse test"

    def test_missing_message_returns_400(self, client):
        resp = client.post("/chat", json={})
        assert resp.status_code == 400

    def test_empty_message_returns_400(self, client):
        resp = client.post("/chat", json={"message": "  "})
        assert resp.status_code == 400

    def test_chat_saves_to_history(self, client):
        client.post("/chat", json={"message": "bonjour"})
        data = client.get("/history").get_json()
        roles = [m["role"] for m in data["messages"]]
        assert "user" in roles
        assert "assistant" in roles

    def test_llm_down_returns_graceful_message(self, client, monkeypatch):
        import server.app as app_module
        monkeypatch.setattr(app_module, "is_alive", lambda *a, **kw: False)
        data = client.post("/chat", json={"message": "test"}).get_json()
        assert "reply" in data
        assert "indisponible" in data["reply"].lower()


# ---------------------------------------------------------------------------
# /config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_get_returns_200(self, client):
        assert client.get("/config").status_code == 200

    def test_get_flags_present(self, client):
        data = client.get("/config").get_json()
        assert "AGENT_ACTIVE" in data
        assert "AUTO_UPDATE_FROM_CHAT" in data

    def test_defaults_are_false(self, client):
        data = client.get("/config").get_json()
        assert data["AGENT_ACTIVE"] is False
        assert data["AUTO_UPDATE_FROM_CHAT"] is False

    def test_post_updates_agent_active(self, client):
        client.post("/config", json={"AGENT_ACTIVE": True})
        data = client.get("/config").get_json()
        assert data["AGENT_ACTIVE"] is True

    def test_post_unknown_key_returns_400(self, client):
        resp = client.post("/config", json={"UNKNOWN_FLAG": True})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /history
# ---------------------------------------------------------------------------

class TestHistory:
    def test_get_returns_200(self, client):
        assert client.get("/history").status_code == 200

    def test_empty_history(self, client):
        data = client.get("/history").get_json()
        assert data["messages"] == []
        assert data["count"] == 0

    def test_limit_param(self, client):
        for i in range(5):
            client.post("/chat", json={"message": f"msg{i}"})
        data = client.get("/history?limit=3").get_json()
        assert len(data["messages"]) == 3


# ---------------------------------------------------------------------------
# /agent
# ---------------------------------------------------------------------------

class TestAgentControl:
    def test_start_returns_200(self, client):
        resp = client.post("/agent", json={"action": "start"})
        assert resp.status_code == 200

    def test_start_sets_agent_active(self, client):
        client.post("/agent", json={"action": "start"})
        data = client.get("/config").get_json()
        assert data["AGENT_ACTIVE"] is True

    def test_start_response_fields(self, client):
        data = client.post("/agent", json={"action": "start"}).get_json()
        assert "agent_active" in data
        assert "started" in data
        assert "message" in data

    def test_stop_returns_200(self, client):
        client.post("/agent", json={"action": "start"})
        resp = client.post("/agent", json={"action": "stop"})
        assert resp.status_code == 200

    def test_stop_clears_agent_active(self, client):
        client.post("/agent", json={"action": "start"})
        client.post("/agent", json={"action": "stop"})
        data = client.get("/config").get_json()
        assert data["AGENT_ACTIVE"] is False

    def test_stop_response_fields(self, client):
        client.post("/agent", json={"action": "start"})
        data = client.post("/agent", json={"action": "stop"}).get_json()
        assert "agent_active" in data
        assert "stopped" in data
        assert "message" in data

    def test_invalid_action_returns_400(self, client):
        resp = client.post("/agent", json={"action": "restart"})
        assert resp.status_code == 400

    def test_missing_action_returns_400(self, client):
        resp = client.post("/agent", json={})
        assert resp.status_code == 400

    def test_agent_not_started_at_boot(self, client):
        data = client.get("/config").get_json()
        assert data["AGENT_ACTIVE"] is False


# ---------------------------------------------------------------------------
# Import canary
# ---------------------------------------------------------------------------

def test_all_modules_importable():
    """Try importing every project module; list failures without crashing."""
    import importlib

    modules = [
        "config",
        "core.llm",
        "core.memory",
        "core.sandbox",
        "core.safety",
        "brain.agent",
        "brain.tasks",
        "brain.consciousness",
        "brain.metrics",
        "ops.self_update",
        "ops.monitor",
        "ops.watchdog",
        "server.app",
    ]

    failed = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            failed.append(f"{mod}: {exc}")

    if failed:
        pytest.fail("The following modules failed to import:\n" + "\n".join(failed))
