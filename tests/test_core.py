"""
Unit tests for core/ modules — no network, no real LLM.
"""

import pytest


# ===========================================================================
# core.llm — strip_think
# ===========================================================================

class TestStripThink:
    from core.llm import strip_think  # class-level import is fine here

    def test_closed_think_tag(self):
        from core.llm import strip_think
        result = strip_think("before<think>hidden</think>after")
        assert result == "beforeafter"

    def test_closed_thinking_tag(self):
        from core.llm import strip_think
        result = strip_think("A<thinking>cot</thinking>B")
        assert result == "AB"

    def test_open_think_truncates(self):
        from core.llm import strip_think
        result = strip_think("visible<think>unfinished reasoning")
        assert result == "visible"

    def test_passthrough_no_tags(self):
        from core.llm import strip_think
        text = "Hello, world!"
        assert strip_think(text) == text

    def test_case_insensitive_closed(self):
        from core.llm import strip_think
        result = strip_think("x<THINK>secret</THINK>y")
        assert result == "xy"

    def test_case_insensitive_open(self):
        from core.llm import strip_think
        result = strip_think("ok<THINKING>leftover")
        assert result == "ok"

    def test_multiline_closed(self):
        from core.llm import strip_think
        result = strip_think("start\n<think>\nline1\nline2\n</think>\nend")
        assert result == "start\n\nend"

    def test_empty_string(self):
        from core.llm import strip_think
        assert strip_think("") == ""


# ===========================================================================
# core.memory — isolated via tmp_path + monkeypatch
# ===========================================================================

@pytest.fixture()
def mem(tmp_path, monkeypatch):
    """Fresh memory module backed by a temporary SQLite file."""
    import importlib
    import config
    import core.memory as memory

    db_file = str(tmp_path / "test.db")
    monkeypatch.setattr(config, "DB_PATH", db_file)

    # Reset the singleton connection so next call opens the tmp DB
    memory._conn = None
    memory.init_db()

    yield memory

    # Teardown: close and reset singleton
    if memory._conn:
        memory._conn.close()
    memory._conn = None


class TestMemoryDB:
    def test_init_creates_tables(self, mem):
        conn = mem._get_conn()
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cur.fetchall()}
        assert {"conversations", "knowledge", "tasks", "objectives", "metrics", "state"} <= tables

    def test_save_and_load_message(self, mem):
        mem.save_message("user", "Hello")
        mem.save_message("assistant", "Hi there")
        history = mem.load_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"

    def test_count_and_clear_history(self, mem):
        mem.save_message("user", "msg1")
        mem.save_message("user", "msg2")
        assert mem.count_history() == 2
        mem.clear_history()
        assert mem.count_history() == 0

    def test_load_history_limit(self, mem):
        for i in range(10):
            mem.save_message("user", f"msg{i}")
        assert len(mem.load_history(limit=3)) == 3

    def test_add_task_and_update_status(self, mem):
        task_id = mem.add_task("Fix bug", "Reproduce and patch", "tests green")
        tasks = mem.get_tasks()
        assert len(tasks) == 1
        assert tasks[0]["title"] == "Fix bug"
        assert tasks[0]["status"] == "pending"

        mem.update_task_status(task_id, "done")
        tasks = mem.get_tasks("done")
        assert len(tasks) == 1
        assert tasks[0]["status"] == "done"

    def test_get_tasks_filter(self, mem):
        mem.add_task("T1", "", "")
        mem.add_task("T2", "", "")
        tid = mem.add_task("T3", "", "")
        mem.update_task_status(tid, "in_progress")

        assert len(mem.get_tasks("pending")) == 2
        assert len(mem.get_tasks("in_progress")) == 1

    def test_invalid_task_status_raises(self, mem):
        tid = mem.add_task("X", "", "")
        with pytest.raises(ValueError):
            mem.update_task_status(tid, "flying")

    def test_save_and_latest_metrics(self, mem):
        snapshot = {"cpu": 42.5, "ram": 1024, "label": "test"}
        mem.save_metrics(snapshot)
        result = mem.latest_metrics()
        assert result == snapshot

    def test_latest_metrics_empty(self, mem):
        assert mem.latest_metrics() == {}

    def test_set_and_get_state(self, mem):
        mem.set_state("mood", "curious")
        assert mem.get_state("mood") == "curious"

    def test_get_state_default(self, mem):
        assert mem.get_state("nonexistent", default=99) == 99

    def test_set_state_upsert(self, mem):
        mem.set_state("k", "v1")
        mem.set_state("k", "v2")
        assert mem.get_state("k") == "v2"

    def test_state_preserves_types(self, mem):
        mem.set_state("data", {"list": [1, 2, 3], "flag": True})
        result = mem.get_state("data")
        assert result == {"list": [1, 2, 3], "flag": True}


# ===========================================================================
# core.sandbox
# ===========================================================================

class TestSandbox:
    def test_basic_execution(self):
        from core.sandbox import execute_code
        result = execute_code("print(2+2)")
        assert result["blocked"] is False
        assert "4" in result["output"]

    def test_os_system_blocked(self):
        from core.sandbox import execute_code
        result = execute_code("import os; os.system('ls')")
        assert result["blocked"] is True

    def test_shell_command_blocked(self):
        from core.sandbox import execute_code
        result = execute_code("pip install x")
        assert result["blocked"] is True

    def test_subprocess_blocked(self):
        from core.sandbox import execute_code
        result = execute_code("import subprocess; subprocess.run(['ls'])")
        assert result["blocked"] is True

    def test_eval_blocked(self):
        from core.sandbox import execute_code
        result = execute_code("eval('1+1')")
        assert result["blocked"] is True

    def test_exec_blocked(self):
        from core.sandbox import execute_code
        result = execute_code("exec('x=1')")
        assert result["blocked"] is True

    def test_legitimate_python(self):
        from core.sandbox import execute_code
        code = "x = [i**2 for i in range(5)]\nprint(x)"
        result = execute_code(code)
        assert result["blocked"] is False
        assert "[0, 1, 4, 9, 16]" in result["output"]

    def test_syntax_error_not_blocked(self):
        from core.sandbox import execute_code
        result = execute_code("def broken(:")
        assert result["blocked"] is False
        assert result["error"]  # stderr contains the SyntaxError


# ===========================================================================
# core.safety — is_modifiable
# ===========================================================================

class TestIsModifiable:
    def test_brain_file_allowed(self):
        from core.safety import is_modifiable
        assert is_modifiable("brain/agent.py") is True

    def test_brain_consciousness_allowed(self):
        from core.safety import is_modifiable
        assert is_modifiable("brain/consciousness.py") is True

    def test_config_blocked(self):
        from core.safety import is_modifiable
        assert is_modifiable("config.py") is False

    def test_safety_itself_blocked(self):
        from core.safety import is_modifiable
        assert is_modifiable("core/safety.py") is False

    def test_llm_blocked(self):
        from core.safety import is_modifiable
        assert is_modifiable("core/llm.py") is False

    def test_env_blocked(self):
        from core.safety import is_modifiable
        assert is_modifiable(".env") is False

    def test_path_traversal_blocked(self):
        from core.safety import is_modifiable
        assert is_modifiable("../etc/passwd") is False

    def test_core_memory_not_in_allowlist(self):
        from core.safety import is_modifiable
        assert is_modifiable("core/memory.py") is False

    def test_memory_json_allowed(self):
        from core.safety import is_modifiable
        assert is_modifiable("memory/state.json") is True

    def test_tests_dir_not_allowed(self):
        from core.safety import is_modifiable
        assert is_modifiable("tests/test_core.py") is False
